from __future__ import annotations

import operator
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    import p800_flat_gather_native
except ImportError:
    p800_flat_gather_native = None


_execution_counter_lock = threading.Lock()
_forward_native_eligible_calls = 0
_PLAN_FACTORY_TOKEN = object()


def _require_native_extension():
    if p800_flat_gather_native is None:
        raise RuntimeError(
            "p800_flat_gather_native is unavailable; build the XTDK extension first"
        )
    return p800_flat_gather_native


def _require_native_counter_api():
    native = _require_native_extension()
    reset = getattr(native, "reset_backward_execution_count", None)
    query = getattr(native, "get_backward_execution_count", None)
    if not callable(reset) or not callable(query):
        raise RuntimeError(
            "p800_flat_gather_native lacks the backward execution counter API"
        )
    return reset, query


def reset_native_backward_execution_count() -> int:
    """Reset the native counter and return its value before the reset.

    This API fails closed when the extension is missing or too old. Only reset
    it in an isolated process with no operator calls in flight.
    """

    reset_native, _ = _require_native_counter_api()
    return int(reset_native())


def get_native_backward_execution_count() -> int:
    """Read successful non-empty native backward enqueues in this process."""

    _, query_native = _require_native_counter_api()
    return int(query_native())


def reset_flat_gather_csr_execution_counters() -> Dict[str, int]:
    """Reset and return the process-local execution-proof counters."""

    global _forward_native_eligible_calls
    prior_native = reset_native_backward_execution_count()
    with _execution_counter_lock:
        prior_forward = int(_forward_native_eligible_calls)
        _forward_native_eligible_calls = 0
    return {
        "forward_native_eligible_calls": prior_forward,
        "native_backward_executed_calls": prior_native,
    }


def get_flat_gather_csr_execution_counters() -> Dict[str, int]:
    """Return process-local execution-proof counters.

    XPU launches are asynchronous. Call ``torch.cuda.synchronize()`` before
    treating ``native_backward_executed_calls`` as completed device work.
    """

    with _execution_counter_lock:
        forward_eligible = int(_forward_native_eligible_calls)
    return {
        "forward_native_eligible_calls": forward_eligible,
        "native_backward_executed_calls": get_native_backward_execution_count(),
    }


def _record_forward_native_eligible() -> None:
    global _forward_native_eligible_calls
    with _execution_counter_lock:
        _forward_native_eligible_calls += 1


def _normalize_index_shape(index: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if not isinstance(index, torch.Tensor):
        raise TypeError("index must be a torch.Tensor")
    if index.dtype != torch.int64:
        raise TypeError(f"index must have dtype torch.int64, got {index.dtype}")
    if index.ndim < 1:
        raise ValueError("index must have at least one dimension")

    normalized = index
    if normalized.ndim >= 3 and normalized.shape[0] == 1:
        normalized = normalized.squeeze(0)
    normalized = normalized.contiguous()
    return normalized, tuple(int(size) for size in normalized.shape)


class _FlatGatherCSRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, flat_index, native_plan, index_shape):
        ctx.native_plan = native_plan
        ctx.num_rows = int(values.shape[1])
        ctx.channels = int(values.shape[2])
        return values.index_select(1, flat_index).reshape(
            1, *index_shape, values.shape[2]
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_edges = grad_output.contiguous().reshape(-1, ctx.channels)
        native = _require_native_extension()
        grad_values = native.backward(grad_edges, ctx.native_plan)
        return grad_values, None, None, None


class FlatGatherCSRPlan:
    """Owned reverse-CSR plan for one static gather index.

    Construct plans with :func:`prepare_flat_gather_csr`. The plan owns copies
    of the index and CSR tensors, so later mutation of the caller's original
    index does not change the plan. Tensor inspection properties return copies;
    the validated native CSR storage is held by an opaque extension object.
    """

    __slots__ = (
        "_flat_index",
        "_row_ptr",
        "_edge_ids",
        "_index_shape",
        "_num_values",
        "_native_plan",
        "_tensor_versions",
    )

    def __setattr__(self, name, value) -> None:
        if hasattr(self, name):
            raise AttributeError("FlatGatherCSRPlan attributes are read-only")
        object.__setattr__(self, name, value)

    def __delattr__(self, name) -> None:
        raise AttributeError("FlatGatherCSRPlan attributes are read-only")

    def __init__(
        self,
        flat_index: torch.Tensor,
        row_ptr: torch.Tensor,
        edge_ids: torch.Tensor,
        index_shape: Tuple[int, ...],
        num_values: int,
        *,
        _factory_token=None,
    ) -> None:
        if _factory_token is not _PLAN_FACTORY_TOKEN:
            raise TypeError(
                "FlatGatherCSRPlan cannot be constructed directly; "
                "use prepare_flat_gather_csr()"
            )
        self._flat_index = flat_index
        self._row_ptr = row_ptr
        self._edge_ids = edge_ids
        self._index_shape = tuple(index_shape)
        self._num_values = int(num_values)
        self._validate_structure()
        self._native_plan = self._prepare_native_plan()
        self._tensor_versions = self._current_tensor_versions()

    def __repr__(self) -> str:
        return (
            "FlatGatherCSRPlan("
            f"index_shape={self.index_shape}, num_values={self.num_values}, "
            f"edges={self._flat_index.numel()}, device={self.device})"
        )

    @property
    def flat_index(self) -> torch.Tensor:
        return self._flat_index.clone()

    @property
    def row_ptr(self) -> torch.Tensor:
        return self._row_ptr.clone()

    @property
    def edge_ids(self) -> torch.Tensor:
        return self._edge_ids.clone()

    @property
    def index_shape(self) -> Tuple[int, ...]:
        return self._index_shape

    @property
    def num_values(self) -> int:
        return self._num_values

    @property
    def device(self) -> torch.device:
        return self._flat_index.device

    def _current_tensor_versions(self) -> Tuple[int, int, int]:
        return (
            int(self._flat_index._version),
            int(self._row_ptr._version),
            int(self._edge_ids._version),
        )

    def _validate_structure(self) -> None:
        expected_edges = int(np.prod(self._index_shape, dtype=np.int64))
        if self._flat_index.dtype != torch.int64:
            raise TypeError("plan flat_index must be int64")
        if self._row_ptr.dtype != torch.int32:
            raise TypeError("plan row_ptr must be int32")
        if self._edge_ids.dtype != torch.int32:
            raise TypeError("plan edge_ids must be int32")
        if not all(
            tensor.is_contiguous()
            for tensor in (self._flat_index, self._row_ptr, self._edge_ids)
        ):
            raise ValueError("all plan tensors must be contiguous")
        if not (
            self._flat_index.device == self._row_ptr.device == self._edge_ids.device
        ):
            raise ValueError("all plan tensors must be on the same device")
        if self._flat_index.ndim != 1 or self._edge_ids.ndim != 1:
            raise ValueError("flat_index and edge_ids must be one-dimensional")
        if self._row_ptr.ndim != 1:
            raise ValueError("row_ptr must be one-dimensional")
        if self._flat_index.numel() != expected_edges:
            raise ValueError("flat_index length does not match index_shape")
        if self._edge_ids.numel() != expected_edges:
            raise ValueError("edge_ids length does not match flat_index")
        if self._row_ptr.numel() != self._num_values + 1:
            raise ValueError("row_ptr length must equal num_values + 1")

    def _prepare_native_plan(self):
        if not self._flat_index.is_cuda or p800_flat_gather_native is None:
            return None
        prepare_plan = getattr(p800_flat_gather_native, "prepare_plan", None)
        if not callable(prepare_plan):
            return None
        return prepare_plan(self._row_ptr, self._edge_ids, self._num_values)

    def _assert_integrity(self) -> None:
        if self._current_tensor_versions() != self._tensor_versions:
            raise RuntimeError(
                "FlatGatherCSRPlan tensors were modified in place; rebuild the plan"
            )

    def native_rejection_reason(self, values: torch.Tensor) -> Optional[str]:
        """Return ``None`` when the native path supports ``values``."""

        self._assert_integrity()
        if not isinstance(values, torch.Tensor):
            return "values must be a torch.Tensor"
        if p800_flat_gather_native is None:
            return "the p800_flat_gather_native extension is unavailable"
        if not callable(getattr(p800_flat_gather_native, "backward", None)):
            return "the native extension lacks the backward API"
        if self._native_plan is None:
            return "the native extension lacks the validated plan API"
        if values.ndim != 3:
            return f"values must have shape [B, N, C], got ndim={values.ndim}"
        if values.shape[0] != 1:
            return f"native execution requires B=1, got B={values.shape[0]}"
        if values.shape[1] != self.num_values:
            return (
                "values.shape[1] must match plan.num_values, got "
                f"{values.shape[1]} and {self.num_values}"
            )
        if values.dtype != torch.float32:
            return f"native execution requires float32 values, got {values.dtype}"
        if not values.is_cuda:
            return "values must be on the xpytorch P800 CUDA-compatible device"
        if values.device != self.device:
            return (
                "values and plan must share a device, got "
                f"{values.device} and {self.device}"
            )
        if not values.is_contiguous():
            return "native execution requires contiguous values"
        channels = int(values.shape[2])
        if not (0 < channels <= 128 and channels % 16 == 0):
            return (
                "native execution requires channels in {16, 32, ..., 128}, "
                f"got {channels}"
            )
        return None

    def native_supported(self, values: torch.Tensor) -> bool:
        return self.native_rejection_reason(values) is None

    def __call__(
        self,
        values: torch.Tensor,
        *,
        require_native: bool = False,
    ) -> torch.Tensor:
        self._assert_integrity()
        if not isinstance(require_native, bool):
            raise TypeError("require_native must be bool")
        reason = self.native_rejection_reason(values)
        if reason is None:
            _record_forward_native_eligible()
            return _FlatGatherCSRFunction.apply(
                values,
                self._flat_index,
                self._native_plan,
                self._index_shape,
            )
        if require_native:
            raise RuntimeError(
                f"flat_gather_csr native path rejected the input: {reason}"
            )
        return values.index_select(1, self._flat_index.to(values.device)).reshape(
            values.shape[0], *self._index_shape, values.shape[-1]
        )


def prepare_flat_gather_csr(
    index: torch.Tensor,
    num_values: int,
    device: Optional[torch.device] = None,
) -> FlatGatherCSRPlan:
    """Build a reverse-CSR plan for an index that will be reused across steps."""

    try:
        normalized_num_values = operator.index(num_values)
    except TypeError as error:
        raise TypeError("num_values must be an integer") from error
    if isinstance(num_values, bool):
        raise TypeError("num_values must be an integer, not bool")
    if normalized_num_values < 0:
        raise ValueError("num_values must be non-negative")
    if normalized_num_values > np.iinfo(np.int32).max:
        raise ValueError("initial native plan requires num_values to fit int32")

    normalized, index_shape = _normalize_index_shape(index)
    index_cpu = normalized.detach().to(device="cpu").contiguous().clone()
    flat_numpy = index_cpu.reshape(-1).numpy()
    if flat_numpy.size:
        minimum = int(flat_numpy.min())
        maximum = int(flat_numpy.max())
        if minimum < 0 or maximum >= normalized_num_values:
            raise IndexError(
                f"index range [{minimum}, {maximum}] is outside "
                f"[0, {normalized_num_values})"
            )
    if flat_numpy.size > np.iinfo(np.int32).max:
        raise ValueError("initial native plan supports fewer than 2^31 gather edges")

    counts = np.bincount(flat_numpy, minlength=normalized_num_values)
    cumulative = np.cumsum(counts, dtype=np.int64)
    if cumulative.size and cumulative[-1] > np.iinfo(np.int32).max:
        raise ValueError("initial native plan requires int32 CSR offsets")
    row_ptr_numpy = np.empty(normalized_num_values + 1, dtype=np.int32)
    row_ptr_numpy[0] = 0
    row_ptr_numpy[1:] = cumulative.astype(np.int32, copy=False)
    edge_ids_numpy = np.argsort(flat_numpy, kind="stable").astype(np.int32, copy=False)

    target = torch.device("cpu") if device is None else torch.device(device)
    flat_index = index_cpu.reshape(-1).to(target).contiguous()
    row_ptr = torch.from_numpy(row_ptr_numpy).to(target).contiguous()
    edge_ids = torch.from_numpy(edge_ids_numpy).to(target).contiguous()
    return FlatGatherCSRPlan(
        flat_index,
        row_ptr,
        edge_ids,
        index_shape,
        normalized_num_values,
        _factory_token=_PLAN_FACTORY_TOKEN,
    )


def flat_gather_csr(
    values: torch.Tensor,
    plan: FlatGatherCSRPlan,
    *,
    require_native: bool = False,
) -> torch.Tensor:
    """Gather with native reverse-CSR backward when the contract is satisfied.

    Set ``require_native=True`` in benchmarks and acceptance tests so an
    unsupported input cannot silently measure the stock fallback.
    """

    if not isinstance(plan, FlatGatherCSRPlan):
        raise TypeError("plan must be a FlatGatherCSRPlan")
    return plan(values, require_native=require_native)
