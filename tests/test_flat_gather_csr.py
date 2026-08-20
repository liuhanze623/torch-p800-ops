from __future__ import annotations

import importlib
import unittest

import torch

from p800_ops import (
    FlatGatherCSRPlan,
    flat_gather_csr,
    prepare_flat_gather_csr,
)
from p800_ops.flat_gather_csr import (
    get_flat_gather_csr_execution_counters,
    get_native_backward_execution_count,
    reset_flat_gather_csr_execution_counters,
    reset_native_backward_execution_count,
)

try:
    import p800_flat_gather_native
except ImportError:
    p800_flat_gather_native = None


HAS_P800_NATIVE = bool(
    p800_flat_gather_native is not None and torch.cuda.is_available()
)


def _normalize_index_shape(index):
    normalized = index
    if normalized.ndim >= 3 and normalized.shape[0] == 1:
        normalized = normalized.squeeze(0)
    normalized = normalized.contiguous()
    return normalized, tuple(int(size) for size in normalized.shape)


def _advanced_gather(values, index):
    normalized, index_shape = _normalize_index_shape(index)
    device_index = normalized.to(values.device)
    return values[:, device_index, :].reshape(
        values.shape[0], *index_shape, values.shape[-1]
    )


def _index_select_gather(values, index):
    normalized, index_shape = _normalize_index_shape(index)
    flat_index = normalized.reshape(-1).to(values.device)
    return values.index_select(1, flat_index).reshape(
        values.shape[0], *index_shape, values.shape[-1]
    )


def _run_and_grad(function, values, upstream):
    candidate = values.detach().clone().requires_grad_(True)
    output = function(candidate)
    output.backward(upstream)
    torch.cuda.synchronize()
    return output.detach(), candidate.grad.detach()


class PlanAndFallbackTests(unittest.TestCase):
    def test_plan_owns_an_index_copy(self):
        index = torch.tensor([[2, 0], [2, 1]], dtype=torch.int64)
        original = index.clone()
        plan = prepare_flat_gather_csr(index, num_values=3)
        index.zero_()

        values = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        expected = _index_select_gather(values, original)
        actual = flat_gather_csr(values, plan)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_plan_inspection_tensors_are_defensive_copies(self):
        index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        plan = prepare_flat_gather_csr(index, num_values=3)
        exposed_index = plan.flat_index
        exposed_index.data[0] = 2
        exposed_row_ptr = plan.row_ptr
        exposed_row_ptr.numpy()[1] = 99
        exposed_edge_ids = plan.edge_ids
        exposed_edge_ids.zero_()

        values = torch.randn(1, 3, 4)
        expected = _index_select_gather(values, index)
        actual = flat_gather_csr(values, plan)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_plan_cannot_be_constructed_directly(self):
        index = torch.tensor([0], dtype=torch.int64)
        row_ptr = torch.tensor([0, 1], dtype=torch.int32)
        edge_ids = torch.tensor([0], dtype=torch.int32)
        with self.assertRaisesRegex(TypeError, "prepare_flat_gather_csr"):
            FlatGatherCSRPlan(index, row_ptr, edge_ids, (1,), 1)

    def test_non_int64_indexes_are_rejected(self):
        for dtype in (torch.int32, torch.float32, torch.bool):
            with self.subTest(dtype=dtype):
                index = torch.tensor([[0, 1]], dtype=dtype)
                with self.assertRaisesRegex(TypeError, "torch.int64"):
                    prepare_flat_gather_csr(index, num_values=2)

    def test_out_of_range_indexes_are_rejected(self):
        for index in (
            torch.tensor([[0, -1]], dtype=torch.int64),
            torch.tensor([[0, 3]], dtype=torch.int64),
        ):
            with self.subTest(index=index.tolist()):
                with self.assertRaises(IndexError):
                    prepare_flat_gather_csr(index, num_values=3)

    def test_num_values_contract_is_strict(self):
        index = torch.empty((0, 2), dtype=torch.int64)
        with self.assertRaisesRegex(TypeError, "not bool"):
            prepare_flat_gather_csr(index, num_values=True)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            prepare_flat_gather_csr(index, num_values=-1)
        with self.assertRaisesRegex(ValueError, "fit int32"):
            prepare_flat_gather_csr(index, num_values=2**31)

    def test_leading_singleton_index_dimension_is_normalized(self):
        index = torch.tensor([[[2, 0], [1, 2]]], dtype=torch.int64)
        plan = prepare_flat_gather_csr(index, num_values=3)
        self.assertEqual(plan.index_shape, (2, 2))
        values = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        self.assertEqual(tuple(flat_gather_csr(values, plan).shape), (1, 2, 2, 2))

    def test_cpu_fallback_and_require_native(self):
        index = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
        plan = prepare_flat_gather_csr(index, num_values=3)
        values = torch.randn(2, 3, 5, dtype=torch.float32)
        expected = _index_select_gather(values, index)
        actual = flat_gather_csr(values, plan)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        with self.assertRaisesRegex(RuntimeError, "native path rejected"):
            flat_gather_csr(values, plan, require_native=True)


@unittest.skipUnless(
    HAS_P800_NATIVE,
    "requires the built p800_flat_gather_native extension and P800 runtime",
)
class P800NativeTests(unittest.TestCase):
    def setUp(self):
        reset_flat_gather_csr_execution_counters()

    def test_forward_backward_matches_stock_with_repeated_indexes(self):
        generator = torch.Generator().manual_seed(0)
        num_values, queries, neighbors, channels = 257, 263, 12, 80
        index_cpu = torch.randint(
            num_values,
            (queries, neighbors),
            generator=generator,
            dtype=torch.int64,
        )
        values = torch.randn(1, num_values, channels, device="cuda")
        upstream = torch.randn(1, queries, neighbors, channels, device="cuda")
        index_xpu = index_cpu.to("cuda")
        plan = prepare_flat_gather_csr(index_cpu, num_values, device=values.device)
        self.assertTrue(plan.native_supported(values))

        advanced_output, advanced_grad = _run_and_grad(
            lambda tensor: _advanced_gather(tensor, index_xpu), values, upstream
        )
        select_output, select_grad = _run_and_grad(
            lambda tensor: _index_select_gather(tensor, index_xpu), values, upstream
        )
        csr_output, csr_grad = _run_and_grad(
            lambda tensor: flat_gather_csr(tensor, plan, require_native=True),
            values,
            upstream,
        )

        torch.testing.assert_close(select_output, advanced_output, rtol=0.0, atol=0.0)
        torch.testing.assert_close(csr_output, advanced_output, rtol=0.0, atol=0.0)
        torch.testing.assert_close(select_grad, advanced_grad, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(csr_grad, advanced_grad, rtol=1e-5, atol=1e-5)
        self.assertEqual(
            get_flat_gather_csr_execution_counters(),
            {
                "forward_native_eligible_calls": 1,
                "native_backward_executed_calls": 1,
            },
        )

        repeat_output, repeat_grad = _run_and_grad(
            lambda tensor: flat_gather_csr(tensor, plan, require_native=True),
            values,
            upstream,
        )
        torch.testing.assert_close(repeat_output, csr_output, rtol=0.0, atol=0.0)
        torch.testing.assert_close(repeat_grad, csr_grad, rtol=0.0, atol=0.0)

    def test_high_fan_in_gradient(self):
        index = torch.full((31, 12), 3, dtype=torch.int64)
        values = torch.randn(1, 8, 16, device="cuda")
        upstream = torch.randn(1, 31, 12, 16, device="cuda")
        plan = prepare_flat_gather_csr(index, 8, device=values.device)

        stock_output, stock_grad = _run_and_grad(
            lambda tensor: _index_select_gather(tensor, index.to("cuda")),
            values,
            upstream,
        )
        native_output, native_grad = _run_and_grad(
            lambda tensor: flat_gather_csr(tensor, plan, require_native=True),
            values,
            upstream,
        )
        torch.testing.assert_close(native_output, stock_output, rtol=0.0, atol=0.0)
        torch.testing.assert_close(native_grad, stock_grad, rtol=1e-5, atol=1e-5)
        self.assertEqual(int(torch.count_nonzero(native_grad[:, :3])), 0)
        self.assertEqual(int(torch.count_nonzero(native_grad[:, 4:])), 0)

    def test_native_backward_uses_current_nondefault_stream(self):
        index = torch.tensor([[2, 0, 2], [1, 2, 0]], dtype=torch.int64)
        values = torch.arange(48, device="cuda", dtype=torch.float32).reshape(1, 3, 16)
        values.requires_grad_(True)
        plan = prepare_flat_gather_csr(index, 3, device=values.device)
        stream = torch.cuda.Stream(device=values.device)

        with torch.cuda.stream(stream):
            output = flat_gather_csr(values, plan, require_native=True)
            output.sum().backward()
        stream.synchronize()

        expected = torch.tensor([2.0, 1.0, 3.0], device=values.device)
        torch.testing.assert_close(values.grad[0, :, 0], expected, rtol=0.0, atol=0.0)
        self.assertEqual(get_native_backward_execution_count(), 1)

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2,
        "requires two visible P800 devices",
    )
    def test_native_backward_guards_a_noncurrent_device(self):
        original_device = torch.cuda.current_device()
        target_device = (original_device + 1) % torch.cuda.device_count()
        target = torch.device("cuda", target_device)
        index = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
        values = torch.randn(
            1, 3, 16, device=target, dtype=torch.float32, requires_grad=True
        )
        plan = prepare_flat_gather_csr(index, 3, device=target)

        torch.cuda.set_device(original_device)
        output = flat_gather_csr(values, plan, require_native=True)
        output.sum().backward()
        torch.cuda.synchronize(target)

        self.assertEqual(torch.cuda.current_device(), original_device)
        expected = torch.tensor([1.0, 1.0, 2.0], device=target)
        torch.testing.assert_close(values.grad[0, :, 0], expected, rtol=0.0, atol=0.0)

    def test_unsupported_channels_fall_back_or_fail_closed(self):
        index = torch.tensor([[0, 2], [2, 1]], dtype=torch.int64)
        values = torch.randn(
            1, 3, 17, device="cuda", dtype=torch.float32, requires_grad=True
        )
        plan = prepare_flat_gather_csr(index, 3, device=values.device)
        self.assertFalse(plan.native_supported(values))
        output = flat_gather_csr(values, plan)
        output.sum().backward()
        torch.cuda.synchronize()
        expected = torch.tensor([1.0, 1.0, 2.0], device="cuda")
        torch.testing.assert_close(values.grad[0, :, 0], expected, rtol=0.0, atol=0.0)
        with self.assertRaisesRegex(RuntimeError, "channels"):
            flat_gather_csr(values.detach(), plan, require_native=True)
        self.assertEqual(
            get_flat_gather_csr_execution_counters(),
            {
                "forward_native_eligible_calls": 0,
                "native_backward_executed_calls": 0,
            },
        )

    def test_non_contiguous_values_use_fallback(self):
        index = torch.tensor([[0, 2], [2, 1]], dtype=torch.int64)
        base = torch.randn(1, 16, 3, device="cuda")
        values = base.transpose(1, 2).detach().requires_grad_(True)
        self.assertFalse(values.is_contiguous())
        plan = prepare_flat_gather_csr(index, 3, device=values.device)
        expected = _index_select_gather(values, index.to("cuda"))
        actual = flat_gather_csr(values, plan)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            flat_gather_csr(values, plan, require_native=True)

    def test_forward_only_multiple_backward_reload_and_reset(self):
        index = torch.tensor([[2, 0, 2], [1, 2, 0]], dtype=torch.int64)
        values = torch.arange(48, device="cuda", dtype=torch.float32).reshape(1, 3, 16)
        values.requires_grad_(True)
        plan = prepare_flat_gather_csr(index, 3, device=values.device)

        output = flat_gather_csr(values, plan, require_native=True)
        torch.cuda.synchronize()
        self.assertEqual(get_native_backward_execution_count(), 0)
        scalar_output = output.sum()
        scalar_output.backward(retain_graph=True)
        scalar_output.backward()
        torch.cuda.synchronize()
        self.assertEqual(get_native_backward_execution_count(), 2)

        native_path = p800_flat_gather_native.__file__
        reloaded = importlib.reload(p800_flat_gather_native)
        self.assertEqual(reloaded.__file__, native_path)
        self.assertEqual(reloaded.get_backward_execution_count(), 2)
        self.assertEqual(reset_native_backward_execution_count(), 2)
        self.assertEqual(get_native_backward_execution_count(), 0)

    def test_native_plan_rejects_malformed_csr(self):
        device = torch.device("cuda")
        edge_ids = torch.tensor([0, 1], device=device, dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, r"row_ptr\[0\] must be zero"):
            p800_flat_gather_native.prepare_plan(
                torch.tensor([1, 2], device=device, dtype=torch.int32),
                edge_ids,
                1,
            )
        with self.assertRaisesRegex(RuntimeError, "monotonic"):
            p800_flat_gather_native.prepare_plan(
                torch.tensor([0, 2, 1], device=device, dtype=torch.int32),
                edge_ids,
                2,
            )
        with self.assertRaisesRegex(RuntimeError, "edge_ids entries"):
            p800_flat_gather_native.prepare_plan(
                torch.tensor([0, 2], device=device, dtype=torch.int32),
                torch.tensor([0, 2], device=device, dtype=torch.int32),
                1,
            )

    def test_failed_native_backward_does_not_increment(self):
        index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        plan = prepare_flat_gather_csr(index, 3, device="cuda")
        native_plan = p800_flat_gather_native.prepare_plan(
            plan.row_ptr, plan.edge_ids, 3
        )
        with self.assertRaisesRegex(RuntimeError, "row count"):
            p800_flat_gather_native.backward(
                torch.ones(5, 16, device="cuda"), native_plan
            )
        with self.assertRaisesRegex(RuntimeError, "P800 tensor"):
            p800_flat_gather_native.backward(torch.ones(4, 16), native_plan)
        torch.cuda.synchronize()
        self.assertEqual(get_native_backward_execution_count(), 0)

    def test_zero_rows_do_not_count_a_native_launch(self):
        index = torch.empty((0, 12), dtype=torch.int64)
        values = torch.empty(
            (1, 0, 16), device="cuda", dtype=torch.float32, requires_grad=True
        )
        plan = prepare_flat_gather_csr(index, 0, device=values.device)
        output = flat_gather_csr(values, plan, require_native=True)
        self.assertEqual(tuple(output.shape), (1, 0, 12, 16))
        output.sum().backward()
        torch.cuda.synchronize()
        self.assertEqual(
            get_flat_gather_csr_execution_counters(),
            {
                "forward_native_eligible_calls": 1,
                "native_backward_executed_calls": 0,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
