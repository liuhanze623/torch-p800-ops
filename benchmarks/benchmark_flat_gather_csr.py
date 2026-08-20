import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from p800_ops import (  # noqa: E402
    flat_gather_csr,
    prepare_flat_gather_csr,
)
from p800_ops.flat_gather_csr import (  # noqa: E402
    get_flat_gather_csr_execution_counters,
    reset_flat_gather_csr_execution_counters,
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


def timed_training_step(function, values, upstream, warmup, repeats):
    for _ in range(warmup):
        values.grad = None
        output = function(values)
        output.backward(upstream)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        values.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = function(values)
        output.backward(upstream)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def timed_backward(function, values, upstream, warmup, repeats):
    values.grad = None
    output = function(values)
    torch.cuda.synchronize()
    for _ in range(warmup):
        values.grad = None
        output.backward(upstream, retain_graph=True)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        values.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        output.backward(upstream, retain_graph=True)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    del output
    values.grad = None
    return samples


def run_and_grad(function, values, upstream):
    candidate = values.detach().clone().requires_grad_(True)
    output = function(candidate)
    output.backward(upstream)
    torch.cuda.synchronize()
    return output.detach(), candidate.grad.detach()


def error_metrics(candidate, reference):
    difference = candidate - reference
    max_abs = difference.abs().max().item()
    difference_l2 = torch.sqrt(torch.sum(difference * difference)).item()
    reference_l2 = torch.sqrt(torch.sum(reference * reference)).item()
    rel_l2 = difference_l2 / max(reference_l2, 1.0e-30)
    return max_abs, rel_l2


def summarize(name, samples):
    ordered = sorted(samples)
    median = statistics.median(samples)
    p20 = ordered[max(0, int(0.20 * (len(ordered) - 1)))]
    p80 = ordered[min(len(ordered) - 1, int(0.80 * (len(ordered) - 1)))]
    print(
        f"{name}: median_ms={median:.3f} p20_ms={p20:.3f} "
        f"p80_ms={p80:.3f} samples={len(samples)}"
    )
    return median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-values", type=int, default=65536)
    parser.add_argument("--queries", type=int, default=65536)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--channels", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    index_cpu = torch.randint(
        args.num_values,
        (args.queries, args.neighbors),
        generator=generator,
        dtype=torch.int64,
    )
    index_numpy = index_cpu.numpy()

    index_xpu = index_cpu.to("cuda")
    torch.manual_seed(args.seed)
    values = torch.randn(
        1,
        args.num_values,
        args.channels,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    upstream = torch.randn(
        1,
        args.queries,
        args.neighbors,
        args.channels,
        device="cuda",
        dtype=torch.float32,
    )

    plan_start = time.perf_counter()
    plan = prepare_flat_gather_csr(index_cpu, args.num_values, device=values.device)
    torch.cuda.synchronize()
    plan_ms = (time.perf_counter() - plan_start) * 1000.0
    degree_numpy = np.bincount(index_numpy.reshape(-1), minlength=args.num_values)
    unique_degree, degree_counts = np.unique(degree_numpy, return_counts=True)
    print(
        "shape",
        tuple(values.shape),
        "output",
        (1, args.queries, args.neighbors, args.channels),
    )
    print(
        "plan_ms",
        f"{plan_ms:.3f}",
        "degree_min",
        int(degree_numpy.min()),
        "degree_max",
        int(degree_numpy.max()),
        "degree_mean",
        f"{degree_numpy.mean():.3f}",
        "native_supported",
        plan.native_supported(values),
    )
    rejection_reason = plan.native_rejection_reason(values)
    if rejection_reason is not None:
        raise RuntimeError(
            f"benchmark refuses to measure the stock fallback: {rejection_reason}"
        )
    percentiles = np.percentile(degree_numpy, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(
        "degree_percentiles",
        " ".join(
            f"p{percentile}={value:.3f}"
            for percentile, value in zip(
                (0, 1, 5, 25, 50, 75, 95, 99, 100), percentiles
            )
        ),
    )
    print(
        "degree_histogram",
        " ".join(
            f"{int(degree)}:{int(count)}"
            for degree, count in zip(unique_degree, degree_counts)
        ),
    )

    methods = [
        ("advanced", lambda tensor: _advanced_gather(tensor, index_xpu)),
        (
            "index_select",
            lambda tensor: _index_select_gather(tensor, index_xpu),
        ),
        (
            "flat_gather_csr",
            lambda tensor: flat_gather_csr(tensor, plan, require_native=True),
        ),
    ]

    prior_counters = reset_flat_gather_csr_execution_counters()
    print("execution_counters_before_reset", prior_counters)
    reference_output, reference_grad = run_and_grad(methods[0][1], values, upstream)
    correctness = {}
    repeatability = None
    for name, function in methods[1:]:
        output, grad = run_and_grad(function, values, upstream)
        forward_max_abs, forward_rel_l2 = error_metrics(output, reference_output)
        grad_max_abs, grad_rel_l2 = error_metrics(grad, reference_grad)
        correctness[name] = {
            "forward_max_abs": forward_max_abs,
            "forward_rel_l2": forward_rel_l2,
            "grad_max_abs": grad_max_abs,
            "grad_rel_l2": grad_rel_l2,
        }
        print(
            f"correctness {name} "
            f"forward_max_abs={forward_max_abs:.9g} "
            f"forward_rel_l2={forward_rel_l2:.9g} "
            f"grad_max_abs={grad_max_abs:.9g} "
            f"grad_rel_l2={grad_rel_l2:.9g}"
        )
        if name == "flat_gather_csr":
            repeat_output, repeat_grad = run_and_grad(function, values, upstream)
            repeat_forward_max_abs, repeat_forward_rel_l2 = error_metrics(
                repeat_output, output
            )
            repeat_grad_max_abs, repeat_grad_rel_l2 = error_metrics(repeat_grad, grad)
            repeatability = {
                "forward_max_abs": repeat_forward_max_abs,
                "forward_rel_l2": repeat_forward_rel_l2,
                "grad_max_abs": repeat_grad_max_abs,
                "grad_rel_l2": repeat_grad_rel_l2,
            }
            print(
                "repeatability flat_gather_csr "
                f"forward_max_abs={repeat_forward_max_abs:.9g} "
                f"forward_rel_l2={repeat_forward_rel_l2:.9g} "
                f"grad_max_abs={repeat_grad_max_abs:.9g} "
                f"grad_rel_l2={repeat_grad_rel_l2:.9g}"
            )
            del repeat_output, repeat_grad
        del output, grad
    del reference_output, reference_grad

    step_medians = {}
    backward_medians = {}
    for name, function in methods:
        samples = timed_training_step(
            function, values, upstream, args.warmup, args.repeats
        )
        step_medians[name] = summarize(f"step_{name}", samples)
        backward_samples = timed_backward(
            function, values, upstream, args.warmup, args.repeats
        )
        backward_medians[name] = summarize(f"backward_{name}", backward_samples)

    print(
        "step_speedup_vs_advanced",
        f"{step_medians['advanced'] / step_medians['flat_gather_csr']:.3f}x",
    )
    print(
        "step_speedup_vs_index_select",
        f"{step_medians['index_select'] / step_medians['flat_gather_csr']:.3f}x",
    )
    print(
        "backward_speedup_vs_advanced",
        f"{backward_medians['advanced'] / backward_medians['flat_gather_csr']:.3f}x",
    )
    backward_index_select_speedup = (
        backward_medians["index_select"] / backward_medians["flat_gather_csr"]
    )
    print(
        "backward_speedup_vs_index_select",
        f"{backward_index_select_speedup:.3f}x",
    )

    best_stock_step = min(step_medians["advanced"], step_medians["index_select"])
    gate_speedup = best_stock_step / step_medians["flat_gather_csr"]
    candidate_metrics = correctness["flat_gather_csr"]
    if repeatability is None:
        raise AssertionError("flat_gather_csr repeatability metrics are missing")
    all_error_metrics = [*candidate_metrics.values(), *repeatability.values()]
    gate_finite = all(math.isfinite(value) for value in all_error_metrics)
    gate_forward = (
        candidate_metrics["forward_max_abs"] <= 1.0e-5
        and candidate_metrics["forward_rel_l2"] <= 1.0e-4
    )
    gate_gradient = (
        candidate_metrics["grad_max_abs"] <= 1.0e-5
        and candidate_metrics["grad_rel_l2"] <= 1.0e-4
    )
    gate_repeatability = (
        repeatability["forward_max_abs"] <= 1.0e-5
        and repeatability["forward_rel_l2"] <= 1.0e-4
        and repeatability["grad_max_abs"] <= 1.0e-5
        and repeatability["grad_rel_l2"] <= 1.0e-4
    )
    gate_performance = math.isfinite(gate_speedup) and gate_speedup >= 1.2
    gate_pass = (
        gate_finite
        and gate_forward
        and gate_gradient
        and gate_repeatability
        and gate_performance
    )
    print(
        "candidate_gate",
        f"finite={gate_finite}",
        f"forward={gate_forward}",
        f"gradient={gate_gradient}",
        f"repeatability={gate_repeatability}",
        f"step_speedup_vs_fastest_stock={gate_speedup:.3f}x",
        f"pass={gate_pass}",
    )

    torch.cuda.synchronize()
    observed_counters = get_flat_gather_csr_execution_counters()
    expected_counters = {
        "forward_native_eligible_calls": 3 + args.warmup + args.repeats,
        "native_backward_executed_calls": 2 + 2 * (args.warmup + args.repeats),
    }
    print("execution_counters_expected", expected_counters)
    print("execution_counters_observed", observed_counters)
    if observed_counters != expected_counters:
        raise AssertionError(
            "native backward execution proof mismatch: "
            f"observed={observed_counters} expected={expected_counters}"
        )
    if not gate_pass:
        raise AssertionError("flat_gather_csr failed the correctness/performance gate")


if __name__ == "__main__":
    main()
