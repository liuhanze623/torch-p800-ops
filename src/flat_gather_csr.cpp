#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

// Declared by the public XMLIR runtime header and exported by
// libXMLIRRuntime.so in the validated xpytorch stack.
namespace xmlir_rt {
void* getCurrentStream();
}

extern "C" void p800_launch_flat_gather_csr_backward(
    const float* grad_edges,
    const int* row_ptr,
    const int* edge_ids,
    float* grad_values,
    int num_rows,
    int channels,
    void* stream);

namespace {

// This counter proves that the native binding successfully enqueued a
// non-empty kernel. Callers must synchronize the selected device before using
// the count as proof that device execution completed.
std::atomic<std::uint64_t> native_backward_execution_count{0};

class FlatGatherCSRNativePlan final {
   public:
    FlatGatherCSRNativePlan(
        torch::Tensor row_ptr,
        torch::Tensor edge_ids,
        std::int64_t num_rows)
        : row_ptr_(std::move(row_ptr)),
          edge_ids_(std::move(edge_ids)),
          num_rows_(num_rows) {}

    const torch::Tensor& row_ptr() const { return row_ptr_; }
    const torch::Tensor& edge_ids() const { return edge_ids_; }
    std::int64_t num_rows() const { return num_rows_; }
    std::int64_t edge_count() const { return edge_ids_.numel(); }

   private:
    torch::Tensor row_ptr_;
    torch::Tensor edge_ids_;
    std::int64_t num_rows_;
};

using NativePlanPtr = std::shared_ptr<FlatGatherCSRNativePlan>;

NativePlanPtr prepare_flat_gather_csr_plan(
    const torch::Tensor& row_ptr,
    const torch::Tensor& edge_ids,
    std::int64_t num_rows) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a P800 tensor");
    TORCH_CHECK(edge_ids.is_cuda(), "edge_ids must be a P800 tensor");
    TORCH_CHECK(
        row_ptr.device() == edge_ids.device(),
        "row_ptr and edge_ids must be on the same P800 device");
    TORCH_CHECK(
        row_ptr.scalar_type() == at::ScalarType::Int,
        "row_ptr must be int32");
    TORCH_CHECK(
        edge_ids.scalar_type() == at::ScalarType::Int,
        "edge_ids must be int32");
    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be one-dimensional");
    TORCH_CHECK(edge_ids.dim() == 1, "edge_ids must be one-dimensional");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(edge_ids.is_contiguous(), "edge_ids must be contiguous");
    TORCH_CHECK(num_rows >= 0, "num_rows must be non-negative");
    TORCH_CHECK(
        num_rows <= std::numeric_limits<int>::max(),
        "num_rows exceeds the initial int32 kernel contract");
    TORCH_CHECK(
        edge_ids.numel() <= std::numeric_limits<int>::max(),
        "edge count exceeds the initial int32 kernel contract");
    TORCH_CHECK(
        row_ptr.numel() == num_rows + 1,
        "row_ptr length must equal num_rows + 1");

    const c10::DeviceGuard device_guard(row_ptr.device());
    auto owned_row_ptr = row_ptr.clone();
    auto owned_edge_ids = edge_ids.clone();

    // Validate once while constructing an opaque plan. The owned device
    // tensors are never exposed through pybind, so backward need not rescan
    // the CSR structure on every training step.
    auto row_ptr_cpu = owned_row_ptr.to(torch::kCPU);
    auto edge_ids_cpu = owned_edge_ids.to(torch::kCPU);
    const auto* row_data = row_ptr_cpu.data_ptr<int>();
    const auto* edge_data = edge_ids_cpu.data_ptr<int>();
    const auto edge_count = edge_ids_cpu.numel();

    TORCH_CHECK(row_data[0] == 0, "row_ptr[0] must be zero");
    int previous = 0;
    for (std::int64_t row = 1; row <= num_rows; ++row) {
        const int current = row_data[row];
        TORCH_CHECK(
            current >= previous && current <= edge_count,
            "row_ptr must be monotonic and bounded by edge count");
        previous = current;
    }
    TORCH_CHECK(
        previous == edge_count,
        "row_ptr[-1] must equal the edge count");
    for (std::int64_t position = 0; position < edge_count; ++position) {
        TORCH_CHECK(
            edge_data[position] >= 0 && edge_data[position] < edge_count,
            "edge_ids entries must be in [0, edge_count)");
    }

    return std::make_shared<FlatGatherCSRNativePlan>(
        std::move(owned_row_ptr), std::move(owned_edge_ids), num_rows);
}

torch::Tensor flat_gather_csr_backward(
    const torch::Tensor& grad_edges,
    const NativePlanPtr& plan) {
    TORCH_CHECK(plan, "plan must be a valid native CSR plan");
    const auto& row_ptr = plan->row_ptr();
    const auto& edge_ids = plan->edge_ids();
    const auto num_rows = plan->num_rows();
    TORCH_CHECK(grad_edges.is_cuda(), "grad_edges must be a P800 tensor");
    TORCH_CHECK(
        grad_edges.device() == row_ptr.device() &&
            grad_edges.device() == edge_ids.device(),
        "grad_edges, row_ptr, and edge_ids must be on the same P800 device");
    TORCH_CHECK(
        grad_edges.scalar_type() == at::ScalarType::Float,
        "grad_edges must be float32");
    TORCH_CHECK(
        grad_edges.dim() == 2,
        "grad_edges must have shape [edges, channels]");
    TORCH_CHECK(grad_edges.is_contiguous(), "grad_edges must be contiguous");
    TORCH_CHECK(
        plan->edge_count() == grad_edges.size(0),
        "grad_edges row count must equal the plan edge count");

    const auto channels = grad_edges.size(1);
    TORCH_CHECK(
        channels > 0 && channels <= 128 && channels % 16 == 0,
        "initial native kernel requires channels in {16, 32, ..., 128}");

    const c10::DeviceGuard device_guard(grad_edges.device());
    auto grad_values = torch::empty(
        {1, num_rows, channels}, grad_edges.options());
    if (num_rows != 0) {
        p800_launch_flat_gather_csr_backward(
            grad_edges.data_ptr<float>(),
            row_ptr.data_ptr<int>(),
            edge_ids.data_ptr<int>(),
            grad_values.data_ptr<float>(),
            static_cast<int>(num_rows),
            static_cast<int>(channels),
            xmlir_rt::getCurrentStream());
        native_backward_execution_count.fetch_add(
            1, std::memory_order_relaxed);
    }
    return grad_values;
}

std::uint64_t reset_backward_execution_count() {
    return native_backward_execution_count.exchange(
        0, std::memory_order_relaxed);
}

std::uint64_t get_backward_execution_count() {
    return native_backward_execution_count.load(std::memory_order_relaxed);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<FlatGatherCSRNativePlan, NativePlanPtr>(
        module, "_NativePlan");
    module.def(
        "prepare_plan",
        &prepare_flat_gather_csr_plan,
        "Validate and own a P800 reverse-CSR plan");
    module.def(
        "backward",
        &flat_gather_csr_backward,
        "P800 reverse-CSR gather backward (float32, B=1)");
    module.def(
        "reset_backward_execution_count",
        &reset_backward_execution_count,
        "Reset the native backward enqueue counter and return its prior value");
    module.def(
        "get_backward_execution_count",
        &get_backward_execution_count,
        "Read the process-local native backward enqueue counter");
}
