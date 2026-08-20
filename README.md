# torch-p800-ops

English | [简体中文](README.zh-CN.md)

`torch-p800-ops` is an experimental source package for custom xpytorch
operators targeting Kunlunxin P800. It currently provides `flat_gather_csr`,
which optimizes the backward pass of a gather with a reusable static index.

## How It Works

The forward pass delegates to xpytorch `index_select`. Before execution, the
package builds a reverse-CSR plan that groups gathered positions by source row.
The native backward kernel then sums each row's gradients locally and writes the
result once.

The native path accepts:

- Kunlunxin P800 PCIe KL3 tensors exposed through xpytorch's CUDA-compatible API;
- contiguous FP32 `values` with shape `[1, N, C]`;
- `C` in `{16, 32, ..., 128}`;
- a static `torch.int64` index and a plan reused across calls.

Unsupported inputs use xpytorch `index_select` unless `require_native=True` is
set.

## Compatibility

| Component | Validated version |
|---|---|
| Python | 3.10 |
| xpytorch | 2.5.1 |
| `torch_plugin` | 0.1.0 |
| `xmlir` | 1.0.0.1 |
| XTDK | 3.6.0.1 |

The repository does not include the vendor toolchain, runtime, or drivers. Use
a Kunlunxin xpytorch environment; ordinary PyPI PyTorch is not supported for the
native build.

## Install

```bash
export CUDA_HOME="${CONDA_PREFIX}/xcudart"
python -m pip install --no-build-isolation -v .
```

## Example

```python
import torch

from p800_ops import flat_gather_csr, prepare_flat_gather_csr

device = torch.device("cuda:0")
index = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
values = (
    torch.arange(3 * 16, device=device, dtype=torch.float32)
    .reshape(1, 3, 16)
    .requires_grad_()
)

plan = prepare_flat_gather_csr(index, num_values=3, device=device)
output = flat_gather_csr(values, plan, require_native=True)
assert output.shape == (1, 2, 2, 16)

output.sum().backward()
expected = torch.tensor([1.0, 1.0, 2.0], device=device)
torch.testing.assert_close(values.grad[0, :, 0], expected)
```

## Build And Verify

From a source checkout in the validated vendor environment:

```bash
python setup.py build_ext --inplace
python -m unittest discover -s tests -v
python benchmarks/benchmark_flat_gather_csr.py
```

The benchmark uses generated input by default and rejects silent fallback to the
stock path.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
