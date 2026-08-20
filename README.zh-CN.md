# torch-p800-ops

[English](README.md) | 简体中文

`torch-p800-ops` 是面向昆仑芯 P800 的实验性 xpytorch 自定义算子源码包。
当前提供 `flat_gather_csr`，用于优化静态索引可重复使用时的 gather 反向传播。

## 工作原理

前向直接调用 xpytorch `index_select`。执行前，软件包会构建 reverse-CSR
plan，按源行整理被读取的位置；原生反向 kernel 在本地累加每一行的梯度，
最后统一写回。

原生路径要求：

- 通过 xpytorch CUDA 兼容接口访问的昆仑芯 P800 PCIe KL3；
- 连续 FP32 `values`，形状为 `[1, N, C]`；
- `C` 属于 `{16, 32, ..., 128}`；
- 静态 `torch.int64` 索引，并在多次调用间复用 plan。

输入不满足约束时默认使用 xpytorch `index_select`；设置
`require_native=True` 后会直接拒绝回退。

## 兼容环境

| 组件 | 已验证版本 |
|---|---|
| Python | 3.10 |
| xpytorch | 2.5.1 |
| `torch_plugin` | 0.1.0 |
| `xmlir` | 1.0.0.1 |
| XTDK | 3.6.0.1 |

仓库不包含厂商工具链、运行时或驱动。原生扩展必须在昆仑芯 xpytorch 环境
中构建，不能使用普通 PyPI PyTorch 替代 xpytorch。

## 安装

```bash
export CUDA_HOME="${CONDA_PREFIX}/xcudart"
python -m pip install --no-build-isolation -v .
```

## 示例

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

## 构建与验证

在已验证的厂商环境和源码仓库中执行：

```bash
python setup.py build_ext --inplace
python -m unittest discover -s tests -v
python benchmarks/benchmark_flat_gather_csr.py
```

benchmark 默认使用自动生成的输入，并会拒绝把静默回退误认为原生执行。

## 许可证

本项目采用 Apache License 2.0，详见 [LICENSE](LICENSE)。
