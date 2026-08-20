from pathlib import Path

from setuptools import find_packages, setup
from torch_xmlir.utils.cpp_extension import BuildExtension, XPUExtension

ROOT = Path(__file__).resolve().parent


setup(
    name="torch-p800-ops",
    version="0.1.0a1",
    description=(
        "Experimental xpytorch custom operators for Kunlunxin P800, "
        "starting with reverse-CSR gather backward"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Hanze Liu",
    url="https://github.com/liuhanze623/torch-p800-ops",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=["numpy>=1.23"],
    ext_modules=[
        XPUExtension(
            name="p800_flat_gather_native",
            sources=[
                "src/flat_gather_csr.cpp",
                "src/flat_gather_csr_kernel.xpu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "xtdk": ["-O2"],
            },
            extra_link_args=[
                "-Wl,-rpath,$ORIGIN/torch_xmlir",
                "-Wl,-rpath,$ORIGIN/torch/lib",
                "-Wl,-rpath,$ORIGIN/../../../xcudart/lib",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
