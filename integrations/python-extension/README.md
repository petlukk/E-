# Python setuptools — Eä kernel as a pip-installable package

This example shows how to package an Eä SIMD kernel as a Python extension
that installs with `pip install .` and imports like any normal module.

## How it works

`setup.py` defines a custom `build_py` command that:
1. Runs `ea kernel.ea --lib` to compile the kernel to a `.so`
2. Copies the `.so` into the `ea_kernels/` package directory
3. Continues with the normal setuptools build

The `__init__.py` loads the `.so` via ctypes and exposes a Pythonic
`scale(data, factor)` function that accepts numpy arrays.

## Prerequisites

- `ea` compiler on PATH
- Python 3.8+
- numpy

## Usage

```bash
pip install .
python example.py
```

Or use directly:

```python
from ea_kernels import scale
import numpy as np

result = scale(np.ones(100, dtype=np.float32), 2.0)
```

## Files

| File | Purpose |
|------|---------|
| `kernel.ea` | SIMD scale kernel (f32x8, masked tail) |
| `setup.py` | Custom build command that invokes `ea` |
| `pyproject.toml` | Package metadata and build config |
| `ea_kernels/__init__.py` | ctypes loader + numpy wrapper |
| `example.py` | Usage demo |
