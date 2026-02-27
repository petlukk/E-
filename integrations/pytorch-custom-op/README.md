# PyTorch custom op — Eä kernel as a torch.autograd.Function

This example shows how to wrap an Eä SIMD kernel as a PyTorch custom
operation using `torch.autograd.Function`.

## How it works

`ea_ops.py`:
1. Compiles `kernel.ea` on first import (if `.so` doesn't exist)
2. Loads the shared library via ctypes
3. Defines `EaFMA(torch.autograd.Function)` whose `forward()` extracts
   `.data_ptr()` from contiguous float32 tensors and calls the kernel

The kernel is **forward-only** — no backward pass. Eä kernels are raw
SIMD compute, not differentiable ops. This is the right pattern for
inference, preprocessing, or leaf operations in a larger pipeline.

## Prerequisites

- `ea` compiler on PATH
- Python 3.8+
- PyTorch (`pip install torch`)

## Usage

```bash
python example.py
```

Output:
```
Eä FMA result (first 5):    [...]
PyTorch result (first 5):   [...]
max absolute error: 0.00e+00
match: True
```

## Files

| File | Purpose |
|------|---------|
| `kernel.ea` | FMA kernel (f32x8, scalar tail) |
| `ea_ops.py` | Compile-on-import + autograd wrapper |
| `example.py` | Verification against torch.addcmul |
