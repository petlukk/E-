"""ea_ops — Eä SIMD kernels as PyTorch autograd Functions.

Compiles the kernel on first import and wraps it in a torch.autograd.Function
for seamless use in PyTorch programs.
"""

import ctypes
import subprocess
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Compile kernel on import
# ---------------------------------------------------------------------------
_dir = Path(__file__).parent
_so_path = _dir / "kernel.so"

if not _so_path.exists():
    print("Compiling kernel.ea...", file=sys.stderr)
    result = subprocess.run(
        ["ea", str(_dir / "kernel.ea"), "--lib"],
        cwd=str(_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Eä compilation failed")
    print(result.stderr, end="", file=sys.stderr)

_lib = ctypes.CDLL(str(_so_path))
_lib.fma.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # a
    ctypes.POINTER(ctypes.c_float),  # b
    ctypes.POINTER(ctypes.c_float),  # c
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.c_int32,                  # n
]
_lib.fma.restype = None


# ---------------------------------------------------------------------------
# PyTorch autograd wrapper
# ---------------------------------------------------------------------------
class EaFMA(torch.autograd.Function):
    """Fused multiply-add: out = a * b + c

    Forward-only — Eä kernels are raw compute, not differentiable ops.
    Use this for inference or as a leaf operation.
    """

    @staticmethod
    def forward(ctx, a, b, c):
        # Ensure contiguous float32 on CPU
        a = a.contiguous().float()
        b = b.contiguous().float()
        c = c.contiguous().float()
        assert a.device.type == "cpu", "Eä kernels run on CPU"
        assert a.shape == b.shape == c.shape, "shapes must match"

        out = torch.empty_like(a)
        n = a.numel()

        F = ctypes.POINTER(ctypes.c_float)
        _lib.fma(
            ctypes.cast(a.data_ptr(), F),
            ctypes.cast(b.data_ptr(), F),
            ctypes.cast(c.data_ptr(), F),
            ctypes.cast(out.data_ptr(), F),
            ctypes.c_int32(n),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "EaFMA is forward-only. Eä kernels are raw SIMD compute, "
            "not differentiable ops. Use this for inference or leaf operations."
        )
