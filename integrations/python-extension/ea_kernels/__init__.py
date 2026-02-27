"""ea_kernels — SIMD scale kernel compiled with the Eä compiler."""

import ctypes
import numpy as np
from pathlib import Path

# Load the shared library from the package directory
_lib_path = Path(__file__).parent / "kernel.so"
if not _lib_path.exists():
    raise RuntimeError(
        f"kernel.so not found at {_lib_path}. "
        "Run 'pip install .' to build the extension."
    )

_lib = ctypes.CDLL(str(_lib_path))

_lib.scale.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # data
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.c_int32,                  # n
    ctypes.c_float,                  # factor
]
_lib.scale.restype = None


def scale(data: np.ndarray, factor: float) -> np.ndarray:
    """Multiply every element of a float32 array by a scalar factor.

    Uses SIMD f32x8 with masked tail — works for any array length.
    """
    if data.dtype != np.float32:
        raise TypeError(f"expected float32 array, got {data.dtype}")
    data = np.ascontiguousarray(data)
    out = np.empty_like(data)
    _lib.scale(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(data.size),
        ctypes.c_float(factor),
    )
    return out
