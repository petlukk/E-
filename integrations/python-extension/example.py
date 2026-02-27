#!/usr/bin/env python3
"""Quick demo: scale a numpy array using an Eä SIMD kernel."""

import numpy as np
from ea_kernels import scale

data = np.ones(100, dtype=np.float32)
result = scale(data, 2.0)

print(f"input:  {data[:8]}  (100 elements)")
print(f"output: {result[:8]}  (100 elements)")
print(f"all correct: {np.allclose(result, 2.0)}")
