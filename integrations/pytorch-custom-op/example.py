#!/usr/bin/env python3
"""Demo: Eä FMA kernel as a PyTorch custom op."""

import torch
from ea_ops import EaFMA

# Create random input tensors
torch.manual_seed(42)
a = torch.randn(1000, dtype=torch.float32)
b = torch.randn(1000, dtype=torch.float32)
c = torch.randn(1000, dtype=torch.float32)

# Run through Eä kernel
result = EaFMA.apply(a, b, c)

# Verify against PyTorch
expected = torch.addcmul(c, a, b)
max_err = (result - expected).abs().max().item()

print(f"Eä FMA result (first 5):    {result[:5].tolist()}")
print(f"PyTorch result (first 5):   {expected[:5].tolist()}")
print(f"max absolute error: {max_err:.2e}")
print(f"match: {max_err < 1e-6}")
