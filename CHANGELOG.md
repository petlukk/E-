# Changelog

## v0.4.0

### Breaking Changes
- `maddubs(u8x16, i8x16) -> i16x8` renamed to `maddubs_i16` — update all kernels

### New Intrinsics
- `maddubs_i32(u8x16, i8x16) -> i32x4` — safe accumulation via pmaddubsw+pmaddwd chain.
  Programmer explicitly chooses the overflow model by choosing the instruction.
  No silent widening.

### Demos
- `demo/conv2d_3x3/conv_safe.ea` — i32x4 accumulator variant, immune to accumulator overflow
