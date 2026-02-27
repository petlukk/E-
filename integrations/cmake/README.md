# CMake — Eä kernel in a C/C++ project

This example shows how to integrate Eä kernels into a CMake build system.
The `EaCompiler.cmake` module provides an `add_ea_kernel()` function that
compiles `.ea` files and exposes them as linkable CMake targets.

## How it works

`cmake/EaCompiler.cmake` defines `add_ea_kernel(target source)` which:
1. Runs `ea kernel.ea --header` to produce both `kernel.o` and `kernel.h`
2. Wraps the `.o` in a static library target
3. Exposes the generated `.h` via the target's include path

Your `CMakeLists.txt` just does:
```cmake
include(cmake/EaCompiler.cmake)
add_ea_kernel(kernel kernel.ea)
target_link_libraries(app kernel m)
```

And your C code includes the generated header:
```c
#include "kernel.h"
fma_kernel(a, b, c, out, n);
```

## Prerequisites

- `ea` compiler on PATH
- CMake 3.16+
- C compiler (gcc, clang, etc.)

## Usage

```bash
cmake -B build && cmake --build build && ./build/app
```

## Files

| File | Purpose |
|------|---------|
| `kernel.ea` | FMA kernel (f32x4, scalar tail) |
| `cmake/EaCompiler.cmake` | Reusable CMake module |
| `CMakeLists.txt` | Project build file |
| `main.c` | C program that calls the kernel |
