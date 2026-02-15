// ISPC FMA kernel: result[i] = a[i] * b[i] + c[i]

export void fma_kernel_ispc(
    uniform float a[],
    uniform float b[],
    uniform float c[],
    uniform float result[],
    uniform int32 len
) {
    foreach (i = 0 ... len) {
        result[i] = a[i] * b[i] + c[i];
    }
}
