#include <stdio.h>
#include "kernel.h"

int main(void) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float b[] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float c[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float out[7];
    int n = 7;

    fma_kernel(a, b, c, out, n);

    printf("fma_kernel: out[i] = a[i] * b[i] + c[i]\n");
    for (int i = 0; i < n; i++) {
        printf("  %g * %g + %g = %g\n", a[i], b[i], c[i], out[i]);
    }

    return 0;
}
