#include <stdio.h>
#include <stdlib.h>

extern void simple_fma(const float* a, const float* b, const float* c, float* result, int len);

int main() {
    printf("Simple FMA test\n");
    
    float a[4] = {1.0, 2.0, 3.0, 4.0};
    float b[4] = {2.0, 2.0, 2.0, 2.0}; 
    float c[4] = {1.0, 1.0, 1.0, 1.0};
    float result[4];
    
    // Expected: a*b+c = [1*2+1, 2*2+1, 3*2+1, 4*2+1] = [3, 5, 7, 9]
    
    simple_fma(a, b, c, result, 4);
    
    printf("Results: ");
    for (int i = 0; i < 4; i++) {
        printf("%.0f ", result[i]);
    }
    printf("\n");
    
    return 0;
}