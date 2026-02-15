#include <stdio.h>
#include <stdlib.h>

extern void simple_add(const float* a, const float* b, float* result, int len);

int main() {
    printf("Simple vector add test\n");
    
    float a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float b[8] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0}; 
    float result[8];
    
    simple_add(a, b, result, 8);
    
    printf("Results: ");
    for (int i = 0; i < 8; i++) {
        printf("%.0f ", result[i]);
    }
    printf("\n");
    
    // Expected: 11 22 33 44 55 66 77 88
    return 0;
}