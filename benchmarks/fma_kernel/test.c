#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Eä functions  
extern void fma_kernel_f32x4(const float* a, const float* b, const float* c, float* result, int len);

// C reference functions
extern void fma_kernel_f32x4_c(const float* a, const float* b, const float* c, float* result, int len);

#define SIZE 100000
#define RUNS 100

int main() {
    printf("=== FMA Kernel Test ===\n");
    
    // Allocate arrays
    float* a = malloc(SIZE * sizeof(float));
    float* b = malloc(SIZE * sizeof(float)); 
    float* c = malloc(SIZE * sizeof(float));
    float* ea_result = malloc(SIZE * sizeof(float));
    float* c_result = malloc(SIZE * sizeof(float));
    
    // Initialize with test data
    for (int i = 0; i < SIZE; i++) {
        a[i] = (float)i * 0.001f;
        b[i] = 2.0f;
        c[i] = 1.0f;
    }
    
    printf("Testing correctness with %d elements...\n", SIZE);
    
    // Test f32x4 versions
    fma_kernel_f32x4(a, b, c, ea_result, SIZE);
    fma_kernel_f32x4_c(a, b, c, c_result, SIZE);
    
    // Check first few results
    printf("First 5 results (expected: 1.0, 1.002, 1.004, 1.006, 1.008):\n");
    printf("Eä f32x4:  ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", ea_result[i]);
    }
    printf("\n");
    
    printf("C f32x4:   ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", c_result[i]);
    }
    printf("\n");
    
    // Simple correctness check
    int errors = 0;
    for (int i = 0; i < 100; i++) {  // Check first 100
        float diff = ea_result[i] - c_result[i];
        if (diff > 1e-5f || diff < -1e-5f) {
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("✓ Correctness test passed!\n");
    } else {
        printf("✗ Found %d errors in first 100 elements\n", errors);
    }
    
    // Simple timing test
    printf("\nBasic timing test (%d runs):\n", RUNS);
    
    clock_t start, end;
    double ea_time = 0, c_time = 0;
    
    // Time Eä version
    for (int run = 0; run < RUNS; run++) {
        start = clock();
        fma_kernel_f32x4(a, b, c, ea_result, SIZE);
        end = clock();
        ea_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    }
    ea_time /= RUNS;
    
    // Time C version
    for (int run = 0; run < RUNS; run++) {
        start = clock();
        fma_kernel_f32x4_c(a, b, c, c_result, SIZE);
        end = clock();  
        c_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    }
    c_time /= RUNS;
    
    printf("Eä f32x4:  %.4f seconds\n", ea_time);
    printf("C f32x4:   %.4f seconds\n", c_time);
    printf("Ratio:     %.3fx (Eä/C)\n", ea_time / c_time);
    
    if (ea_time / c_time <= 1.1) {
        printf("✅ Within 10%% of C performance!\n");
    } else {
        printf("❌ More than 10%% slower than C\n");
    }
    
    free(a);
    free(b);
    free(c);
    free(ea_result);
    free(c_result);
    
    return 0;
}