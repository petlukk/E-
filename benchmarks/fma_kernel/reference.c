// Hand-optimized C reference implementation with AVX2 intrinsics
#include <immintrin.h>
#include <stdint.h>

// FMA with AVX2 f32x8 (256-bit vectors)
void fma_kernel_f32x8_c(
    const float* a, 
    const float* b, 
    const float* c, 
    float* result, 
    int32_t len
) {
    int32_t i = 0;
    
    // Process 8 elements at a time with AVX2
    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]); 
        __m256 vc = _mm256_loadu_ps(&c[i]);
        
        // Fused multiply-add: va * vb + vc
        __m256 vresult = _mm256_fmadd_ps(va, vb, vc);
        
        _mm256_storeu_ps(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < len; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// FMA with SSE f32x4 (128-bit vectors) 
void fma_kernel_f32x4_c(
    const float* a,
    const float* b, 
    const float* c,
    float* result,
    int32_t len
) {
    int32_t i = 0;
    
    // Process 4 elements at a time with SSE
    for (; i + 4 <= len; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vc = _mm_loadu_ps(&c[i]);
        
        // Fused multiply-add: va * vb + vc  
        __m128 vresult = _mm_fmadd_ps(va, vb, vc);
        
        _mm_storeu_ps(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < len; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// Scalar reference (no SIMD)
void fma_kernel_scalar_c(
    const float* a,
    const float* b, 
    const float* c,
    float* result,
    int32_t len
) {
    for (int32_t i = 0; i < len; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}