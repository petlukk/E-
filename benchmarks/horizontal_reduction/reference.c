// Hand-optimized C reference: horizontal reductions with AVX2/SSE intrinsics
#include <immintrin.h>
#include <stdint.h>
#include <float.h>

// --- Sum ---

// AVX2 f32x8 sum
float sum_f32x8_c(const float* data, int32_t len) {
    __m256 acc = _mm256_setzero_ps();
    int32_t i = 0;

    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        acc = _mm256_add_ps(acc, v);
    }

    // Horizontal sum: reduce 8 floats to 1
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ss(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    float total = _mm_cvtss_f32(sum1);

    for (; i < len; i++) {
        total += data[i];
    }
    return total;
}

// SSE f32x4 sum
float sum_f32x4_c(const float* data, int32_t len) {
    __m128 acc = _mm_setzero_ps();
    int32_t i = 0;

    for (; i + 4 <= len; i += 4) {
        __m128 v = _mm_loadu_ps(&data[i]);
        acc = _mm_add_ps(acc, v);
    }

    // Horizontal sum: reduce 4 floats to 1
    __m128 shuf = _mm_movehdup_ps(acc);
    __m128 sum2 = _mm_add_ps(acc, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    float total = _mm_cvtss_f32(sum1);

    for (; i < len; i++) {
        total += data[i];
    }
    return total;
}

// Scalar sum
float sum_scalar_c(const float* data, int32_t len) {
    float total = 0.0f;
    for (int32_t i = 0; i < len; i++) {
        total += data[i];
    }
    return total;
}

// --- Max ---

// SSE f32x4 max
float max_f32x4_c(const float* data, int32_t len) {
    __m128 acc = _mm_loadu_ps(&data[0]);
    int32_t i = 4;

    for (; i + 4 <= len; i += 4) {
        __m128 v = _mm_loadu_ps(&data[i]);
        acc = _mm_max_ps(acc, v);
    }

    // Horizontal max
    __m128 shuf = _mm_movehdup_ps(acc);
    __m128 max2 = _mm_max_ps(acc, shuf);
    shuf = _mm_movehl_ps(shuf, max2);
    __m128 max1 = _mm_max_ss(max2, shuf);
    float result = _mm_cvtss_f32(max1);

    for (; i < len; i++) {
        if (data[i] > result) result = data[i];
    }
    return result;
}

// Scalar max
float max_scalar_c(const float* data, int32_t len) {
    float result = data[0];
    for (int32_t i = 1; i < len; i++) {
        if (data[i] > result) result = data[i];
    }
    return result;
}

// --- Min ---

// SSE f32x4 min
float min_f32x4_c(const float* data, int32_t len) {
    __m128 acc = _mm_loadu_ps(&data[0]);
    int32_t i = 4;

    for (; i + 4 <= len; i += 4) {
        __m128 v = _mm_loadu_ps(&data[i]);
        acc = _mm_min_ps(acc, v);
    }

    // Horizontal min
    __m128 shuf = _mm_movehdup_ps(acc);
    __m128 min2 = _mm_min_ps(acc, shuf);
    shuf = _mm_movehl_ps(shuf, min2);
    __m128 min1 = _mm_min_ss(min2, shuf);
    float result = _mm_cvtss_f32(min1);

    for (; i < len; i++) {
        if (data[i] < result) result = data[i];
    }
    return result;
}

// Scalar min
float min_scalar_c(const float* data, int32_t len) {
    float result = data[0];
    for (int32_t i = 1; i < len; i++) {
        if (data[i] < result) result = data[i];
    }
    return result;
}
