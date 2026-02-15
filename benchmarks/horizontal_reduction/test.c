#include <stdio.h>
#include <math.h>

// Ea kernel functions
extern float sum_f32x4(const float*, int);
extern float sum_f32x8(const float*, int);
extern float max_f32x4(const float*, int);
extern float min_f32x4(const float*, int);

int main() {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f, 5.0f, 3.0f};
    int n = 10;

    float expected_sum = 0.0f;
    float expected_max = data[0];
    float expected_min = data[0];
    for (int i = 0; i < n; i++) {
        expected_sum += data[i];
        if (data[i] > expected_max) expected_max = data[i];
        if (data[i] < expected_min) expected_min = data[i];
    }

    float s4 = sum_f32x4(data, n);
    float s8 = sum_f32x8(data, n);
    float mx = max_f32x4(data, n);
    float mn = min_f32x4(data, n);

    printf("sum_f32x4: %g (expected %g) %s\n", s4, expected_sum, fabsf(s4 - expected_sum) < 0.01f ? "OK" : "FAIL");
    printf("sum_f32x8: %g (expected %g) %s\n", s8, expected_sum, fabsf(s8 - expected_sum) < 0.01f ? "OK" : "FAIL");
    printf("max_f32x4: %g (expected %g) %s\n", mx, expected_max, fabsf(mx - expected_max) < 0.01f ? "OK" : "FAIL");
    printf("min_f32x4: %g (expected %g) %s\n", mn, expected_min, fabsf(mn - expected_min) < 0.01f ? "OK" : "FAIL");

    return 0;
}
