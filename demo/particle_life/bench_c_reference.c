#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void particle_life_step_c(
    float *px, float *py, float *vx, float *vy,
    const int *types, const float *matrix,
    int n, int num_types,
    float r_max, float dt, float friction, float size
) {
    float r_max2 = r_max * r_max;
    for (int i = 0; i < n; i++) {
        float xi = px[i], yi = py[i];
        int ti = types[i];
        float fx = 0.0f, fy = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = px[j] - xi;
            float dy = py[j] - yi;
            float dist2 = dx * dx + dy * dy;
            if (dist2 > 0.0f && dist2 < r_max2) {
                float dist = sqrtf(dist2);
                float strength = matrix[ti * num_types + types[j]];
                float force = strength * (1.0f - dist / r_max);
                fx += force * dx / dist;
                fy += force * dy / dist;
            }
        }

        vx[i] = (vx[i] + fx * dt) * friction;
        vy[i] = (vy[i] + fy * dt) * friction;
        px[i] += vx[i];
        py[i] += vy[i];

        if (px[i] < 0.0f) px[i] += size;
        if (px[i] >= size) px[i] -= size;
        if (py[i] < 0.0f) py[i] += size;
        if (py[i] >= size) py[i] -= size;
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int n = 2000;
    if (argc > 1) n = atoi(argv[1]);

    int num_types = 6;
    float r_max = 80.0f, dt = 0.5f, friction = 0.5f, size = 800.0f;

    float *px = malloc(n * sizeof(float));
    float *py = malloc(n * sizeof(float));
    float *vx = calloc(n, sizeof(float));
    float *vy = calloc(n, sizeof(float));
    int   *types = malloc(n * sizeof(int));
    float *matrix = malloc(num_types * num_types * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        px[i] = (float)rand() / RAND_MAX * size;
        py[i] = (float)rand() / RAND_MAX * size;
        types[i] = rand() % num_types;
    }
    for (int i = 0; i < num_types * num_types; i++)
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    int warmup = 5, runs = 50;

    for (int i = 0; i < warmup; i++)
        particle_life_step_c(px, py, vx, vy, types, matrix,
                             n, num_types, r_max, dt, friction, size);

    double times[50];
    for (int i = 0; i < runs; i++) {
        double t0 = now_ms();
        particle_life_step_c(px, py, vx, vy, types, matrix,
                             n, num_types, r_max, dt, friction, size);
        times[i] = now_ms() - t0;
    }

    /* simple sort for median */
    for (int i = 0; i < runs - 1; i++)
        for (int j = i + 1; j < runs; j++)
            if (times[j] < times[i]) {
                double tmp = times[i]; times[i] = times[j]; times[j] = tmp;
            }

    printf("C reference (N=%d): %.3f ms\n", n, times[runs / 2]);

    free(px); free(py); free(vx); free(vy); free(types); free(matrix);
    return 0;
}
