/*
 * attn_tiled.c -- Q3b: loops + cache tiling + register tiling.
 *
 * Both matrix products are computed in MR x NR blocks (4 x 16 = eight 8-wide AVX
 * registers) that stay in registers for the whole k (or j) loop, so each loaded value of
 * Q / K^T / S / V is reused MR or NR times instead of once. In step 1 the outer j0 loop
 * pins one ~32 KB tile of K^T in L1 while every row of Q streams past it.
 *
 * Assumes N is a multiple of 16 and d a multiple of 16 (true for all given inputs).
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
#define MR 4
#define NR 16

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    if (N % NR || d % NR) {
        fprintf(stderr, "attn_tiled: N and d must be multiples of %d\n", NR);
        exit(1);
    }
    const float scale = 1.0f / sqrtf((float)d);
    const int NB = d >= 512 ? NR : 8192 / d;
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Kt = malloc((size_t)d * N * sizeof(float));

    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];

    /* 1. S = Q K^T * scale */
    for (int j0 = 0; j0 < N; j0 += NB)
        for (int i = 0; i < N; i += MR)
            for (int j = j0; j < j0 + NB && j < N; j += NR) {
                float acc[MR][NR] = {{0}};
                for (int k = 0; k < d; k++)
                    for (int r = 0; r < MR; r++)
                        for (int c = 0; c < NR; c++)
                            acc[r][c] += Q[(size_t)(i + r) * d + k] * Kt[(size_t)k * N + j + c];
                for (int r = 0; r < MR; r++)
                    for (int c = 0; c < NR; c++)
                        S[(size_t)(i + r) * N + j + c] = acc[r][c] * scale;
            }

    /* 2. softmax of each row of S */
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            max = row[j] > max ? row[j] : max;
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }

    /* 3. O = S V */
    for (int i = 0; i < N; i += MR)
        for (int c0 = 0; c0 < d; c0 += NR) {
            float acc[MR][NR] = {{0}};
            for (int j = 0; j < N; j++)
                for (int r = 0; r < MR; r++)
                    for (int c = 0; c < NR; c++)
                        acc[r][c] += S[(size_t)(i + r) * N + j] * V[(size_t)j * d + c0 + c];
            for (int r = 0; r < MR; r++)
                for (int c = 0; c < NR; c++)
                    O[(size_t)(i + r) * d + c0 + c] = acc[r][c];
        }

    free(Kt);
    free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
