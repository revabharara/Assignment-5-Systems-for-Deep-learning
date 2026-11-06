/*
 * attn_best.c -- Q3d: fastest single-threaded version.
 *
 * Fused like attn_fused.c, but 4 query rows at a time and hand-written AVX2/FMA:
 *   - K is packed once into 16-column panels, so the score kernel reads K contiguously.
 *   - Q is pre-scaled by log2(e)/sqrt(d), so softmax needs exp2 instead of exp.
 *   - Scores: a 4 x 16 register tile (8 accumulators), 8 FMAs per 2 loads + 4 broadcasts.
 *   - Softmax: vectorised max / sum and a degree-6 polynomial exp2 (rel. error ~1e-7)
 *     instead of N scalar expf calls per row.
 *   - P V: the same 4 x 16 register tile over V's columns.
 *
 * Assumes N is a multiple of 16 and d a multiple of 16 (true for all given inputs).
 */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
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
#define R 4

static inline __m256 exp2_ps(__m256 x)
{
    x = _mm256_max_ps(x, _mm256_set1_ps(-126.0f));
    const __m256 n = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(x, n);
    __m256 p = _mm256_set1_ps(1.5403530e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3333558e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504109e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022651e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_slli_epi32(_mm256_cvtps_epi32(n), 23);
    return _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p), e));
}

static inline float hmax(__m256 v)
{
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}

static inline float hsum(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
}

/* O[i0 .. i0+R) for query rows q (R x d, pre-scaled); Kp: packed K; S: R x N scratch. */
static void row_block(const float *q, const float *Kp, const float *V, float *O, float *S,
                      int N, int d)
{
    for (int j = 0; j < N; j += 16) {
        const float *kp = Kp + (size_t)j * d;
        __m256 acc[R][2];
        for (int r = 0; r < R; r++)
            acc[r][0] = acc[r][1] = _mm256_setzero_ps();
        for (int k = 0; k < d; k++) {
            const __m256 b0 = _mm256_loadu_ps(kp + 16 * k), b1 = _mm256_loadu_ps(kp + 16 * k + 8);
            for (int r = 0; r < R; r++) {
                const __m256 a = _mm256_broadcast_ss(q + r * d + k);
                acc[r][0] = _mm256_fmadd_ps(a, b0, acc[r][0]);
                acc[r][1] = _mm256_fmadd_ps(a, b1, acc[r][1]);
            }
        }
        for (int r = 0; r < R; r++) {
            _mm256_storeu_ps(S + (size_t)r * N + j, acc[r][0]);
            _mm256_storeu_ps(S + (size_t)r * N + j + 8, acc[r][1]);
        }
    }

    float inv[R];
    for (int r = 0; r < R; r++) {
        float *s = S + (size_t)r * N;
        __m256 m = _mm256_loadu_ps(s);
        for (int j = 8; j < N; j += 8)
            m = _mm256_max_ps(m, _mm256_loadu_ps(s + j));
        const __m256 mx = _mm256_set1_ps(hmax(m));
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < N; j += 8) {
            const __m256 p = exp2_ps(_mm256_sub_ps(_mm256_loadu_ps(s + j), mx));
            _mm256_storeu_ps(s + j, p);
            sum = _mm256_add_ps(sum, p);
        }
        inv[r] = 1.0f / hsum(sum);
    }

    for (int c = 0; c < d; c += 16) {
        __m256 acc[R][2];
        for (int r = 0; r < R; r++)
            acc[r][0] = acc[r][1] = _mm256_setzero_ps();
        for (int j = 0; j < N; j++) {
            const float *v = V + (size_t)j * d + c;
            const __m256 b0 = _mm256_loadu_ps(v), b1 = _mm256_loadu_ps(v + 8);
            for (int r = 0; r < R; r++) {
                const __m256 a = _mm256_broadcast_ss(S + (size_t)r * N + j);
                acc[r][0] = _mm256_fmadd_ps(a, b0, acc[r][0]);
                acc[r][1] = _mm256_fmadd_ps(a, b1, acc[r][1]);
            }
        }
        for (int r = 0; r < R; r++) {
            const __m256 iv = _mm256_set1_ps(inv[r]);
            _mm256_storeu_ps(O + (size_t)r * d + c, _mm256_mul_ps(acc[r][0], iv));
            _mm256_storeu_ps(O + (size_t)r * d + c + 8, _mm256_mul_ps(acc[r][1], iv));
        }
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    if (N % 16 || d % 16) {
        fprintf(stderr, "attn_best: N and d must be multiples of 16\n");
        exit(1);
    }
    const float qscale = 1.4426950408889634f / sqrtf((float)d);
    float *Kp = malloc((size_t)N * d * sizeof(float));
    float *Qs = malloc((size_t)N * d * sizeof(float));
    float *S = malloc((size_t)R * N * sizeof(float));

    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kp[((size_t)(j / 16) * d + k) * 16 + j % 16] = K[(size_t)j * d + k];
    for (size_t t = 0; t < (size_t)N * d; t++)
        Qs[t] = Q[t] * qscale;

    for (int i = 0; i < N; i += R)
        row_block(Qs + (size_t)i * d, Kp, V, O + (size_t)i * d, S, N, d);

    free(S);
    free(Qs);
    free(Kp);
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
