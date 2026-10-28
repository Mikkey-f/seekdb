/*
 * Dataset generator + exact ground truth for the seekdb cuVS DDL harness.
 *
 * Data model: low-intrinsic-dimension latent factor model (a stand-in for real
 * embeddings such as SIFT / image descriptors). Points live on a K-dimensional
 * manifold inside `dim` ambient dimensions, so nearest-neighbour ranks are
 * well separated (no near-ties) and recall is a meaningful measurement.
 *
 * Writes into <outdir>:
 *   base.f32 / query.f32   float32, values are multiples of 1e-6 so the TSV
 *                          round-trip through LOAD DATA is lossless
 *   base.tsv / query.tsv   "id\t[v1,v2,...]" rows for LOAD DATA
 *   gt.i32                 nq*topk exact L2 neighbour ids (omp brute force)
 *
 * usage: ./gen_data <outdir> <n> <dim> <nq> <topk> <omp> [latent]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static unsigned long long rs = 88172645463325252ULL;
static inline unsigned long long rnd(void)
{
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
  return rs;
}
static inline double urnd(void) { return ((double)(rnd() % 1000000) + 1.0) / 1000001.0; }
static inline double nrnd(void)
{
  return sqrt(-2.0 * log(urnd())) * cos(6.283185307179586 * urnd());
}
static inline float q6(double x) { return (float)(round(x * 1000000.0) / 1000000.0); }

int main(int argc, char **argv)
{
  if (argc < 7) { fprintf(stderr, "usage: %s outdir n dim nq topk omp [latent]\n", argv[0]); return 2; }
  const char *out = argv[1];
  long n = atol(argv[2]), dim = atol(argv[3]), nq = atol(argv[4]), topk = atol(argv[5]);
  int nthr = atoi(argv[6]);
  long K = argc > 7 ? atol(argv[7]) : (dim >= 256 ? 20 : 12);
  if (K > dim) K = dim;  omp_set_num_threads(nthr);
  char p[1200];

  float *base = malloc(sizeof(float) * (size_t)n * dim);
  float *qry  = malloc(sizeof(float) * (size_t)nq * dim);
  double *A   = (K > 0) ? malloc(sizeof(double) * (size_t)dim * K) : NULL;
  double *z   = (K > 0) ? malloc(sizeof(double) * K) : NULL;
  if (!base || !qry) { fprintf(stderr, "oom\n"); return 1; }
  if (K > 0 && (!A || !z)) { fprintf(stderr, "oom\n"); return 1; }
  for (size_t i = 0; K > 0 && i < (size_t)dim * K; i++) A[i] = nrnd() / sqrt((double)K);

  double t0 = (double)clock() / CLOCKS_PER_SEC;
  if (K < 0) {
    /* isotropic mode: no latent projection, i.i.d. uniform cube -> worst case
       for ANN (distances concentrate), so graph quality differences show up */
    for (long i = 0; i < n; i++) {
      float *v = base + (size_t)i * dim;
      for (long d = 0; d < dim; d++) v[d] = (float)((rnd() % 20000) - 10000) / 10000.0f;
    }
    for (long q = 0; q < nq; q++) {
      const float *src = base + (size_t)(rnd() % n) * dim;
      float *v = qry + (size_t)q * dim;
      for (long d = 0; d < dim; d++) v[d] = q6(src[d] + 0.02 * nrnd());
    }
    printf("gen data (isotropic) %.1fs\n", (double)clock() / CLOCKS_PER_SEC - t0);
  } else {
  for (long i = 0; i < n; i++) {
    for (long k = 0; k < K; k++) z[k] = nrnd();
    float *v = base + (size_t)i * dim;
    for (long d = 0; d < dim; d++) {
      double acc = 0;
      for (long k = 0; k < K; k++) acc += A[(size_t)d * K + k] * z[k];
      v[d] = q6(acc);
    }
  }
  for (long q = 0; q < nq; q++) {
    /* perturbed latent point: near the manifold, slightly off, like a real query */
    for (long k = 0; k < K; k++) z[k] = nrnd();
    float *v = qry + (size_t)q * dim;
    for (long d = 0; d < dim; d++) {
      double acc = 0;
      for (long k = 0; k < K; k++) acc += A[(size_t)d * K + k] * z[k];
      v[d] = q6(acc + 0.02 * nrnd());
    }
  }
  printf("gen data %.1fs (latent=%ld)\n", (double)clock() / CLOCKS_PER_SEC - t0, K);
  }

  snprintf(p, sizeof(p), "%s/base.f32", out);
  FILE *f = fopen(p, "wb"); if (!f) { perror(p); return 1; }
  fwrite(base, sizeof(float), (size_t)n * dim, f); fclose(f);
  snprintf(p, sizeof(p), "%s/query.f32", out);
  f = fopen(p, "wb"); if (!f) { perror(p); return 1; }
  fwrite(qry, sizeof(float), (size_t)nq * dim, f); fclose(f);

  t0 = (double)clock() / CLOCKS_PER_SEC;
  snprintf(p, sizeof(p), "%s/base.tsv", out);
  f = fopen(p, "w"); if (!f) { perror(p); return 1; }
  for (long i = 0; i < n; i++) {
    fprintf(f, "%ld\t[", i);
    const float *v = base + (size_t)i * dim;
    for (long d = 0; d < dim; d++) fprintf(f, d ? ",%.6f" : "%.6f", v[d]);
    fputs("]\n", f);
  }
  fclose(f);
  snprintf(p, sizeof(p), "%s/query.tsv", out);
  f = fopen(p, "w"); if (!f) { perror(p); return 1; }
  for (long i = 0; i < nq; i++) {
    fprintf(f, "%ld\t[", i);
    const float *v = qry + (size_t)i * dim;
    for (long d = 0; d < dim; d++) fprintf(f, d ? ",%.6f" : "%.6f", v[d]);
    fputs("]\n", f);
  }
  fclose(f);
  printf("write tsv %.1fs\n", (double)clock() / CLOCKS_PER_SEC - t0);

  t0 = (double)clock() / CLOCKS_PER_SEC;
  int *gt = malloc(sizeof(int) * (size_t)nq * topk);
#pragma omp parallel for schedule(dynamic, 4)
  for (long q = 0; q < nq; q++) {
    float *bd = malloc(sizeof(float) * topk);
    for (long k = 0; k < topk; k++) { bd[k] = 1e30f; gt[q * topk + k] = -1; }
    const float *u = qry + (size_t)q * dim;
    for (long i = 0; i < n; i++) {
      const float *v = base + (size_t)i * dim;
      float acc = 0;
      for (long d = 0; d < dim; d++) { float t = u[d] - v[d]; acc += t * t; }
      if (acc < bd[topk - 1]) {
        long k = topk - 1;
        while (k > 0 && bd[k - 1] > acc) { bd[k] = bd[k - 1]; gt[q * topk + k] = gt[q * topk + k - 1]; k--; }
        bd[k] = acc; gt[q * topk + k] = (int)i;
      }
    }
    free(bd);
  }
  snprintf(p, sizeof(p), "%s/gt.i32", out);
  f = fopen(p, "wb"); if (!f) { perror(p); return 1; }
  fwrite(gt, sizeof(int), (size_t)nq * topk, f); fclose(f);
  printf("gt %.1fs\n", (double)clock() / CLOCKS_PER_SEC - t0);
  printf("DONE n=%ld dim=%ld nq=%ld topk=%ld latent=%ld\n", n, dim, nq, topk, K);
  return 0;
}
