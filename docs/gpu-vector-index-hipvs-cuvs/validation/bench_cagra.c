/*
 * CAGRA parameter sweep on a fixed dataset (so recall matches the DB harness).
 *
 * usage: bench2 <dir> <n> <dim> <nq> <topk> <algo> <gd> <igd> <niter> [more configs...]
 *        algo: 0=AUTO 1=IVF_PQ 2=NN_DESCENT 3=ITERATIVE_CAGRA_SEARCH
 *
 * For each config: builds the index once (timed), searches (timed), prints
 * recall@10 / recall@100 against <dir>/gt.i32.
 */
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#include <cuvs/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_s(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void set_t(DLManagedTensor *t, void *data, DLDeviceType dev, DLDataTypeCode code,
                  int bits, long r, long c, int64_t *shape)
{
  memset(t, 0, sizeof(*t));
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = dev;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = code;
  t->dl_tensor.dtype.bits = bits;
  t->dl_tensor.dtype.lanes = 1;
  shape[0] = r; shape[1] = c;
  t->dl_tensor.shape = shape;
  t->dl_tensor.strides = 0;
  t->dl_tensor.byte_offset = 0;
}

static float *load_f32(const char *path, size_t count)
{
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  float *b = malloc(sizeof(float) * count);
  if (!b || fread(b, sizeof(float), count, f) != count) { fprintf(stderr, "short read %s\n", path); exit(1); }
  fclose(f);
  return b;
}

int main(int argc, char **argv)
{
  if (argc < 10) { fprintf(stderr, "usage: %s dir n dim nq topk algo gd igd niter ...\n", argv[0]); return 2; }
  const char *dir = argv[1];
  long n = atol(argv[2]), dim = atol(argv[3]), nq = atol(argv[4]), topk = atol(argv[5]);

  char p[1200];
  snprintf(p, sizeof(p), "%s/base.f32", dir);  float *base = load_f32(p, (size_t)n * dim);
  snprintf(p, sizeof(p), "%s/query.f32", dir); float *qry  = load_f32(p, (size_t)nq * dim);
  snprintf(p, sizeof(p), "%s/gt.i32", dir);
  FILE *gf = fopen(p, "rb");
  if (!gf) { perror(p); return 1; }
  int *gt = malloc(sizeof(int) * (size_t)nq * topk);
  if (fread(gt, sizeof(int), (size_t)nq * topk, gf) != (size_t)nq * topk) { fprintf(stderr, "gt short\n"); return 1; }
  fclose(gf);
  printf("dataset=%s n=%ld dim=%ld nq=%ld topk=%ld\n", dir, n, dim, nq, topk);

  cuvsResources_t res;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { fprintf(stderr, "res fail\n"); return 1; }

  float *qd = 0; unsigned *nd = 0; float *dd = 0;
  cuvsRMMAlloc(res, (void **)&qd, sizeof(float) * nq * dim);
  cuvsRMMAlloc(res, (void **)&nd, sizeof(unsigned) * nq * topk);
  cuvsRMMAlloc(res, (void **)&dd, sizeof(float) * nq * topk);
  unsigned *host_ids = malloc(sizeof(unsigned) * (size_t)nq * topk);
  int64_t qs[2], osh[2];

  int ci = 0;
  for (int a = 6; a + 3 < argc; a += 4, ci++) {
    int algo = atoi(argv[a]);
    long gd = atol(argv[a + 1]), igd = atol(argv[a + 2]), niter = atol(argv[a + 3]);

    cuvsCagraIndexParams_t ip;
    cuvsCagraIndexParamsCreate(&ip);
    ip->build_algo = (algo == 1) ? IVF_PQ : (algo == 2 ? NN_DESCENT
                     : (algo == 3 ? ITERATIVE_CAGRA_SEARCH : AUTO_SELECT));
    ip->graph_degree = (size_t)gd;
    ip->intermediate_graph_degree = (size_t)igd;
    ip->nn_descent_niter = (size_t)niter;

    cuvsCagraIndex_t idx;
    cuvsCagraIndexCreate(&idx);
    int64_t ds[2];
    DLManagedTensor dset;
    set_t(&dset, base, kDLCPU, kDLFloat, 32, n, dim, ds);

    double t0 = now_s();
    cuvsError_t be = cuvsCagraBuild(res, ip, &dset, idx);
    double tb = now_s() - t0;
    if (be != CUVS_SUCCESS) {
      printf("cfg%d algo=%d gd=%ld igd=%ld niter=%ld BUILD_FAIL %s\n",
             ci, algo, gd, igd, niter, cuvsGetLastErrorText());
      fflush(stdout);
      cuvsCagraIndexDestroy(idx); cuvsCagraIndexParamsDestroy(ip);
      continue;
    }

    cudaMemcpy(qd, qry, sizeof(float) * nq * dim, cudaMemcpyDefault);
    DLManagedTensor qt, nt, dt;
    set_t(&qt, qd, kDLCUDA, kDLFloat, 32, nq, dim, qs);
    set_t(&nt, nd, kDLCUDA, kDLUInt, 32, nq, topk, osh);
    set_t(&dt, dd, kDLCUDA, kDLFloat, 32, nq, topk, osh);
    cuvsCagraSearchParams_t sp;
    cuvsCagraSearchParamsCreate(&sp);
    size_t req = (size_t)topk * 4;
    if (req < 512) req = 512;
    {
      const char *e = getenv("BENCH_ITOPK");
      if (e && atoi(e) > 0) req = (size_t)atoi(e);
      if (req < (size_t)topk) req = (size_t)topk;
    }
    if (sp->itopk_size < req) sp->itopk_size = req;
    sp->search_width = 4;
    {
      const char *e = getenv("BENCH_WIDTH");
      if (e && atoi(e) > 0) sp->search_width = (size_t)atoi(e);
    }
    cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;

    double best = 1e9;
    cuvsError_t rc = CUVS_SUCCESS;
    for (int r = 0; r < 2; r++) {
      double s0 = now_s();
      rc = cuvsCagraSearch(res, sp, idx, &qt, &nt, &dt, f);
      double el = now_s() - s0;
      if (el < best) best = el;
    }
    if (rc != CUVS_SUCCESS) {
      printf("cfg%d algo=%d gd=%ld igd=%ld niter=%ld SEARCH_FAIL\n", ci, algo, gd, igd, niter);
      fflush(stdout);
      cuvsCagraSearchParamsDestroy(sp); cuvsCagraIndexDestroy(idx); cuvsCagraIndexParamsDestroy(ip);
      continue;
    }
    cudaMemcpy(host_ids, nd, sizeof(unsigned) * (size_t)nq * topk, cudaMemcpyDefault);

    double r10 = 0, r100 = 0;
    long h10 = 0, h100 = 0, t10 = 0, t100 = 0;
    for (long q = 0; q < nq; q++) {
      for (long k = 0; k < 10; k++) {
        int got = (int)host_ids[q * topk + k];
        t10++;
        for (long j = 0; j < 10; j++) if (gt[q * topk + j] == got) { h10++; break; }
      }
      for (long k = 0; k < topk; k++) {
        int got = (int)host_ids[q * topk + k];
        t100++;
        for (long j = 0; j < topk; j++) if (gt[q * topk + j] == got) { h100++; break; }
      }
    }
    r10 = t10 ? (double)h10 / t10 : 0.0;
    r100 = t100 ? (double)h100 / t100 : 0.0;
    printf("cfg%d algo=%d gd=%ld igd=%ld niter=%ld build_s=%.2f search_ms=%.2f recall@10=%.4f recall@100=%.4f\n",
           ci, algo, gd, igd, niter, tb, best * 1000.0, r10, r100);
    fflush(stdout);

    cuvsCagraSearchParamsDestroy(sp);
    cuvsCagraIndexDestroy(idx);
    cuvsCagraIndexParamsDestroy(ip);
  }

  cuvsRMMFree(res, qd, sizeof(float) * nq * dim);
  cuvsRMMFree(res, nd, sizeof(unsigned) * nq * topk);
  cuvsRMMFree(res, dd, sizeof(float) * nq * topk);
  cuvsResourcesDestroy(res);
  return 0;
}
