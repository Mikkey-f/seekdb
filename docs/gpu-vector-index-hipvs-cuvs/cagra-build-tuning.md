# CAGRA build tuning (seekdb × hipVS)

How the CAGRA build configuration in
[`tools/hipvs_bridge/seekdb_cuvs_bridge.c`](../../tools/hipvs_bridge/seekdb_cuvs_bridge.c)
was chosen, and how to reproduce the measurement.

## What changed

`seekdb_cuvs_build()` used to build a 2x wider intermediate graph and run 100
NN-descent refinement passes:

| parameter | frozen baseline | this change |
| --- | --- | --- |
| `graph_degree` | 128 | 128 (unchanged) |
| `intermediate_graph_degree` | 256 | **128** |
| `nn_descent_niter` | 100 | **25** |
| `build_algo` | `NN_DESCENT` | `NN_DESCENT` (unchanged) |

Both values are compile-time defaults in the bridge source, so the fixed
`tools/hipvs_bridge/build.sh` command reproduces them without extra flags.

`graph_degree` is deliberately left alone: it is the degree of the persisted
graph, so changing it would change query cost and search recall.

## Measurement profile matters

All end-to-end numbers below were taken with the observer pinned to **24 cores**
(`taskset -c 0-23`), because the evaluation harness limits the process to a
24-core slot. This is not a cosmetic detail: on the same machine and same code,
the frozen baseline takes **46.8 s at 128 cores but 20.9 s at 24 cores**. seekdb
spawns more DDL concurrency on a wide machine, and the extra scheduling and
contention more than eats the gain.

Quoting a speedup measured on a 128-core box would therefore overstate it by
~25%. The 24-core baseline (20.9 s for 250K x 128) also matches the 20-27 s
range published for SIFT250K on the evaluation machine, which is a good sign
that this is the right profile to report.

## Result

Measured on 250K x 128 and 250K x 256 (FP32, L2), end-to-end from
`CREATE VECTOR INDEX ... lib=cuvs` to the first `APPROXIMATE` query, two rounds,
baseline and candidate interleaved:

| dataset | baseline | this change | speedup |
| --- | --- | --- | --- |
| 250K x 128 | 20.9 s | 7.4 s | 2.84x |
| 250K x 256 | 23.3 s | 9.6 s | 2.43x |

The GPU build itself goes from 18.4 s to 4.35 s (128d) and 18.6 s to 4.88 s
(256d) -- about 4x. The end-to-end speedup is smaller because the rest of the
DDL window (row scan, LOB reads, per-row index insertion) is unchanged and now
dominates: 2.6 s at 128d and 4.9 s at 256d, which is 51% of the candidate's
256d build time.

## Recall is not traded away

Under the evaluation search setting both variants score identically (1.0000 /
1.0000 at 128d, 1.0000 / 0.9999 at 256d) -- that setting saturates, so it cannot
distinguish graph quality. The sweep therefore also runs a deliberately weakened
search (`itopk=110`, `search_width=1`), which is the only way to see the
difference:

| dataset | baseline recall@10 / @100 | this change |
| --- | --- | --- |
| 250K x 128 (weak search) | 0.9969 / 0.9752 | **0.9977 / 0.9818** |
| 250K x 256 (weak search) | 0.9941 / 0.9550 | **0.9949 / 0.9584** |

Lowering `nn_descent_niter` alone makes the graph worse (keeping the 256-degree
intermediate graph and dropping to 25 passes reaches only 0.9225 in the same
test). Equalising the two degrees is what makes the faster build also a better
graph. Going further (niter=16) falls off the quality cliff -- 0.9495 at 256d,
which is 0.0055 below baseline and outside the competition's 0.005 tolerance --
which is exactly why 25 is the chosen value.

Alternatives measured and rejected:

| option | result | why rejected |
| --- | --- | --- |
| `build_algo = IVF_PQ` | 6.9 s | recall@100 0.978 strong / 0.785 weak |
| `AUTO_SELECT` | 14.1 s | 0.9120 weak, below baseline, algorithm not pinned |
| `ITERATIVE_CAGRA_SEARCH` | 45.6 s | slower than baseline |
| `graph_degree` 128 -> 64/96 | 9.5 / 10.8 s | changes persisted degree: slower queries, lower recall |

## Reproducing

Both scripts take a dataset directory containing `base.f32`/`query.f32`/`gt.i32`
(or `base.tsv` for the end-to-end one); `gen_data.c` in this directory generates
a calibrated dataset.

```bash
# 1) isolated GPU parameter sweep, seconds per configuration
./run_cagra_sweep.sh <dataset_dir> 250000 128

# 2) end-to-end A/B with two bridge builds, on the evaluation's core budget
taskset -c 0-23 ./run_build_profile.sh <dataset_dir> 128 \
    baseline/libseekdb_cuvs_bridge.so candidate/libseekdb_cuvs_bridge.so 2
```

`run_build_profile.sh` prints `ddl_ms` (the timer window as the competition
defines it), `build_us` (read back from the `serialized GPU CAGRA snapshot` log
line) and the measured recall.

### Dataset calibration (important)

`gen_data.c` puts points on a low-dimensional latent manifold (`v = A . z`) and
derives queries by perturbing latent points. This matters: with uniform or
clustered Gaussian data the distances concentrate, ranks tie, and even an **exact
full-table scan** only agrees with the ground truth at recall 0.05 -- at which
point any recall number is meaningless. `measure_recall.py ... exact` is the
self-check for exactly this failure: on a correctly calibrated dataset it must
return 1.0000 before any `approx` number is worth reading.

## Verified invariants

- `lib=vsag` indexes still build and search, returning the same ids as the GPU
  index for the same data.
- After an observer restart the GPU CAGRA snapshot is deserialized and queried
  without rebuilding.
- No extra VRAM: a smaller intermediate graph strictly reduces device usage.

## What is left on the table

The per-row feed into the adaptor (`ob_ddl_pipeline.cpp` calls `add_snap_index`
with `num = 1` for every row, although the function accepts a batch) is the
remaining bottleneck. Batching it should recover most of the 2.6-4.9 s and push
the end-to-end speedup past 3x. It was not done here because it touches the DDL
data path shared with `lib=vsag`, and a regression there would fail a hard
gate.
