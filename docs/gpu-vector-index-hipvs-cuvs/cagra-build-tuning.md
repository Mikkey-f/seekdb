# CAGRA build tuning (seekdb × hipVS)

This directory documents how the CAGRA build configuration in
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

## Why

Measured on 250K x 128 and 250K x 256 (FP32, L2), end-to-end from
`CREATE VECTOR INDEX ... lib=cuvs` to the first `APPROXIMATE` query:

| dataset | baseline | this change | speedup |
| --- | --- | --- | --- |
| 250K x 128 | 46.8 s | 14.2 s | 3.3x |
| 250K x 256 | 47.6 s | 16.3 s | 2.9x |

The GPU build itself goes from 43 s to 11.3 s (~3.8x); the remainder of the
end-to-end time is seekdb's row scan, which is identical for both variants.

Recall is not traded away. Under the evaluation search setting both variants
score identically (1.0000/1.0000 for 128d, 1.0000/0.9999 for 256d). To expose
graph-quality differences the sweep also uses a deliberately weakened search
setting (`itopk=110`, `search_width=1`), which is the only way to tell the
graphs apart:

| dataset | baseline recall@10 / @100 | this change |
| --- | --- | --- |
| 250K x 128 (weak search) | 0.9969 / 0.9752 | **0.9977 / 0.9818** |
| 250K x 256 (weak search) | 0.9941 / 0.9550 | **0.9949 / 0.9584** |

Reducing `nn_descent_niter` alone makes the graph *worse* (e.g. 256/25 drops to
0.9225 in the weak setting); pairing it with an intermediate degree equal to the
final degree is what makes the faster build also a better graph. Going further
(`niter=16`) falls off the quality cliff (-0.005 to -0.006), which is why 25 is
the chosen value.

Alternatives measured and rejected:

| option | result | why rejected |
| --- | --- | --- |
| `build_algo = IVF_PQ` | 6.9 s (6x) | recall@100 0.978 strong / 0.785 weak |
| `AUTO_SELECT` | 14.1 s | 0.9120 weak, below baseline, algorithm not pinned |
| `ITERATIVE_CAGRA_SEARCH` | 45.6 s | slower than baseline |
| `graph_degree` 128 -> 64/96 | 9.5 / 10.8 s | changes persisted degree: slower queries, lower recall |

## Reproducing

Both scripts take a dataset directory containing `base.f32`/`query.f32`/`gt.i32`
(or `base.tsv` for the end-to-end one); `gen_data.c` in this directory generates
a calibrated dataset (low intrinsic dimension so nearest-neighbour ranks are
unambiguous, and a numerically lossless TSV round-trip).

```bash
# 1) isolated GPU parameter sweep, seconds per configuration
validate_cagra_sweep=./run_cagra_sweep.sh
$validate_cagra_sweep <dataset_dir> 250000 128

# 2) end-to-end A/B with two bridge builds
./run_build_profile.sh <dataset_dir> 128 \
    /path/to/baseline/libseekdb_cuvs_bridge.so \
    /path/to/candidate/libseekdb_cuvs_bridge.so 2
```

`run_build_profile.sh` prints `ddl_ms` (timer window as the competition defines
it), `build_us` (read back from the `serialized GPU CAGRA snapshot` log line) and
the measured recall. It also runs the recall measurement in `approx` mode; run
`measure_recall.py ... exact` once on a fresh dataset as a self-check — an exact
full-scan query must match `gt.i32` exactly, otherwise the dataset has near-ties
and recall numbers are meaningless.

## Verified invariants

- `lib=vsag` indexes still build and search, returning the same ids as the GPU
  index for the same data.
- After an observer restart the GPU CAGRA snapshot is deserialized and queried
  without rebuilding.
- No extra VRAM: a smaller intermediate graph strictly reduces device usage.
