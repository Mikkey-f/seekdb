#!/usr/bin/env bash
# CAGRA build/search parameter sweep on one fixed dataset.
#
#   ./run_cagra_sweep.sh <dataset_dir> <n> <dim> [nq] [topk]
#
# <dataset_dir> must contain base.f32 / query.f32 / gt.i32 (gen_data.c in this
# directory generates a calibrated dataset of that shape).
#
# Each configuration is (algo, graph_degree, intermediate_degree, niter) with
# algo: 0=AUTO_SELECT 1=IVF_PQ 2=NN_DESCENT 3=ITERATIVE_CAGRA_SEARCH.
#
# Two search settings are used on purpose:
#   strong (itopk=512, width=4)  -- the competition setting
#   weak   (itopk=110, width=1)  -- amplifies graph-quality differences, which
#                                   the strong setting hides because recall
#                                   saturates at 1.0 for every graph
set -Eeuo pipefail

DIR=${1:?dataset dir required}
N=${2:?row count required}
DIM=${3:?dimension required}
NQ=${4:-256}
TOPK=${5:-100}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
WORK=${WORK:-/tmp/cagra-sweep}
mkdir -p "$WORK"

export LD_LIBRARY_PATH="$HIPVS_ROOT/lib:$ROCM_ROOT/lib:${LD_LIBRARY_PATH:-}"

echo "[sweep] building bench_cagra"
"$ROCM_ROOT/bin/hipcc" -O2 -D__HIP_PLATFORM_AMD__ \
  -I"$HIPVS_ROOT/include" -I"$ROCM_ROOT/include" \
  "$SCRIPT_DIR/bench_cagra.c" \
  -L"$HIPVS_ROOT/lib" -Wl,-rpath,"$HIPVS_ROOT/lib:$ROCM_ROOT/lib" -lcuvs_c \
  -o "$WORK/bench_cagra"

CFG="2 128 256 100
2 128 256 50
2 128 256 25
2 128 128 100
2 128 128 25
2 128 128 16
1 128 256 100
0 128 256 100"

echo "[sweep] dataset=$DIR n=$N dim=$DIM nq=$NQ topk=$TOPK"
echo
echo "[sweep] strong search (itopk=512 width=4, the evaluation setting)"
BENCH_ITOPK=512 BENCH_WIDTH=4 "$WORK/bench_cagra" "$DIR" "$N" "$DIM" "$NQ" "$TOPK" $CFG
echo
echo "[sweep] weak search (itopk=110 width=1) -- exposes graph-quality gaps"
BENCH_ITOPK=110 BENCH_WIDTH=1 "$WORK/bench_cagra" "$DIR" "$N" "$DIM" "$NQ" "$TOPK" $CFG
