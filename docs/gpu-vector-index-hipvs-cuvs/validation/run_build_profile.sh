#!/usr/bin/env bash
# End-to-end A/B of the index build path: frozen baseline bridge vs a candidate
# bridge, on one machine and one dataset, following the competition timing rules
# (timer starts at CREATE VECTOR INDEX, stops after the first APPROXIMATE query).
#
#   ./run_build_profile.sh <dataset_dir> <dim> <baseline_bridge.so> <candidate_bridge.so> [rounds]
#
# <dataset_dir> must contain base.tsv / query.f32 / gt.i32.
# Prints one line per round:
#   ds=<dim> variant=<v> ddl_ms=<n> build_us=<n> recall@10=<f> recall@100=<f>
set -Eeuo pipefail

DIR=${1:?dataset dir required}
DIM=${2:?dimension required}
BASE_SO=${3:?baseline bridge .so required}
CAND_SO=${4:?candidate bridge .so required}
ROUNDS=${5:-2}

SEEKDB=${SEEKDB:-$PWD/build_release/src/observer/seekdb}
PORT=${PORT:-2891}
NQ_RECALL=${NQ_RECALL:-128}
TOPK=${TOPK:-100}

[[ -x "$SEEKDB" ]] || { echo "observer not found: $SEEKDB" >&2; exit 2; }

run_variant() {
  local variant=$1 so=$2 tag=$3
  local libdir work
  libdir=$(mktemp -d)
  cp "$so" "$libdir/libseekdb_cuvs_bridge.so"
  work=/tmp/build-profile-$tag
  rm -rf "$work"; mkdir -p "$work"

  export LD_LIBRARY_PATH="$libdir:/opt/hipvs/lib:/opt/rocm/lib"
  pkill -f "seekdb --base-dir=$work" 2>/dev/null || true
  sleep 2
  ( cd "$work" && nohup "$SEEKDB" --base-dir="$work" --port="$PORT" --nodaemon \
      --parameter _memory_budget=32G --parameter datafile_size=20G \
      --parameter datafile_maxsize=40G --parameter log_disk_size=10G \
      --log-level INFO > "$work/out.log" 2>&1 & )
  for _ in $(seq 1 90); do grep -q "bootstrap success" "$work/log/seekdb.log" 2>/dev/null && break; sleep 2; done
  for _ in $(seq 1 30); do mysql -h127.0.0.1 -P"$PORT" -uroot -e "select 1" >/dev/null 2>&1 && break; sleep 1; done

  local M="mysql --local-infile=1 -h127.0.0.1 -P$PORT -uroot"
  $M -e "create database if not exists vec; use vec;
         drop table if exists items;
         create table items(id bigint primary key, v vector($DIM));
         alter system set ob_vector_memory_limit_percentage=30;" 2>/dev/null
  $M -e "use vec; load data local infile '$DIR/base.tsv' into table items
         fields terminated by '\t' lines terminated by '\n' (id,v);" 2>/dev/null

  local probe t0 t1 tddl tq
  probe=$($M -N -e "use vec; select v from items where id=0" 2>/dev/null)
  t0=$(date +%s%N)
  $M -e "use vec; create vector index idx on items(v)
         with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64);" 2>/dev/null
  tddl=$(date +%s%N)
  $M -N -e "use vec; select id from items order by l2_distance(v,$probe) approximate limit 10;" >/dev/null 2>&1
  tq=$(date +%s%N)

  local bus
  bus=$(grep -ao 'build_us=[0-9]*' "$work/log/seekdb.log" | tail -1 | cut -d= -f2)
  local rec
  rec=$(python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/measure_recall.py" \
          "$DIR" "$PORT" "$DIM" "$NQ_RECALL" "$TOPK" approx 2>/dev/null | tr '\n' ' ')

  echo "ds=$DIM variant=$variant ddl_ms=$(( (tq - t0) / 1000000 )) ddl_only_ms=$(( (tddl - t0) / 1000000 )) build_us=${bus:-0} $rec"
  pkill -f "seekdb --base-dir=$work" 2>/dev/null || true
  sleep 4
  rm -rf "$libdir"
}

for r in $(seq 1 "$ROUNDS"); do
  echo "# round $r"
  run_variant baseline "$BASE_SO" "base-$r"
  run_variant candidate "$CAND_SO" "cand-$r"
done
