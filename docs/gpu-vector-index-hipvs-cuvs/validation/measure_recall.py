#!/usr/bin/env python3
"""Recall measurement for the CAGRA build profile harness.

Runs every query through one mysql batch (each statement tagged with its index)
and compares against the exact ground truth in <dataset>/gt.i32.

usage: measure_recall.py <dataset_dir> <port> <dim> <nq> <topk> <approx|exact> [table]

  approx  -- the APPROXIMATE (GPU) query used by the competition harness
  exact   -- a full-scan query, used as a self-check: it must match gt.i32
             exactly, otherwise the dataset (near-ties) or the ground truth is
             wrong and a low approx recall would be meaningless.
"""
import struct
import subprocess
import sys

ds, port, dim, nq, topk, mode = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6],
)
table = sys.argv[7] if len(sys.argv) > 7 else "items"

vals = struct.unpack("<%df" % (nq * dim), open("%s/query.f32" % ds, "rb").read(nq * dim * 4))
gt = struct.unpack("<%di" % (nq * topk), open("%s/gt.i32" % ds, "rb").read(nq * topk * 4))

lines = ["use vec;"]
for q in range(nq):
    lit = "[" + ",".join("%.6f" % v for v in vals[q * dim:(q + 1) * dim]) + "]"
    hint = " approximate" if mode == "approx" else ""
    lines.append(
        "select %d as q, id from %s order by l2_distance(v,%s)%s limit %d;"
        % (q, table, lit, hint, topk)
    )

p = subprocess.run(
    ["mysql", "-N", "-h127.0.0.1", "-P" + port, "-uroot"],
    input="\n".join(lines) + "\n", capture_output=True, text=True,
)
got = {}
for ln in p.stdout.splitlines():
    parts = ln.split("\t")
    if len(parts) != 2:
        continue
    got.setdefault(int(parts[0]), []).append(int(parts[1]))
if p.returncode != 0:
    sys.stderr.write(p.stderr[-1500:])


def recall_at(k):
    hit = tot = 0
    for q in range(nq):
        ids = got.get(q, [])[:k]
        truth = set(gt[q * topk: q * topk + k])
        hit += sum(1 for x in ids if x in truth)
        tot += k
    return hit / tot if tot else 0.0


steps = [10] + ([100] if topk >= 100 else [])
short = sum(1 for q in range(nq) if len(got.get(q, [])) < topk)
dupe = sum(1 for lst in got.values() if len(set(lst)) != len(lst))
print("mode=%s queries=%d full_rows=%d/%d dup_queries=%d"
      % (mode, nq, nq - short, nq, dupe))
for k in steps:
    print("recall@%d=%.4f" % (k, recall_at(k)))
