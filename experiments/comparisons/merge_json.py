"""
--------------------------------------------------------------------------------
Merge per-cell / per-policy comparison JSONs (from SLURM array tasks) into
one figure JSON.

Rows are grouped by (N, p_gen, p_swap, n_ch, cutoff) and dict-merged, so an
agent-only file and a purify_swap-only file for the same cell become one row
with both policies' columns. `.meta.json` sidecars are skipped.

  PYTHONPATH=src:. python experiments/comparisons/merge_json.py \
      'results/comparisons/dvN_c30_H40000/*.json' -o results/comparisons/delivery_vs_N_c30_H40000.json
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, glob, json, os

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pattern", help="glob of per-task JSONs (quote it)")
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()

files = [f for f in sorted(glob.glob(a.pattern)) if not f.endswith(".meta.json")]
assert files, f"no files match {a.pattern}"
merged: dict[tuple, dict] = {}
for f in files:
    for row in json.load(open(f)):
        key = (row.get("N"), row.get("p_gen"), row.get("p_swap"),
               row.get("n_ch"), row.get("cutoff"))
        merged.setdefault(key, {}).update(row)
rows = [merged[k] for k in sorted(merged)]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rows, open(a.out, "w"), indent=2)
print(f"merged {len(files)} files -> {a.out} ({len(rows)} rows)")
