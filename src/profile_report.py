"""Difficulty profile over the MATH train pool: how many problems sit in each
empirical pass@1 band, which is what decides the trainable RL subset.

A prompt only produces GRPO gradient when 0 < pass@1 < 1: all-correct and
all-wrong groups both give zero advantage.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--records", required=True)
ap.add_argument("--target", type=float, default=0.08, help="blog's hard-subset pass@1")
ap.add_argument("--band", type=float, default=0.04, help="+/- window around target")
a = ap.parse_args()

recs = [json.loads(l) for l in open(a.records)]
n = recs[0]["n"]
N = len(recs)
for r in recs:
    r["p"] = r["c"] / r["n"]

print(f"records={N}  samples/problem={n}  resolution={1/n:.3%}")
print(f"mean pass@1 = {sum(r['p'] for r in recs)/N:.3%}")
print()

print("=== pass@1 分布 ===")
# half-open (lo, hi] so the bands partition exactly; p==0 and p==1 are their own rows
edges = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
rows = [("p = 0%", [r for r in recs if r["p"] == 0])]
lo = 0.0
for hi in edges:
    sel = [r for r in recs if lo < r["p"] < hi] if hi == 1.0 else \
          [r for r in recs if lo < r["p"] <= hi]
    rows.append((f"{lo:.0%} < p < 100%" if hi == 1.0 else f"{lo:.0%} < p <= {hi:.0%}", sel))
    lo = hi
rows.append(("p = 100%", [r for r in recs if r["p"] == 1]))
tot = 0
for label, sel in rows:
    tot += len(sel)
    print(f"  {label:22} {len(sel):6}  {100*len(sel)/N:5.1f}%")
print(f"  {'--- sum (must = N)':22} {tot:6}  {'OK' if tot == N else 'MISMATCH'}")

zero = sum(1 for r in recs if r["p"] == 0)
one = sum(1 for r in recs if r["p"] == 1)
print()
print(f"  全错 (p=0, GRPO 零梯度):  {zero:6}  {100*zero/N:5.1f}%")
print(f"  全对 (p=1, GRPO 零梯度):  {one:6}  {100*one/N:5.1f}%")
print(f"  可训练 (0<p<1):           {N-zero-one:6}  {100*(N-zero-one)/N:5.1f}%")

lo, hi = a.target - a.band, a.target + a.band
hard = [r for r in recs if lo <= r["p"] <= hi]
print()
print(f"=== 论文口径 hard subset: pass@1 in [{lo:.1%}, {hi:.1%}] ===")
print(f"  命中 {len(hard)} 道 ({100*len(hard)/N:.1f}%)")
if hard:
    print("  level 分布:", dict(sorted(Counter(str(r.get('level')) for r in hard).items())))
    print("  subject 分布:", dict(sorted(Counter(str(r.get('subject')) for r in hard).items())))

print()
print("=== 按 level 的实测 pass@1 ===")
for lv in sorted(set(str(r.get("level")) for r in recs)):
    sel = [r for r in recs if str(r.get("level")) == lv]
    p = sum(x["p"] for x in sel) / len(sel)
    z = sum(1 for x in sel if x["p"] == 0)
    print(f"  Level {lv}: n={len(sel):5}  mean pass@1={p:6.1%}   全错={100*z/len(sel):4.1f}%")

nt = sum(s["truncated"] for r in recs for s in r["samples"])
ne = sum(s["extracted"] is None for r in recs for s in r["samples"])
tot = sum(r["n"] for r in recs)
print()
print(f"harness 健康度: truncation={100*nt/tot:.1f}%  no_answer={100*ne/tot:.1f}%")
