"""Merge one sharded evaluation directory into a single result.

A sharded run writes summary.sNofM.json / records.sNofM.jsonl /
generations.sNofM.jsonl into its output directory; this recombines them into
summary.json / records.jsonl / generations.jsonl and removes the shard files.

Metrics are recomputed over the whole problem set rather than averaged across
shards: pass@k is a mean over problems, and averaging shard means is only the
same number when the shards are equal sized, which round-robin sharding does
not guarantee.

  python eval/merge_shards.py --output-path outputs/grpo/<exp>__step20__aime
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from metrics import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--output-path", required=True, help="the run's output directory")
ap.add_argument("--keep-shards", action="store_true",
                help="leave the per-shard files in place")
a = ap.parse_args()

d = Path(a.output_path)
shards = sorted(d.glob("records.s*of*.jsonl"))
if not shards:
    raise SystemExit(f"no shard records in {d} (records.sNofM.jsonl)")

records, gen_s, models, meta = [], 0.0, set(), {}
for sp in shards:
    records += [json.loads(l) for l in sp.open()]
    sm = json.loads((sp.parent / sp.name.replace("records.", "summary.")
                     .replace(".jsonl", ".json")).read_text())
    gen_s = max(gen_s, sm["gen_seconds"])       # shards ran in parallel
    models.add(sm.get("model"))
    meta = sm

# Shards that loaded different models must not be averaged into one score, and
# the merged summary must name the model that was actually evaluated rather than
# whatever a config says.
assert len(models) == 1, f"shards evaluated different models: {sorted(models)}"

records.sort(key=lambda r: r["id"])
summary = summarize(records, extra={
    "model": models.pop(), "label": meta.get("label"),
    "verifier": meta.get("verifier"), "sampling": meta.get("sampling"),
    "headline_metric": meta.get("headline_metric"),
    "data_files": meta.get("data_files"),
    "gen_seconds": gen_s, "merged_from_shards": len(shards),
})
(d / "summary.json").write_text(json.dumps(summary, indent=2))
with (d / "records.jsonl").open("w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with (d / "generations.jsonl").open("w") as f:
    for sp in shards:
        g = sp.parent / sp.name.replace("records.", "generations.")
        if g.exists():
            f.write(g.read_text())

if not a.keep_shards:
    for sp in shards:
        for kind in ("records", "summary", "generations"):
            q = sp.parent / sp.name.replace("records.", kind + ".")
            if kind == "summary":
                q = q.with_suffix(".json") if q.suffix == ".jsonl" else q
            q.unlink(missing_ok=True)

ks = [k for k in summary if k.startswith("pass@") and not k.endswith("_stderr")]
print(f"merged {len(shards)} shards -> {len(records)} problems in {d}")
print("  " + "  ".join(f"{k}={100*summary[k]:.1f}" for k in ks))
