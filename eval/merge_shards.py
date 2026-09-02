"""Merge a sharded evaluation back into one summary + records pair.

A sharded run writes {tag}__sNofM.{summary.json,records.jsonl,generations.jsonl}
per shard; this recombines the records, recomputes every metric over the whole
problem set, and removes the shard files.

Recomputing rather than averaging matters: pass@k is a mean over problems, and
averaging eight shard means is only the same number when the shards are equal
sized, which round-robin sharding does not guarantee.

  python eval/merge_shards.py --config configs/eval/llama32-3b__aime.yaml
  python eval/merge_shards.py --name <tag> --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from metrics import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None, help="the config the shards were run under")
ap.add_argument("--name", default=None,
                help="output tag, if it was not the config's own; must match what "
                     "the shards were written under")
ap.add_argument("--out", default=str(ROOT / "outputs"))
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config)) if a.config else {}
tag = a.name or (f"{cfg.get('model_label') or Path(cfg['model']).name}__{cfg['label']}"
                 if cfg else None)
if not tag:
    raise SystemExit("no tag: give --name, or --config for a config that sets "
                     "model_label/label.")
outdir = Path(a.out) / cfg.get("out_subdir", "")

shards = sorted(outdir.glob(f"{tag}__s*of*.records.jsonl"))
if not shards:
    raise SystemExit(f"no shard records matching {outdir}/{tag}__s*of*.records.jsonl")

records, gen_s, models = [], 0.0, set()
for sp in shards:
    records += [json.loads(l) for l in sp.open()]
    sm = json.loads(sp.with_suffix("").with_suffix(".summary.json").read_text())
    gen_s = max(gen_s, sm["gen_seconds"])       # shards ran in parallel
    models.add(sm.get("model"))

# summarize() would otherwise take the model from nowhere, and a merged
# checkpoint evaluation would claim to be whatever the config names. Shards that
# loaded different models must not be averaged into one score at all.
assert len(models) == 1, f"shards evaluated different models: {sorted(models)}"

records.sort(key=lambda r: r["id"])
summary = summarize(records, extra={
    "tag": tag, "model": models.pop(), "gen_seconds": gen_s,
    "merged_from_shards": len(shards),
    "label": cfg.get("label"), "verifier": cfg.get("verifier"),
    "headline_metric": cfg.get("headline_metric"),
})
(outdir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
with (outdir / f"{tag}.records.jsonl").open("w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

ks = [k for k in summary if k.startswith("pass@") and not k.endswith("_stderr")]
print(f"merged {len(shards)} shards -> {len(records)} problems")
print("  " + "  ".join(f"{k}={100*summary[k]:.1f}" for k in ks))
print(f"  -> {outdir}/{tag}.summary.json")
