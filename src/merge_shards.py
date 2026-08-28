"""Merge sharded run outputs back into one summary + records file."""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from summarize import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--task", required=True)
ap.add_argument("--out", default=str(ROOT / "outputs"))
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
task_cfg = cfg["tasks"][a.task]
base = f"{cfg['name']}__{a.task}"
outdir = Path(a.out)

shards = sorted(outdir.glob(f"{base}__s*of*.records.jsonl"))
if not shards:
    raise SystemExit(f"no shard records for {base}")

records, gen_s, gen_tok_tot, samp = [], 0.0, 0, 0
for sp in shards:
    rs = [json.loads(l) for l in sp.open()]
    records += rs
    sm = json.loads(sp.with_suffix("").with_suffix(".summary.json").read_text())
    gen_s = max(gen_s, sm["gen_seconds"])          # shards ran in parallel
    n_s = sum(r["n"] for r in rs)
    gen_tok_tot += sm["mean_gen_tokens"] * n_s
    samp += n_s

records.sort(key=lambda r: r["id"])
summary = summarize(records, cfg, a.task, task_cfg, extra={
    "mean_gen_tokens": gen_tok_tot / samp,
    "gen_seconds": gen_s,
    "merged_from_shards": len(shards),
})

(outdir / f"{base}.summary.json").write_text(json.dumps(summary, indent=2))
with (outdir / f"{base}.records.jsonl").open("w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"merged {len(shards)} shards -> {len(records)} problems")
print(json.dumps(summary, indent=2))
