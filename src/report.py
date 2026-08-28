"""Aggregate all summary json files into the final results table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = []
import re
SHARD = re.compile(r"__s\d+of\d+\.summary\.json$")
for p in sorted((ROOT / "outputs").glob("*.summary.json")):
    if SHARD.search(p.name):
        continue          # per-shard partials; the merged file supersedes them
    s = json.loads(p.read_text())
    headline = "pass@1" if s["task"] == "math500" else "pass@4"
    rows.append({
        "config": s["config"], "task": s["task"], "n": s["n_problems"],
        "samples": s["samples_per_problem"], "metric": headline,
        "score": s.get(headline, float("nan")),
        "stderr": s.get(f"{headline}_stderr", float("nan")),
        "pass@1": s.get("pass@1"), "trunc": s["truncation_rate"],
        "no_ans": s["no_answer_rate"], "gen_tok": s["mean_gen_tokens"],
    })

if not rows:
    raise SystemExit("no summaries in outputs/")

hdr = f"{'config':<20}{'task':<10}{'metric':<9}{'score':>9}{'±stderr':>10}{'pass@1':>9}{'trunc':>8}{'no_ans':>8}{'gen_tok':>9}"
print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda x: (x["task"], x["config"])):
    print(f"{r['config']:<20}{r['task']:<10}{r['metric']:<9}"
          f"{100*r['score']:>8.1f}%{100*r['stderr']:>9.1f}%"
          f"{100*r['pass@1']:>8.1f}%{100*r['trunc']:>7.1f}%"
          f"{100*r['no_ans']:>7.1f}%{r['gen_tok']:>9.0f}")
