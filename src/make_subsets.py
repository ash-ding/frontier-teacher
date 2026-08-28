"""Sample difficulty-banded training subsets from a profiled model's records.

Pool policy differs per model, on purpose:

  llama32-3b        -> CLEAN pool only (split_origin == "orig_test", 4,498).
                       The provenance test found positive evidence of contamination:
                       61.0% pass@1 on the original train split vs 40.3% on the
                       original test split vs 38.8% on held-out MATH-500.

  qwen3-4b-nothink  -> FULL pool (11,996). The same test found no asymmetry
                       (82.5 / 84.2 / 83.3), so there is no basis for discarding
                       the original-train problems. Caveat: a differential test is
                       blind to UNIFORM exposure, so this is the absence of
                       differential evidence, not proof the pool is clean.

Sampling is uniform without replacement. Each band draws from its own seeded
stream, so changing one band's size never perturbs another's draw.
"""
import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "llama32-3b": {
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "n_samples": 32,
        "pool": "clean",
        "pool_size": 4498,
        "pool_note": "split_origin == orig_test; original-train problems excluded (contamination)",
        "sampling": {"temperature": 0.6, "top_p": 0.9, "max_tokens": 4096},
        "bands": [
            ("pass1_eq_0",     "p = 0",           lambda p: p == 0.0,          814, 800),
            ("pass1_05-15pct", "5% <= p < 15%",   lambda p: 0.05 <= p < 0.15,  487, 400),
            ("pass1_40-60pct", "40% <= p <= 60%", lambda p: 0.40 <= p <= 0.60, 570, 500),
            ("pass1_85-95pct", "85% < p < 95%",   lambda p: 0.85 < p < 0.95,   437, 400),
            ("pass1_eq_1",     "p = 1",           lambda p: p == 1.0,          131, 100),
        ],
    },
    "qwen3-4b-nothink": {
        "model_id": "Qwen/Qwen3-4B (enable_thinking=false)",
        "n_samples": 8,
        "pool": "full",
        "pool_size": 11996,
        "pool_note": "full MATH train pool; no differential contamination detected",
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 8192},
        # exact values: at n=8 pass@1 is quantised to c/8, so these are single points
        "bands": [
            ("pass1_eq_0",      "p = 0",     lambda p: p == 0.0,   779,  700),
            ("pass1_eq_25pct",  "p = 25%",   lambda p: p == 0.25,  288,  200),
            ("pass1_eq_50pct",  "p = 50%",   lambda p: p == 0.50,  294,  200),
            ("pass1_eq_75pct",  "p = 75%",   lambda p: p == 0.75,  510,  500),
            ("pass1_eq_1",      "p = 1",     lambda p: p == 1.0,  8199,  500),
        ],
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=sorted(MODELS))
ap.add_argument("--records", default="")
ap.add_argument("--out", default="")
ap.add_argument("--seed", type=int, default=20260828)
args = ap.parse_args()

spec = MODELS[args.model]
recs_path = args.records or ROOT / "outputs" / "math_profiling" / f"{args.model}__mathtrain.records.jsonl"
outdir = Path(args.out or ROOT / "data" / "further_improve" / args.model)

recs = [json.loads(l) for l in open(recs_path)]
if spec["pool"] == "clean":
    missing = [r for r in recs if "split_origin" not in r]
    if missing:
        raise SystemExit(f"{len(missing)} records lack split_origin — re-tag before sampling")
    pool_all = [r for r in recs if r["split_origin"] == "orig_test"]
else:
    pool_all = recs
print(f"{args.model}: records={len(recs)}  pool={spec['pool']} ({len(pool_all)})")
assert len(pool_all) == spec["pool_size"], \
    f"pool is {len(pool_all)}, expected {spec['pool_size']}"

outdir.mkdir(parents=True, exist_ok=True)
manifest = {
    "model_tag": args.model,
    "model_id": spec["model_id"],
    "profiled_with": dict(spec["sampling"], samples_per_problem=spec["n_samples"],
                          engine="vLLM 0.27.1", seed=1234),
    "source_pool": "nlile/hendrycks-MATH-benchmark train split, deduplicated to 11,996",
    "pool_policy": spec["pool"],
    "pool_note": spec["pool_note"],
    "pool_size": len(pool_all),
    "sampling": {"method": "uniform without replacement", "seed": args.seed},
    "subsets": [],
}

for slug, label, pred, expect_pool, k in spec["bands"]:
    pool = [r for r in pool_all if pred(r["c"] / r["n"])]
    assert len(pool) == expect_pool, f"{label}: pool is {len(pool)}, expected {expect_pool}"
    assert k <= len(pool), f"{label}: cannot draw {k} from {len(pool)}"

    rng = random.Random(f"{args.seed}:{slug}" if args.model == "llama32-3b"
                        else f"{args.seed}:{args.model}:{slug}")
    picked = sorted(rng.sample(pool, k), key=lambda r: r["id"])

    fname = f"{args.model}__{slug}__n{k}.jsonl"
    with (outdir / fname).open("w") as f:
        for r in picked:
            f.write(json.dumps({
                "id": r["id"],
                "problem": r["problem"],
                "answer": r["answer"],
                "level": r["level"],
                "subject": r["subject"],
                "pass_at_1": r["c"] / r["n"],
                "n_correct": r["c"],
                "n_samples": r["n"],
                "split_origin": r.get("split_origin"),
                "band": label,
                "profiled_model": spec["model_id"],
            }, ensure_ascii=False) + "\n")

    ps = [r["c"] / r["n"] for r in picked]
    lv = dict(sorted(collections.Counter(str(r["level"]) for r in picked).items()))
    so = dict(sorted(collections.Counter(str(r.get("split_origin")) for r in picked).items()))
    manifest["subsets"].append({
        "file": fname, "band": label, "pool_size": len(pool), "sampled": k,
        "mean_pass_at_1": sum(ps) / len(ps), "level_dist": lv, "split_origin_dist": so,
    })
    print(f"  {fname:48} {k:4}/{len(pool):5}  mean p={sum(ps)/len(ps):.4f}  levels={lv}")

(outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"\nwrote {len(spec['bands'])} subsets + manifest.json to {outdir}")
