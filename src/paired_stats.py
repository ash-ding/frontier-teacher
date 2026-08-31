"""Paired endpoint statistics for every cell, at both pass@k and pass@1.

Two levels of resampling, because both matter and only one is obvious. Problems
are resampled because a benchmark is a sample of problems. Generations are
resampled because a benchmark score is a sample of generations: re-running one
identical AIME evaluation of one identical model returned pass@4 15.3% once and
12.0% another time, and treating each problem's measured c-of-n as exact
understates the interval by about a third.

Writes outputs/paired_stats.json, which src/render_report.py reads so the figure
never shows a delta without its interval.
"""
import argparse, glob, json, os, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = {"math500": "pass@1", "aime": "pass@4", "hmmt": "pass@4"}
KS = {"pass@1": 1, "pass@4": 4}
CKPT = re.compile(r"^(?P<cfg>.+?)__(?P<band>pass1_[^_]+)(?:__(?P<variant>g\d+))?"
                  r"__step(?P<step>\d+)__(?P<task>\w+)\.records\.jsonl$")


def pak(n, c, k):
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def load(path):
    if not os.path.exists(path):
        return None
    d = {}
    for line in open(path):
        r = json.loads(line)
        d[r["id"]] = (r["n"], r["c"])
    return d or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "paired_stats.json"))
    ap.add_argument("--boots", type=int, default=6000)
    a = ap.parse_args()
    rnd = random.Random(20260831)
    out_dir = Path(a.outputs)

    runs = {}
    for p in out_dir.glob("*__step*__*.records.jsonl"):
        m = CKPT.match(p.name)
        if not m or m["task"] not in HEADLINE:
            continue
        key = (m["cfg"], m["band"], m["variant"] or "", m["task"])
        runs.setdefault(key, {})[int(m["step"])] = str(p)

    results = []
    for (cfg, band, variant, task), steps in sorted(runs.items()):
        base = load(str(out_dir / f"{cfg}__{task}.records.jsonl"))
        last = max(steps)
        end = load(steps[last])
        if not base or not end:
            continue
        ids = sorted(set(base) & set(end))
        if not ids:
            print(f"  SKIP {cfg}/{band}/{task}: baseline and checkpoint share no ids "
                  f"- run src/check_baselines.py")
            continue
        for metric, k in KS.items():
            obs = sum(pak(*end[i], k) - pak(*base[i], k) for i in ids) / len(ids)
            boots = []
            for _ in range(a.boots):
                tot = 0.0
                for _ in range(len(ids)):
                    i = ids[rnd.randrange(len(ids))]
                    na, ca = base[i]; nb, cb = end[i]
                    da = sum(1 for _ in range(na) if rnd.random() < ca / na)
                    db = sum(1 for _ in range(nb) if rnd.random() < cb / nb)
                    tot += pak(nb, db, k) - pak(na, da, k)
                boots.append(tot / len(ids))
            boots.sort()
            results.append({
                "config": cfg, "band": band, "variant": variant, "task": task,
                "metric": metric, "final_step": last, "n_problems": len(ids),
                "delta": obs,
                "lo": boots[int(.025 * a.boots)], "hi": boots[int(.975 * a.boots)],
                "p_le_zero": sum(1 for x in boots if x <= 0) / a.boots,
                "headline": metric == HEADLINE[task],
            })
    Path(a.out).write_text(json.dumps(results, indent=1))
    sig = sum(1 for r in results if r["lo"] > 0 or r["hi"] < 0)
    print(f"wrote {a.out}  {len(results)} cell-metrics, {sig} with an interval clear of zero")


if __name__ == "__main__":
    main()
