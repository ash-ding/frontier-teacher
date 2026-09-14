"""Paired endpoint statistics for every cell, at both pass@k and pass@1.

Two levels of resampling, because both matter and only one is obvious. Problems
are resampled because a benchmark is a sample of problems. Generations are
resampled because a benchmark score is a sample of generations: re-running one
identical AIME evaluation of one identical model returned pass@4 15.3% once and
12.0% another time, and treating each problem's measured c-of-n as exact
understates the interval by about a third.

Writes outputs/analysis/paired_stats.json, which tools/render_report.py reads so
the figure never shows a delta without its interval.

    python tools/paired_stats.py                  # every GRPO band, endpoint - base
    python tools/paired_stats.py --teacher        # the teacher arm, held-out ids

--teacher reads outputs/frontier-model/run_*/final_eval/ as well, and answers the
question §8 asks: not only teacher - base, but teacher - band for every band of
the same student, on the same benchmark, metric and problems, at the 10,240
rollout endpoint (teacher step 19 == band step 20). Problems are resampled
jointly across base, teacher and every band, so each difference is paired;
generations are resampled independently per arm. Because the teacher could read
the reference fifth of each benchmark, --teacher defaults to --holdout: every arm
is scored on the complement of data/benchmark/*_ref*.manifest.json. It writes
outputs/analysis/paired_stats_teacher.json and leaves paired_stats.json alone.

--exclude takes outputs/analysis/benchmark_contamination.json (tools/teacher_overlap.py
--contamination): benchmark problems found inside something an arm trained on.
For each model, that model's exclusion ids are removed from every arm, base
included, on top of --holdout, so all arms of a model are still scored on one set
of problems. --exclude-level picks `ids` (default: exact + paraphrase) or
`strict_ids` (adds shared-setup items); give both, comma-separated, and the
output is keyed by level, each level resampled from a fresh seed.

    python tools/paired_stats.py --teacher \
        --exclude outputs/analysis/benchmark_contamination.json \
        --exclude-level default,strict \
        --out outputs/analysis/paired_stats_teacher_clean.json
"""
import argparse, glob, json, os, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = {"math500": "pass@1", "aime": "pass@4", "hmmt": "pass@4"}
KS = {"pass@1": 1, "pass@4": 4}
SEED = 20260831
# lazy up to __step: band slugs contain underscores (pass1_eq_0, pass1_eq_1)
CKPT = re.compile(r"^(?P<cfg>.+?)__(?P<band>pass1_.+?)(?:__(?P<variant>g\d+))?"
                  r"__step(?P<step>\d+)__(?P<task>\w+)$")
TEACHER = re.compile(r"^(?P<cfg>.+?)__teacher__step(?P<step>\d+)__(?P<task>\w+)$")
# Same rollouts, two numberings - see tools/compare_teacher.py.
BAND_END_STEP, TEACHER_END_STEP, END_ROLLOUTS = 20, 19, 10240
# The fifth of each benchmark the teacher could read - see tools/holdout_split.py.
REF_MANIFEST = {"math500": "math500_ref100", "aime": "aime_ref30",
                "hmmt": "hmmt_ref19"}


def pak(n, c, k):
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def draw(rnd, n, c):
    """One resample of a problem's n generations at its measured rate c/n."""
    return sum(1 for _ in range(n) if rnd.random() < c / n)


def load(path):
    if not os.path.exists(path):
        return None
    d = {}
    for line in open(path):
        r = json.loads(line)
        d[r["id"]] = (r["n"], r["c"])
    return d or None


def seen_ids(task):
    p = ROOT / "data" / "benchmark" / f"{REF_MANIFEST[task]}.manifest.json"
    return set(json.loads(p.read_text())["ids"])


EXCLUDE_KEY = {"default": "ids", "strict": "strict_ids"}


def excluded_ids(a, cfg, task):
    """The contamination ids to drop for this model and benchmark, or an empty set."""
    if not a.exclude_map:
        return set()
    try:
        e = a.exclude_map[cfg][task]
    except KeyError:
        raise SystemExit(f"--exclude has no entry for {cfg}/{task}; refusing to "
                         f"read that as 'nothing to exclude'")
    return set(e[EXCLUDE_KEY[a.level]])


def interval(boots):
    boots.sort()
    B = len(boots)
    return {"lo": boots[int(.025 * B)], "hi": boots[int(.975 * B)],
            "p_le_zero": sum(1 for x in boots if x <= 0) / B}


def grpo_cells(a, rnd, out_dir):
    runs = {}
    for f in out_dir.glob("grpo/*/records.jsonl"):
        m = CKPT.match(f.parent.name)
        if not m or m["task"] not in HEADLINE:
            continue
        key = (m["cfg"], m["band"], m["variant"] or "", m["task"])
        runs.setdefault(key, {})[int(m["step"])] = str(f)

    results = []
    for (cfg, band, variant, task), steps in sorted(runs.items()):
        base = load(str(out_dir / "benchmarks" / f"{cfg}__{task}" / "records.jsonl"))
        last = max(steps)
        end = load(steps[last])
        if not base or not end:
            continue
        ids = sorted(set(base) & set(end))
        if a.holdout:
            hide = seen_ids(task)
            ids = [i for i in ids if i not in hide]
        if a.exclude_map:
            drop = excluded_ids(a, cfg, task)
            ids = [i for i in ids if i not in drop]
        if not ids:
            print(f"  SKIP {cfg}/{band}/{task}: baseline and checkpoint share no ids "
                  f"- run eval/check_baselines.py")
            continue
        for metric, k in KS.items():
            obs = sum(pak(*end[i], k) - pak(*base[i], k) for i in ids) / len(ids)
            boots = []
            for _ in range(a.boots):
                tot = 0.0
                for _ in range(len(ids)):
                    i = ids[rnd.randrange(len(ids))]
                    na, ca = base[i]; nb, cb = end[i]
                    da = draw(rnd, na, ca)
                    db = draw(rnd, nb, cb)
                    tot += pak(nb, db, k) - pak(na, da, k)
                boots.append(tot / len(ids))
            boots.sort()
            row = {
                "config": cfg, "band": band, "variant": variant, "task": task,
                "metric": metric, "final_step": last, "n_problems": len(ids),
                "delta": obs,
                "lo": boots[int(.025 * a.boots)], "hi": boots[int(.975 * a.boots)],
                "p_le_zero": sum(1 for x in boots if x <= 0) / a.boots,
                "headline": metric == HEADLINE[task],
            }
            if a.holdout:
                row["holdout"] = True
            if a.exclude_map:
                row["exclusion_level"] = a.level
            results.append(row)
    return results


def teacher_arms(out_dir, models):
    """{(cfg, task): {"run": name, "arms": {label: records_path}}} at the endpoint.

    Labels are "teacher" and the band slug with its variant, e.g.
    "pass1_40-60pct__g32", as tools/holdout_split.py names them.
    """
    cells = {}
    for fe in sorted(out_dir.glob("frontier-model/run_*/final_eval")):
        for d in sorted(fe.iterdir()):
            m = TEACHER.match(d.name)
            if not m or m["task"] not in HEADLINE or int(m["step"]) != TEACHER_END_STEP:
                continue
            if models and m["cfg"] not in models:
                continue
            if not (d / "records.jsonl").exists():
                continue
            cell = cells.setdefault((m["cfg"], m["task"]), {"arms": {}})
            if "teacher" in cell["arms"]:
                print(f"  NOTE {m['cfg']}/{m['task']}: more than one teacher run; "
                      f"using the newest, {fe.parent.name}")
            cell["run"] = fe.parent.name
            cell["arms"]["teacher"] = str(d / "records.jsonl")
    for f in sorted(out_dir.glob("grpo/*/records.jsonl")):
        m = CKPT.match(f.parent.name)
        if not m or int(m["step"]) != BAND_END_STEP or (m["cfg"], m["task"]) not in cells:
            continue
        label = m["band"] + (f"__{m['variant']}" if m["variant"] else "")
        cells[(m["cfg"], m["task"])]["arms"][label] = str(f)
    return cells


def teacher_cells(a, rnd, out_dir):
    results = []
    for (cfg, task), cell in sorted(teacher_arms(out_dir, a.model).items()):
        base = load(str(out_dir / "benchmarks" / f"{cfg}__{task}" / "records.jsonl"))
        if not base:
            print(f"  SKIP {cfg}/{task}: no baseline records")
            continue
        labels = ["teacher"] + sorted(l for l in cell["arms"] if l != "teacher")
        arms = {l: load(cell["arms"][l]) for l in labels}
        arms = {l: r for l, r in arms.items() if r}
        labels = [l for l in labels if l in arms]
        common = set(base)
        for l in labels:
            if set(arms[l]) != set(base):
                print(f"  WARN {cfg}/{task}/{l}: ids differ from baseline "
                      f"({len(set(arms[l]) ^ set(base))} not shared)")
            common &= set(arms[l])
        ids = sorted(common)
        if a.holdout:
            hide = seen_ids(task)
            ids = [i for i in ids if i not in hide]
        n_before = len(ids)
        if a.exclude_map:
            drop = excluded_ids(a, cfg, task)
            ids = [i for i in ids if i not in drop]
        if not ids:
            print(f"  SKIP {cfg}/{task}: arms share no ids - run eval/check_baselines.py")
            continue
        for l in labels:
            bad = sum(1 for i in ids if arms[l][i][0] != base[i][0])
            if bad:
                print(f"  NOTE {cfg}/{task}/{l}: n differs from baseline on {bad} problems")
        order = ["base"] + labels
        recs = {"base": base, **arms}
        common_fields = {"config": cfg, "task": task, "teacher_run": cell["run"],
                         "rollouts": END_ROLLOUTS, "teacher_step": TEACHER_END_STEP,
                         "band_step": BAND_END_STEP, "n_problems": len(ids),
                         "holdout": bool(a.holdout)}
        if a.exclude_map:
            common_fields.update(exclusion_level=a.level,
                                 n_excluded=n_before - len(ids))
        for metric, k in KS.items():
            # per-arm mean over the problems, observed and per replicate
            obs = {l: sum(pak(*recs[l][i], k) for i in ids) / len(ids) for l in order}
            reps = {l: [] for l in order}
            for _ in range(a.boots):
                tot = dict.fromkeys(order, 0.0)
                for _ in range(len(ids)):
                    i = ids[rnd.randrange(len(ids))]
                    for l in order:
                        n, c = recs[l][i]
                        tot[l] += pak(n, draw(rnd, n, c), k)
                for l in order:
                    reps[l].append(tot[l] / len(ids))
            hl = metric == HEADLINE[task]
            for l in labels:
                d = [x - y for x, y in zip(reps[l], reps["base"])]
                results.append({**common_fields, "kind": "vs_base", "arm": l,
                                "metric": metric, "score": obs[l],
                                "base_score": obs["base"],
                                "delta": obs[l] - obs["base"],
                                **interval(d), "headline": hl})
            for l in labels:
                if l == "teacher":
                    continue
                d = [x - y for x, y in zip(reps["teacher"], reps[l])]
                results.append({**common_fields, "kind": "teacher_minus_band",
                                "arm": l, "metric": metric,
                                "teacher_score": obs["teacher"], "band_score": obs[l],
                                "delta": obs["teacher"] - obs[l],
                                **interval(d), "headline": hl})
        ex = f" ({n_before - len(ids)} excluded, {a.level})" if a.exclude_map else ""
        print(f"  {cfg}/{task}: {len(ids)} problems{ex}, teacher + {len(labels) - 1} bands")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--out", default=None,
                    help="default outputs/analysis/paired_stats.json, "
                         "or paired_stats_teacher.json with --teacher")
    ap.add_argument("--boots", type=int, default=6000)
    ap.add_argument("--teacher", action="store_true",
                    help="teacher arm: teacher - base and teacher - band at 10,240 rollouts")
    ap.add_argument("--holdout", action=argparse.BooleanOptionalAction, default=None,
                    help="score only the ids outside the teacher's reference sets "
                         "(default: on with --teacher, off otherwise)")
    ap.add_argument("--model", action="append", default=None,
                    help="with --teacher, restrict to this config; repeatable")
    ap.add_argument("--exclude", default=None,
                    help="benchmark_contamination.json: drop each model's exclusion "
                         "ids from every arm of that model, base included")
    ap.add_argument("--exclude-level", default="default",
                    help="default (ids), strict (strict_ids), or both as "
                         "'default,strict'")
    a = ap.parse_args()
    levels = [l.strip() for l in a.exclude_level.split(",") if l.strip()]
    if not levels or any(l not in EXCLUDE_KEY for l in levels):
        ap.error(f"--exclude-level: choose from {sorted(EXCLUDE_KEY)}")
    a.exclude_map = json.loads(Path(a.exclude).read_text())["exclusion"] if a.exclude else None
    if a.holdout is None:
        a.holdout = a.teacher
    if a.out is None:
        a.out = str(ROOT / "outputs" / "analysis" /
                    ("paired_stats_teacher.json" if a.teacher else "paired_stats.json"))
    out_dir = Path(a.outputs)
    cells = teacher_cells if a.teacher else grpo_cells

    if not a.exclude_map:
        a.level = None
        results = cells(a, random.Random(SEED), out_dir)
        flat = results
    else:
        results = {"exclusion_file": a.exclude, "holdout": bool(a.holdout),
                   "boots": a.boots, "seed": SEED, "levels": {}}
        flat = []
        for level in levels:
            a.level = level
            print(f"exclusion level {level} ({EXCLUDE_KEY[level]})")
            rows = cells(a, random.Random(SEED), out_dir)
            results["levels"][level] = rows
            flat += rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    sig = sum(1 for r in flat if r["lo"] > 0 or r["hi"] < 0)
    print(f"wrote {a.out}  {len(flat)} cell-metrics, {sig} with an interval clear of zero")


if __name__ == "__main__":
    main()
