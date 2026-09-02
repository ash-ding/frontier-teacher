"""Recompute every evaluation on the problems the teacher could not see.

The per-step reference tests are one fifth of each benchmark, and their results
-- the problems, the student's full generations, the per-sample verdicts -- land
in the run directory, which the teacher reads. It did read them: the transcripts
show repeated opens of step_*/test/ across the run.

The final evaluation then scored the whole benchmark, that fifth included. So a
teacher curriculum drawn from what it saw there would raise the reported number
through a channel no baseline band had: the bands are slices of the MATH
training pool and never touch a benchmark problem.

This splits every evaluation by the manifest's id list and recomputes the
headline metric on each side, for teacher runs and baseline bands alike -- both
have to be read on the same problems for the comparison to mean anything. The
held-out four fifths is the number to report. The seen fifth is not noise to be
discarded: the gap between them is the measurement of whether it happened.

    python tools/holdout_split.py                    # every model
    python tools/holdout_split.py --model llama32-3b
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from metrics import pass_at_k                                    # noqa: E402

OUT = ROOT / "outputs"
HEADLINE = {"math500": "pass@1", "aime": "pass@4", "hmmt": "pass@4"}
REF_MANIFEST = {"math500": "math500_ref100", "aime": "aime_ref30",
                "hmmt": "hmmt_ref19"}
BASE_STEPS = {5: 2560, 10: 5120, 15: 7680, 20: 10240}
TEACHER_STEPS = {4: 2560, 9: 5120, 14: 7680, 19: 10240}
MODELS = ["llama32-3b", "qwen3-4b-nothink", "qwen3-4b-think"]


def seen_ids(task):
    p = ROOT / "data" / "benchmark" / f"{REF_MANIFEST[task]}.manifest.json"
    return set(json.loads(p.read_text())["ids"])


def score(records, task):
    """Mean pass@k over these records, with the standard error of that mean."""
    if not records:
        return None
    k = int(HEADLINE[task].split("@")[1])
    vals = [pass_at_k(r["n"], r["c"], k) for r in records]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    return {"n": n, "score": mean, "stderr": (var / n) ** 0.5}


def split_dir(d, task):
    """(held_out, seen, full) for one evaluation directory."""
    f = d / "records.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(l) for l in f.open() if l.strip()]
    if not recs or "c" not in recs[0]:
        return None
    ids = seen_ids(task)
    held = [r for r in recs if r["id"] not in ids]
    saw = [r for r in recs if r["id"] in ids]
    return score(held, task), score(saw, task), score(recs, task)


def rows_for(model):
    """[(curriculum, task, rollouts, held, seen, full)] for one model."""
    out = []
    for task in HEADLINE:
        b = OUT / "benchmarks" / f"{model}__{task}"
        s = split_dir(b, task)
        if s:
            out.append(("(base model)", task, 0, *s))
        for d in sorted((OUT / "grpo").glob(f"{model}__pass1_*__step*__{task}")):
            name = d.name[len(model) + 2:]
            band, _, rest = name.partition("__step")
            step = int(rest.split("__")[0])
            if step not in BASE_STEPS:
                continue
            s = split_dir(d, task)
            if s:
                out.append((band, task, BASE_STEPS[step], *s))
        for fe in sorted((OUT / "frontier-model").glob("run_*/final_eval")):
            for step, ro in TEACHER_STEPS.items():
                d = fe / f"{model}__teacher__step{step}__{task}"
                s = split_dir(d, task)
                if s:
                    out.append(("TEACHER", task, ro, *s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--json", default=None, help="also write the split to this path")
    a = ap.parse_args()

    dump = {}
    for model in a.model or MODELS:
        rows = rows_for(model)
        if not rows:
            continue
        dump[model] = [{"curriculum": c, "task": t, "rollouts": r,
                        "held_out": h, "seen": s, "full": f}
                       for c, t, r, h, s, f in rows]
        print(f"\n=== {model} ===")
        for task in HEADLINE:
            end = [x for x in rows if x[1] == task and x[2] in (0, 10240)]
            if not end:
                continue
            base = next((x for x in end if x[2] == 0), None)
            print(f"\n  {task}  ({HEADLINE[task]}, %) at 10,240 rollouts")
            print(f"    {'curriculum':<16}{'held-out':>12}{'seen 1/5':>12}"
                  f"{'full':>10}{'gap':>9}")
            for label, src in ([("(base model)", base)] if base else []) + \
                    [(x[0], x) for x in end if x[2] == 10240]:
                h, s, f = src[3], src[4], src[5]
                if not h or not s:
                    continue
                gap = 100 * (s["score"] - h["score"])
                print(f"    {label:<16}{100*h['score']:>9.1f}±{100*h['stderr']:<2.1f}"
                      f"{100*s['score']:>9.1f}±{100*s['stderr']:<2.1f}"
                      f"{100*f['score']:>10.1f}{gap:>+9.1f}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dump, indent=2))
        print(f"\nwrote {a.json}")
    print("\n  gap = seen minus held-out. The bands never saw the reference fifth,\n"
          "  so their gap is sampling noise and sets the scale for reading the\n"
          "  teacher's.")


if __name__ == "__main__":
    main()
