#!/usr/bin/env python3
"""One screen of what the goal runs have done. Read-only, no GPU, no key.

    python tools/goal_status.py [--run DIR]

Reads the run directories the same way a person would and prints what actually
decides whether a run is worth anything: how far it got, what the teacher chose
(training reward is a direct read of the curriculum's difficulty against the
student that received it), whether the targets moved, and what it cost.
"""
import argparse
import json
import pathlib
import sys

OUT = pathlib.Path.home() / "data/frontier-teacher/outputs/goal-teacher"


def _jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in p.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def summarise(run: pathlib.Path):
    ev = _jsonl(run / "events.jsonl")
    me = _jsonl(run / "metrics.jsonl")
    state = {}
    if (run / "run_state.json").exists():
        state = json.loads((run / "run_state.json").read_text())

    done = any(e.get("event") == "run_done" for e in ev)
    halts = [e for e in ev if e.get("event") == "halt"]
    tfail = [e for e in ev if e.get("event") == "target_eval_failed"]
    last = state.get("last_completed_step", -1)

    print(f"\n{'=' * 78}\n{run.name}   {'COMPLETE' if done else 'in progress'}"
          f"   steps closed: {last + 1}/20")
    if halts:
        print(f"  halts: {len(halts)}")
        for h in halts[-3:]:
            print(f"    step {h.get('step')}: {h.get('error')}")
    if tfail:
        print(f"  target-eval failures: {len(tfail)}"
              f"  (latest: {tfail[-1].get('error')})")

    cost = sum(m.get("teacher_cost_usd") or 0 for m in me)
    tok = sum(m.get("teacher_output_tokens") or 0 for m in me)
    tw = sum(m.get("teacher_wallclock_s") or 0 for m in me) / 3600
    aw = sum(m.get("action_wallclock_s") or 0 for m in me) / 3600
    nev = sum(1 for m in me if m.get("type") == "eval" and m.get("status") == "ok")
    print(f"  teacher: ${cost:.2f}  {tok:,} out-tok  {tw:.1f}h"
          f"   |  actions: {aw:.1f}h   evals: {nev}"
          f" ({nev / max(last + 1, 1):.1f}/step)")

    # What the teacher chose, per closed step: the fraction of the 512 rollouts
    # that were correct. Near 0 or near 1 is a step that taught little.
    rew, adv = [], []
    for k in range(last + 1):
        rp = run / f"step_{k}" / "result.json"
        if not rp.exists():
            continue
        m = (json.loads(rp.read_text()).get("train") or {}).get("metrics") or {}
        if "critic/score/mean" in m:
            rew.append(m["critic/score/mean"])
            adv.append(m.get("critic/advantages/max", 0))
    if rew:
        print(f"  training reward: " + " ".join(f"{r * 100:.0f}" for r in rew))
        print(f"    first {rew[0] * 100:.0f}%  last {rew[-1] * 100:.0f}%  "
              f"mean {sum(rew) / len(rew) * 100:.0f}%   "
              f"max-advantage 5.48 on {sum(1 for a in adv if a > 5.47)}/{len(adv)} steps")

    # The targets. Expected to be a flat zero -- both curriculum-rl arms were,
    # over 600 attempts, with larger students.
    rows = []
    base = _jsonl(run / "targets" / "per_target.jsonl")
    if base:
        rows.append(("base", base))
    for k in range(last + 1):
        p = _jsonl(run / f"step_{k}" / "targets" / "per_target.jsonl")
        if p:
            rows.append((str(k), p))
    if rows:
        ids = [r["id"].rsplit("-", 1)[-1] for r in rows[0][1]]
        print(f"  targets ({' '.join(ids)}):")
        moved = False
        for label, r in rows:
            vals = [x.get("pass_at_1", 0) for x in r]
            if any(v > 0 for v in vals):
                moved = True
            print(f"    {label:>4}: " + "  ".join(f"{v:.3f}" for v in vals))
        if not moved:
            print("    (flat zero throughout -- as both curriculum-rl arms were)")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    runs = ([pathlib.Path(a.run)] if a.run
            else sorted(OUT.glob("run_*"), key=lambda p: p.name))
    if not runs:
        sys.exit(f"no runs under {OUT}")
    for r in runs:
        summarise(r)
    print()


if __name__ == "__main__":
    main()
