"""Does the sanitiser actually keep the answers out? Fabricated input, no GPU.

The three leak paths this checks are the ones evaluate.py really produces:
records.jsonl carries the gold answer because each record is `{**row, ...}`,
summary.json carries the plaintext path in data_files, and generations.jsonl
carries a per-sample `correct` verdict. Nothing here is mocked except the
evaluation itself.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from goal_teacher import target_eval  # noqa: E402

SECRET_A = "t^{-\\frac{1}{t}}"
SECRET_B = "\\left\\lfloor \\frac{n}{2}\\right\\rfloor + n - 1"
SECRET_C = "\\frac{1}{8}"
ANSWERS = {"q1": SECRET_A, "q2": SECRET_B, "q3": SECRET_C}

fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


raw = Path(tempfile.mkdtemp(prefix="sanitise_raw_"))
out = Path(tempfile.mkdtemp(prefix="sanitise_out_")) / "targets"

(raw / "summary.json").write_text(json.dumps({
    "n_problems": 3, "samples_per_problem": 8,
    "pass@1": 0.0, "pass@4": 0.0, "pass@8": 0.0,
    "truncation_rate": 0.04, "no_answer_rate": 0.12, "mean_gen_tokens": 812,
    "headline_metric": "pass@1", "label": "reference_test_targets",
    "model": "/home/x/.local_checkpoints/run/step_4/…/huggingface",
    "verifier": "symbolic", "gen_seconds": 41.2,
    "data_files": ["/tmp/goal_targets_step4_abc/targets.jsonl"],   # leak 2
    "sampling": {"temperature": 0.6, "seed": 1234},
}))

with (raw / "records.jsonl").open("w") as f:
    for pid, ans in ANSWERS.items():
        f.write(json.dumps({                                        # leak 1
            "id": pid, "problem": f"statement of {pid}", "answer": ans,
            "n": 8, "c": 0,
            "samples": [{"correct": False, "extracted": "1/2",
                         "truncated": False, "n_tokens": 700}] * 8,
        }) + "\n")

with (raw / "generations.jsonl").open("w") as f:
    for pid in ANSWERS:
        for i in range(2):
            f.write(json.dumps({                                    # leak 3
                "id": pid, "sample": i, "correct": False,
                "extracted": "1/2", "truncated": False, "n_tokens": 700,
                "text": f"student reasoning for {pid} … \\boxed{{1/2}}",
            }) + "\n")

per = target_eval._sanitise(raw, out)

print("sanitised files:", sorted(p.name for p in out.iterdir()))
print()

summary = (out / "summary.json").read_text()
pertxt = (out / "per_target.jsonl").read_text()
traces = json.loads((out / "target_traces.json").read_text())
both = summary + pertxt

check("no answer string in summary.json or per_target.jsonl",
      not any(a in both for a in ANSWERS.values()))
check("data_files (the plaintext path) is gone",
      "data_files" not in summary and "goal_targets_step4" not in summary)
check("model path is gone", "huggingface" not in summary)
check("pass@k kept", '"pass@1"' in summary and '"pass@8"' in summary)
check("per-target rates written", len(per) == 3 and all("pass_at_1" in r for r in per))
check("no per-sample correct verdict survives",
      "correct" not in pertxt and "correct" not in json.dumps(traces))
check("traces keep the student text", all(len(v) == 2 for v in traces.values()))

print()
print("assert_clean on a good sanitise:")
try:
    target_eval.assert_clean(out, ANSWERS)
    check("passes", True)
except RuntimeError as e:
    check(f"passes (got {e})", False)

print()
print("assert_clean on a BROKEN sanitise (answer written into per_target.jsonl):")
(out / "per_target.jsonl").write_text(
    json.dumps({"id": "q3", "answer": SECRET_C}) + "\n")
try:
    target_eval.assert_clean(out, ANSWERS)
    check("catches the leak", False)
except RuntimeError:
    check("catches the leak", True)

print()
print("assert_clean when the STUDENT wrote a correct answer into its trace:")
(out / "per_target.jsonl").write_text(json.dumps(per[0]) + "\n")
(out / "target_traces.json").write_text(json.dumps(
    {"q3": [f"… therefore the limit is \\boxed{{{SECRET_C}}}"]}))
try:
    target_eval.assert_clean(out, ANSWERS)
    check("does NOT abort the run (this is the result we want)", True)
except RuntimeError:
    check("does NOT abort the run (this is the result we want)", False)

print()
print("ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
