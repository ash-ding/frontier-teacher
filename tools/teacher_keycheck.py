"""How often was the teacher's answer key wrong?

In the open-ended teacher runs (docs/experiment.md §8) the `answer` a teacher
wrote into `step_*/train/data.jsonl` became `reward_model.ground_truth` with no
verification, so a wrong key rewarded the student for a wrong answer. This
measures the rate (docs/plan.md §2 item 2).

Each training problem is solved independently by a different model than the
teacher, through `claude -p` with no tools at all, from an empty temporary
directory, never shown the teacher's key. The solver's boxed answer is graded
against the key with `eval/verifiers` -- the same `grade` the GRPO reward used --
so "disagree" means exactly what "the student would have been scored wrong for
giving this answer" meant in training.

Every disagreement gets a second call that sees the problem and both answers
(blind labels, order shuffled by a seeded rng) and rules on it:

    key_wrong      the solver is right and the teacher's key is not
    both_wrong     neither answer is right -- the key is still wrong
    solver_wrong   the key is right
    equivalent     both are the same answer; the grader missed it
    ill_posed      the statement is ambiguous / underdetermined, so no key is a
                   clean reward signal -- its own category
    unparsed       the adjudication gave no readable verdict

One JSON line per problem is appended to
`outputs/analysis/teacher_keycheck/<run>.jsonl`, keyed by (step, id) because the
teacher reused ids across steps; rows already present are skipped, so an
interrupted run resumes. A failed call writes nothing and is retried next time.
Rows carry `solver_model`, and both resuming and `--summary` count only rows
from the `--model` in force, so results from two solver models can share a file
without one standing in for the other.

Two repairs to the solve pass, both without repeating a solve. `--regrade`:
with thinking on, the CLI returns only the final text block, and Sonnet 5 often
ends a hard problem with a bare `33` and no box; `grade_solver` grades a
one-line final text as if boxed, and rows written before it existed are
re-extracted (agreeing rows drop their adjudication, still-disagreeing rows are
re-adjudicated with the real answer). `--mark-unsolved`: a problem whose solve
timed out on every attempt gets an `unsolved` row and goes to the recheck.

Python recheck (`--recheck`, after the solve pass). Rows it takes: every
disagreement in the thinking run other than `equivalent` -- the hardest problems,
where a model's unaided adjudication is least trustworthy -- and, in the other
two runs, every `key_wrong`, `both_wrong`, `ill_posed`, `unparsed` or `unsolved`. A call
writes a self-contained script (stdlib, sympy, numpy) that computes or
brute-forces the answer, seeing the problem and both answers under fresh blind
labels. This orchestrator, not claude, runs it: fresh temp dir, prlimit (8 GB
address space, 120 s CPU, 50 MB files), 120 s wall clock, and bubblewrap with
every namespace unshared -- no network, and a filesystem of /usr and the conda
env read-only, so it cannot see the repo or the run directory. (`unshare -rn`
also works on these nodes and is the fallback, network isolation only; with
neither, scripts are refused.) A failing script gets one rewrite with its error.
A last call sees the script and its output and rules, with `inconclusive`
available so that an uncheckable problem is recorded as such rather than
forced; a problem the script-writer declares uncheckable is recorded as
`not_checkable` with no run. The recheck's category, where it reached one,
supersedes the adjudication's in `--summary`; `category` on the row stays the
adjudication's, and `recheck.overturned` marks where the two differ.

`--budget-usd` stops new calls once every cost recorded on disk, plus the
ledger of discarded calls, reaches the limit; eight consecutive failed calls
stop the job the same way.

    python tools/teacher_keycheck.py --sample 20 --seed 0     # the cost probe
    python tools/teacher_keycheck.py --all --budget-usd 90    # all 960
    python tools/teacher_keycheck.py --recheck --budget-usd 90
    python tools/teacher_keycheck.py --summary [--selection sample20_seed0]

All take `--model` (default claude-opus-5, which this Vertex project cannot
reach; the full check ran on claude-sonnet-5).

Grading happens in the main thread: math_verify's parse timeout is signal-based,
and a worker-thread grade could fail over to the string-equality fallback
without saying so.
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from verifiers.extract import answer_segment, grade  # noqa: E402

RUNS = {
    "run_20260902_072619": "llama32-3b",
    "run_20260902_072856": "qwen3-4b-nothink",
    "run_20260902_072857": "qwen3-4b-think",
}
TEACHER_MODEL = "claude-opus-4-8"
RUN_DIR = ROOT / "outputs" / "frontier-model"
OUT_DIR = ROOT / "outputs" / "analysis" / "teacher_keycheck"
N_STEPS = 20

SOLVE_SYSTEM = ("You are a careful competition mathematician. Solve the problem "
                "you are given on your own. You have no tools.")
# evaluate.py's PROMPT plus one sentence: the probe's only disagreement was the
# solver boxing `\\tan 75° = 2 + \\sqrt{3}`, which the grader cannot match to
# `2+\\sqrt{3}`. Version 1 (the probe) lacked the sentence; rows record which.
SOLVE_PROMPT_VERSION = 2
SOLVE_PROMPT = ("Solve the following math problem. Reason step by step, and put "
                "your final answer within \\boxed{{}}. The box should contain only "
                "the final answer, not an equation or a label.\n\n{problem}")

ADJ_SYSTEM = ("You are a careful competition mathematician refereeing an answer "
              "key. You have no tools.")
ADJ_PROMPT = """A math problem and two proposed final answers, labelled X and Y, are below. An automatic checker judged that the two answers differ.

Work the problem yourself, carefully, and then rule on it. The possible verdicts:
- "X": answer X is correct and answer Y is not
- "Y": answer Y is correct and answer X is not
- "both": X and Y are the same answer written differently, and it is correct
- "neither": both answers are wrong
- "ill_posed": the problem as stated is ambiguous, underdetermined, contradictory or otherwise admits more than one defensible answer, so it has no single correct key

## Problem
{problem}

## Answer X
{x}

## Answer Y
{y}

End your response with exactly one line of JSON and nothing after it:
{{"verdict": "X" | "Y" | "both" | "neither" | "ill_posed", "correct_answer": "<the answer you believe is correct, or empty if ill_posed>", "reason": "<one sentence>"}}"""


def load_rows():
    """Every training-curriculum row: {run, student, step, id, problem, answer}."""
    rows = []
    for run, student in RUNS.items():
        for step in range(N_STEPS):
            path = RUN_DIR / run / f"step_{step}" / "train" / "data.jsonl"
            for line in path.open():
                if line.strip():
                    r = json.loads(line)
                    rows.append({"run": run, "student": student, "step": step,
                                 "id": r["id"], "problem": r["problem"],
                                 "answer": str(r["answer"])})
    return rows


def stratified_sample(rows, n, seed):
    """n rows spread over (run, early steps 0-9 / late steps 10-19) strata."""
    rng = random.Random(seed)
    strata = {}
    for r in rows:
        strata.setdefault((r["run"], "early" if r["step"] < N_STEPS // 2 else "late"),
                          []).append(r)
    keys = sorted(strata)
    alloc = {k: n // len(keys) for k in keys}
    for k in rng.sample(keys, n % len(keys)):
        alloc[k] += 1
    picked = []
    for k in keys:
        picked += rng.sample(strata[k], alloc[k])
    return picked


def claude_call(prompt, system, model, timeout_s, max_budget, effort):
    """One headless, tool-less `claude -p` from an empty temp cwd. -> dict."""
    argv = ["claude", "-p", prompt,
            "--model", model,
            "--output-format", "json",
            "--tools", "",                 # no built-in tools at all
            "--strict-mcp-config",         # and no MCP servers
            "--setting-sources", "",       # no user/project settings, hooks, CLAUDE.md
            "--no-session-persistence",
            "--system-prompt", system,
            "--max-budget-usd", str(max_budget)]
    if effort:
        argv += ["--effort", effort]
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="keycheck_") as cwd:
        try:
            p = subprocess.run(argv, cwd=cwd, env=os.environ.copy(),
                               stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout_s}s",
                    "wall_s": time.time() - t0}
    wall = time.time() - t0
    try:
        o = json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": f"rc={p.returncode}, stdout not json: "
                f"{p.stdout[:200]!r} stderr: {p.stderr[-300:]!r}", "wall_s": wall}
    usage = o.get("usage") or {}
    out = {
        "ok": p.returncode == 0 and not o.get("is_error"),
        "text": o.get("result") or "",
        "cost_usd": float(o.get("total_cost_usd") or 0.0),
        "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens",
                                            "cache_creation_input_tokens",
                                            "cache_read_input_tokens")},
        "models": sorted((o.get("modelUsage") or {}).keys()),
        # per-model totals; the top-level `usage` can reflect only the last
        # request (it read output_tokens=0 on a 490 s solve in the probe)
        "model_usage": o.get("modelUsage") or {},
        "num_turns": o.get("num_turns"),
        "wall_s": wall,
        "api_s": (o.get("duration_api_ms") or 0) / 1000,
    }
    if not out["ok"]:
        out["error"] = f"rc={p.returncode} is_error={o.get('is_error')}: {out['text'][:300]}"
    elif model not in out["models"]:
        # the CLI falls back silently in some configurations; a solve by the
        # teacher's own model would defeat the point
        out["ok"] = False
        out["error"] = f"requested {model}, CLI used {out['models']}"
    return out


class Spend:
    """Cumulative total_cost_usd, on disk and in flight, against a hard limit.

    Starts from every cost recorded in the output rows plus the ledger of calls
    whose result was thrown away (a failed attempt, a solve whose adjudication
    failed), so a resumed job counts what earlier jobs spent. Once past the
    limit no new call starts; the calls already in flight finish.
    """
    LEDGER = OUT_DIR / "unrecorded_spend.jsonl"

    def __init__(self):
        self.limit = None
        self.lock = threading.Lock()
        self.spent = 0.0
        self.announced = False
        self.consecutive_failures = 0
        self.broken = False

    def load(self, limit):
        self.limit = limit
        total = 0.0
        for run in RUNS:
            p = OUT_DIR / f"{run}.jsonl"
            if p.exists():
                total += sum(row_cost(r) for r in map(json.loads, p.open()))
        if self.LEDGER.exists():
            total += sum(json.loads(l)["cost_usd"] for l in self.LEDGER.open())
        self.spent = total
        print(f"spend so far ${total:.2f}, limit "
              f"{'none' if limit is None else f'${limit:.2f}'}", flush=True)

    def add(self, cost):
        with self.lock:
            self.spent += cost

    def unrecorded(self, cost, why):
        if cost:
            with self.lock, self.LEDGER.open("a") as f:
                f.write(json.dumps({"t": time.time(), "cost_usd": cost, "why": why}) + "\n")

    def failed(self, ok):
        """Circuit breaker: 8 failed calls in a row (lost auth, quota) stop the job
        rather than marking every remaining problem skipped."""
        with self.lock:
            self.consecutive_failures = 0 if ok else self.consecutive_failures + 1
            if self.consecutive_failures >= 8 and not self.broken:
                self.broken = True
                print("CIRCUIT BREAK: 8 consecutive failed calls; no new calls", flush=True)

    def exceeded(self):
        with self.lock:
            if self.broken:
                return True
            over = self.limit is not None and self.spent >= self.limit
            if over and not self.announced:
                self.announced = True
                print(f"BUDGET STOP: spent ${self.spent:.2f} >= ${self.limit:.2f}; "
                      "no new calls", flush=True)
            return over


SPEND = Spend()


def row_cost(r):
    c = r["solve"]["cost_usd"]
    for key in ("adjudication", "adjudication_superseded"):
        if r.get(key):
            c += r[key]["cost_usd"]
    for call in (r.get("recheck") or {}).get("calls", []):
        c += call["cost_usd"]
    return c


def call_with_retry(*a, retries=1):
    res = {"ok": False, "error": "not attempted"}
    for attempt in range(retries + 1):
        if SPEND.exceeded():
            return {"ok": False, "error": "budget", "budget_stop": True, "cost_usd": 0.0}
        res = claude_call(*a)
        SPEND.add(res.get("cost_usd", 0.0))
        SPEND.failed(res["ok"])
        if res["ok"]:
            return res
        SPEND.unrecorded(res.get("cost_usd", 0.0), f"failed call: {res.get('error', '')[:120]}")
        print(f"  call failed (attempt {attempt + 1}): {res.get('error')}",
              file=sys.stderr, flush=True)
    return res


def parse_verdict(text, extra=()):
    """The last well-formed {"verdict": ...} object in the text, else None.

    Decoded from each '{' rather than matched by a brace-free regex: the answers
    quoted inside it are LaTeX, and `\\sqrt{3}` puts braces in the strings.
    """
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{\s*\"verdict\"", text))):
        try:
            v, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict) and v.get("verdict") in (
                "X", "Y", "both", "neither", "ill_posed", *extra):
            return v
    return None


def solve(row, a):
    return call_with_retry(SOLVE_PROMPT.format(problem=row["problem"]), SOLVE_SYSTEM,
                           a.model, a.timeout, a.max_budget, a.effort)


def adjudicate(row, solver_answer, a):
    """-> (call result, teacher_label). Labels blind, order seeded per row."""
    rng = random.Random(f"{a.seed}:{row['run']}:{row['step']}:{row['id']}")
    teacher_label = rng.choice("XY")
    shown_solver = solver_answer if solver_answer is not None else "(no final answer given)"
    x, y = ((row["answer"], shown_solver) if teacher_label == "X"
            else (shown_solver, row["answer"]))
    res = call_with_retry(ADJ_PROMPT.format(problem=row["problem"], x=x, y=y),
                          ADJ_SYSTEM, a.model, a.timeout, a.max_budget, a.effort)
    return res, teacher_label


def category(verdict, teacher_label):
    if verdict is None:
        return "unparsed"
    v = verdict["verdict"]
    if v == "both":
        return "equivalent"
    if v == "neither":
        return "both_wrong"
    if v == "ill_posed":
        return "ill_posed"
    return "solver_wrong" if v == teacher_label else "key_wrong"


def grade_solver(text, key):
    """-> (agree, extracted, source). The repo's grade, plus one solver-side rule.

    With thinking on, the CLI's `result` is only the final text block, and on
    hard problems Sonnet 5 often ends a long hidden derivation with a bare `33`
    and no box -- 11 of the first 25 disagreements were that. extract_answer
    finds nothing there, which says nothing about the key. So a final text that is
    one short line is graded as if boxed. The comparison itself is untouched.
    """
    agree, extracted = grade(answer_segment(text), key, integer_answer=False)
    if extracted is not None:
        return bool(agree), extracted, "boxed"
    bare = re.sub(r"[*`$]|\\\(|\\\)", "", text).strip().rstrip(".")
    if bare and "\n" not in bare and len(bare) <= 60:
        agree, extracted = grade(f"\\boxed{{{bare}}}", key, integer_answer=False)
        return bool(agree), extracted, "bare_final_text"
    return False, None, "none"


def done_keys(run, model):
    path = OUT_DIR / f"{run}.jsonl"
    if not path.exists():
        return set()
    return {(r["step"], r["id"]) for r in map(json.loads, path.open())
            if r["solver_model"] == model}


def run_check(rows, a):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {run: done_keys(run, a.model) for run in RUNS}
    todo = [r for r in rows if (r["step"], r["id"]) not in done[r["run"]]]
    print(f"{len(rows)} selected, {len(rows) - len(todo)} already done, "
          f"{len(todo)} to run  (model={a.model}, concurrency={a.concurrency})",
          flush=True)
    lock = threading.Lock()
    t_start = time.time()
    n_written = 0

    def write(rec):
        with lock, (OUT_DIR / f"{rec['run']}.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(a.concurrency) as pool:
        solves = {pool.submit(solve, r, a): r for r in todo}
        adjs = {}
        for fut in as_completed(solves):
            row = solves[fut]
            res = fut.result()
            if not res["ok"]:
                if not res.get("budget_stop"):
                    print(f"SKIP {row['run']} step {row['step']} {row['id']}: solve failed",
                          flush=True)
                continue
            # grade in the main thread, exactly as the reward did
            agree, extracted, source = grade_solver(res["text"], row["answer"])
            rec = {**row, "teacher_model": TEACHER_MODEL, "solver_model": a.model,
                   "selection": a.selection, "solve_prompt_version": SOLVE_PROMPT_VERSION,
                   "solve": {k: v for k, v in res.items() if k != "ok"},
                   "solver_answer": extracted, "solver_answer_source": source,
                   "agree": bool(agree)}
            if agree:
                rec.update(adjudication=None, category="agree")
                write(rec)
                n_written += 1
                print(f"agree      {row['run'][-6:]} s{row['step']:<2} {row['id']}", flush=True)
            else:
                adjs[pool.submit(adjudicate, row, extracted, a)] = rec
        for fut in as_completed(adjs):
            rec = adjs[fut]
            res, teacher_label = fut.result()
            if not res["ok"]:
                SPEND.unrecorded(rec["solve"]["cost_usd"],
                                 f"solve discarded, adjudication failed: {rec['run']} "
                                 f"step {rec['step']} {rec['id']}")
                print(f"SKIP {rec['run']} step {rec['step']} {rec['id']}: "
                      "adjudication failed", flush=True)
                continue
            verdict = parse_verdict(res["text"])
            rec["adjudication"] = {**{k: v for k, v in res.items() if k != "ok"},
                                   "teacher_label": teacher_label, "verdict": verdict}
            rec["category"] = category(verdict, teacher_label)
            write(rec)
            n_written += 1
            print(f"{rec['category']:<12}{rec['run'][-6:]} s{rec['step']:<2} {rec['id']}  "
                  f"key={rec['answer']!r} solver={rec['solver_answer']!r}", flush=True)
    wall = time.time() - t_start
    print(f"wrote {n_written} rows in {wall:.0f}s wall; spend now ${SPEND.spent:.2f}"
          + ("  (stopped by budget)" if SPEND.announced else "")
          + ("  (stopped by circuit breaker)" if SPEND.broken else ""), flush=True)


CATEGORIES = ("agree", "equivalent", "key_wrong", "solver_wrong", "both_wrong",
              "ill_posed", "unparsed", "unsolved", "unsolved_key_confirmed")


def final_category(r):
    """The recheck's category where it reached one, else the adjudication's."""
    rc = r.get("recheck")
    if not (rc and rc.get("category")):
        return r["category"]
    if r["category"] == "unsolved" and rc["category"] == "solver_wrong":
        return "unsolved_key_confirmed"    # no solve, but the computation agrees with the key
    return rc["category"]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def summarise(recs):
    n = len(recs)
    cats = {c: 0 for c in CATEGORIES}
    for r in recs:
        cats[final_category(r)] = cats.get(final_category(r), 0) + 1
    wrong = cats["key_wrong"] + cats["both_wrong"]
    bad_steps = {}
    for r in recs:
        if final_category(r) in ("key_wrong", "both_wrong", "ill_posed"):
            bad_steps.setdefault((r["run"], r["step"]), []).append(r["id"])
    adj = [r[k] for r in recs for k in ("adjudication", "adjudication_superseded")
           if r.get(k)]
    rc_calls = [c for r in recs for c in (r.get("recheck") or {}).get("calls", [])]
    rechecked = [r["recheck"] for r in recs if r.get("recheck")]
    return {
        "n": n, **cats,
        "key_wrong_rate_wilson95": wilson(wrong, n),
        "ill_posed_rate_wilson95": wilson(cats["ill_posed"], n),
        "steps_with_bad_keys": {f"{run} step_{s}": f"{len(ids)}/16 {ids}"
                                for (run, s), ids in sorted(bad_steps.items())},
        "recheck": {"rows": len(rechecked),
                    "status": {st: sum(x["status"] == st for x in rechecked)
                               for st in sorted({x["status"] for x in rechecked})},
                    "overturned": sum(bool(x.get("overturned")) for x in rechecked)},
        "cost_usd": {"solve": round(sum(r["solve"]["cost_usd"] for r in recs), 3),
                     "adjudication": round(sum(x["cost_usd"] for x in adj), 3),
                     "adjudication_calls": len(adj),
                     "recheck": round(sum(c["cost_usd"] for c in rc_calls), 3),
                     "recheck_calls": len(rc_calls)},
        "wall_s": {"solve_sum": round(sum(r["solve"]["wall_s"] for r in recs)),
                   "adjudication_sum": round(sum(x["wall_s"] for x in adj)),
                   "recheck_sum": round(sum(x.get("wall_s", 0) for x in rechecked))},
    }


def summary(model, selection=None):
    """Per run and overall. Key-wrong = key_wrong + both_wrong (the key is wrong
    either way); ill_posed is its own bad-reward rate and is not merged in."""
    per_run = {}
    for run in RUNS:
        p = OUT_DIR / f"{run}.jsonl"
        if p.exists():
            per_run[run] = [r for r in map(json.loads, p.open()) if r["solver_model"] == model
                            and (selection is None or r.get("selection") == selection)]
    allrecs = [r for rs in per_run.values() for r in rs]
    if not allrecs:
        print("no results")
        return
    out = {"solver_model": model, "selection": selection or "any",
           "runs": {f"{run} ({RUNS[run]})": summarise(rs) for run, rs in per_run.items() if rs},
           "all": summarise(allrecs)}
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Python recheck: a written script, run by this orchestrator, then a verdict
# ---------------------------------------------------------------------------

SCRIPT_SYSTEM = ("You are a careful competition mathematician who checks answers by "
                 "writing Python. You have no tools; you only write the script.")
SCRIPT_PROMPT = """A math problem and two proposed final answers, labelled X and Y, are below. They disagree, and a referee is unsure which (if either) is right.

Write ONE self-contained Python 3 script that settles it by computation: compute the answer directly, or brute-force / enumerate / simulate exactly where that is feasible, and compare the result with X and Y. Rules for the script:
- standard library, sympy and numpy only; no network, no files, no input
- it must finish within 60 seconds on one CPU core, so bound any search
- prefer exact arithmetic (integers, fractions.Fraction, sympy) to floating point
- print what it computed and, at the end, which of X / Y it agrees with (or neither)
- if the statement is ambiguous, have the script check each reading and print each result

If this problem genuinely cannot be checked computationally (for example a proof, or a quantity with no finite computation), do not write a script; reply with a single line `NOT_CHECKABLE: <reason>`.

Otherwise reply with the script in one ```python fenced block and nothing else of substance.

## Problem
{problem}

## Answer X
{x}

## Answer Y
{y}"""

SCRIPT_FIX_PROMPT = """{previous}

The script you wrote for this failed when run ({status}). Its stderr (truncated):
```
{stderr}
```
Its stdout (truncated):
```
{stdout}
```
Write a corrected script under the same rules, in one ```python fenced block."""

RECHECK_SYSTEM = ADJ_SYSTEM
RECHECK_PROMPT = """A math problem and two proposed final answers, labelled X and Y, are below. A Python script was written to check them and was run in a sandbox; the script and its output follow.

Decide which answer is correct, using the computation as evidence but checking that the script actually models the problem as stated (a script can have a bug, or solve a different problem). The possible verdicts:
- "X": answer X is correct and answer Y is not
- "Y": answer Y is correct and answer X is not
- "both": X and Y are the same answer written differently, and it is correct
- "neither": both answers are wrong
- "ill_posed": the problem as stated is ambiguous, underdetermined or contradictory, so it has no single correct key
- "inconclusive": the computation does not settle the question (it failed, timed out, or does not model the problem) -- use this rather than guessing

## Problem
{problem}

## Answer X
{x}

## Answer Y
{y}

## Script
```python
{script}
```

## Run: {status}
stdout (truncated):
```
{stdout}
```
stderr (truncated):
```
{stderr}
```

End your response with exactly one line of JSON and nothing after it:
{{"verdict": "X" | "Y" | "both" | "neither" | "ill_posed" | "inconclusive", "correct_answer": "<the correct answer, or empty>", "reason": "<one sentence>"}}"""

SCRIPT_TIMEOUT_S = 120
SCRIPT_RLIMITS = ["--as=8000000000", "--cpu=120", "--fsize=50000000", "--nofile=256"]
OUT_TRUNC = 6000
BWRAP = "/home/linuxbrew/.linuxbrew/bin/bwrap"


def sandbox_argv(workdir):
    """-> (argv prefix, mode). Probed once; never falls back to no isolation.

    bubblewrap first: new user, net, pid, ipc and uts namespaces, and a
    filesystem of /usr and the conda env read-only plus the script's own temp
    dir -- no network and no view of the repo or the run directory. If bwrap is
    unusable, `unshare -rn` (network namespace only; the filesystem stays
    visible). Neither -> refuse to run scripts.
    """
    env = sys.prefix
    if Path(BWRAP).exists():
        return ([BWRAP, "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
                 "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
                 "--ro-bind", env, env, "--bind", str(workdir), "/work",
                 "--chdir", "/work", "--proc", "/proc", "--dev", "/dev",
                 "--tmpfs", "/tmp", "--unshare-all", "--die-with-parent",
                 "--new-session"], "bwrap --unshare-all")
    return (["unshare", "-rn"], "unshare -rn")


def probe_sandbox():
    """Run a script that must fail to reach the network, under each isolation."""
    test = ("import socket,sympy,numpy\n"
            "try:\n socket.create_connection(('1.1.1.1', 80), timeout=3); print('NET')\n"
            "except OSError: print('NONET')\n")
    results = {}
    for use_bwrap in (True, False):
        with tempfile.TemporaryDirectory(prefix="keycheck_probe_") as d:
            Path(d, "s.py").write_text(test)
            if use_bwrap:
                if not Path(BWRAP).exists():
                    results["bwrap"] = "absent"
                    continue
                prefix, mode = sandbox_argv(d)
                script = "/work/s.py"
            else:
                prefix, mode = ["unshare", "-rn"], "unshare -rn"
                script = str(Path(d, "s.py"))
            try:
                p = subprocess.run(["prlimit", *SCRIPT_RLIMITS, "--", *prefix,
                                    sys.executable, "-I", script], cwd=d,
                                   capture_output=True, text=True, timeout=60,
                                   env=_script_env(d if not use_bwrap else "/work"))
                results[mode] = p.stdout.strip() or f"rc={p.returncode} {p.stderr[-200:]}"
            except Exception as e:  # noqa: BLE001
                results[mode] = f"error {e}"
    return results


def _script_env(home):
    return {"PATH": f"{sys.prefix}/bin:/usr/bin:/bin", "HOME": str(home),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}


def run_script(code, mode):
    """Run model-written code in a fresh temp dir under rlimits and the sandbox."""
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="keycheck_py_") as d:
        Path(d, "check.py").write_text(code)
        if mode.startswith("bwrap"):
            prefix, _ = sandbox_argv(d)
            script, home = "/work/check.py", "/work"
        else:
            prefix, script, home = ["unshare", "-rn"], str(Path(d, "check.py")), d
        argv = ["prlimit", *SCRIPT_RLIMITS, "--", *prefix, sys.executable, "-I", script]
        p = subprocess.Popen(argv, cwd=d, env=_script_env(home), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        try:
            out, err = p.communicate(timeout=SCRIPT_TIMEOUT_S)
            status = f"exit code {p.returncode}"
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)       # by process group id, never by pattern
            out, err = p.communicate()
            status = f"killed after {SCRIPT_TIMEOUT_S}s timeout"
            timed_out = True
    return {"status": status, "returncode": p.returncode, "timed_out": timed_out,
            "stdout": out[-OUT_TRUNC:], "stderr": err[-OUT_TRUNC:],
            "wall_s": round(time.time() - t0, 2)}


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None


def needs_recheck(r):
    if r.get("recheck"):
        return False
    if r["run"] == "run_20260902_072857":
        return r["category"] not in ("agree", "equivalent")
    return r["category"] in ("key_wrong", "both_wrong", "ill_posed", "unparsed", "unsolved")


def recheck_one(r, a, mode):
    """-> recheck dict. Every claude call's result lands in `calls`."""
    rng = random.Random(f"recheck:{a.seed}:{r['run']}:{r['step']}:{r['id']}")
    teacher_label = rng.choice("XY")
    shown_solver = (r["solver_answer"] if r["solver_answer"] is not None
                    else "(no final answer given)")
    x, y = ((r["answer"], shown_solver) if teacher_label == "X"
            else (shown_solver, r["answer"]))
    rec = {"teacher_label": teacher_label, "sandbox": mode, "calls": [], "attempts": []}
    t0 = time.time()

    def call(stage, prompt, system):
        res = call_with_retry(prompt, system, a.model, a.timeout, a.max_budget, a.effort)
        rec["calls"].append({"stage": stage, **{k: v for k, v in res.items()
                                                if k not in ("ok", "text")},
                             "text": res.get("text", "")})
        return res

    def abandon():
        SPEND.unrecorded(sum(c.get("cost_usd", 0.0) for c in rec["calls"] if "error" not in c),
                         f"recheck abandoned: {r['run']} step {r['step']} {r['id']}")

    prompt = SCRIPT_PROMPT.format(problem=r["problem"], x=x, y=y)
    res = call("script", prompt, SCRIPT_SYSTEM)
    if not res["ok"]:
        abandon()
        return None
    for attempt in range(2):
        text = res["text"]
        code = extract_code(text)
        if code is None:
            m = re.search(r"NOT_CHECKABLE:\s*(.*)", text)
            rec.update(status="not_checkable" if m else "no_script",
                       reason=(m.group(1).strip() if m else text[-300:]),
                       verdict=None, category=None)
            rec["wall_s"] = round(time.time() - t0, 1)
            return rec
        run = run_script(code, mode)
        rec["attempts"].append({"script": code, **run})
        if run["returncode"] == 0 and not run["timed_out"]:
            break
        if attempt == 0:
            res = call("script_fix", SCRIPT_FIX_PROMPT.format(
                previous=prompt, status=run["status"], stderr=run["stderr"][-2000:],
                stdout=run["stdout"][-2000:]), SCRIPT_SYSTEM)
            if not res["ok"]:
                abandon()
                return None
    last = rec["attempts"][-1]
    res = call("verdict", RECHECK_PROMPT.format(
        problem=r["problem"], x=x, y=y, script=last["script"], status=last["status"],
        stdout=last["stdout"], stderr=last["stderr"]), RECHECK_SYSTEM)
    if not res["ok"]:
        abandon()
        return None
    v = parse_verdict(res["text"], extra=("inconclusive",))
    rec["verdict"] = v
    if v is None:
        rec.update(status="unparsed", category=None)
    elif v["verdict"] == "inconclusive":
        rec.update(status="inconclusive", category=None)
    else:
        rec.update(status="verdict", category=category(v, teacher_label))
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def load_all():
    files = {run: OUT_DIR / f"{run}.jsonl" for run in RUNS}
    return files, {run: [json.loads(l) for l in p.open()] for run, p in files.items()
                   if p.exists()}


def save_run(files, rows, run):
    tmp = files[run].with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[run]))
    tmp.replace(files[run])


def mark_unsolved(a):
    """Write an `unsolved` row for every training problem still missing one.

    For problems the solver could not finish inside the timeout on any attempt
    (logged in the job's log). The row carries no solve and no cost -- a killed
    call reports none -- and goes to the Python recheck, which checks the key
    on its own.
    """
    files, rows = load_all()
    done = {(run, r["step"], r["id"]) for run, rs in rows.items() for r in rs
            if r["solver_model"] == a.model}
    missing = [r for r in load_rows() if (r["run"], r["step"], r["id"]) not in done]
    for row in missing:
        rec = {**row, "teacher_model": TEACHER_MODEL, "solver_model": a.model,
               "selection": "all", "solve_prompt_version": SOLVE_PROMPT_VERSION,
               "solve": {"cost_usd": 0.0, "wall_s": 0.0, "text": "",
                         "error": a.unsolved_reason},
               "solver_answer": None, "solver_answer_source": "none", "agree": False,
               "adjudication": None, "category": "unsolved"}
        with (OUT_DIR / f"{row['run']}.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"unsolved {row['run'][-6:]} s{row['step']:<2} {row['id']}", flush=True)


def run_regrade(a):
    """Apply grade_solver to rows written before it existed (solver_answer None).

    A row that now agrees drops its adjudication (kept as adjudication_superseded,
    its cost still counted). One that still disagrees is re-adjudicated, since
    the first adjudication was shown "(no final answer given)" instead of the
    solver's actual answer. No solve is repeated.
    """
    files, rows = load_all()
    todo = [(run, i) for run, rs in rows.items() for i, r in enumerate(rs)
            if r["solver_model"] == a.model and r["category"] != "agree"
            and r["solver_answer"] is None and "solver_answer_source" not in r]
    print(f"{len(todo)} rows with no extracted solver answer", flush=True)
    readj = []
    for run, i in todo:
        r = rows[run][i]
        agree, extracted, source = grade_solver(r["solve"]["text"], r["answer"])
        r.update(solver_answer=extracted, solver_answer_source=source, agree=agree,
                 adjudication_superseded=r["adjudication"],
                 regraded="solver answer re-extracted by grade_solver; no new solve")
        if agree:
            r.update(adjudication=None, category="agree")
        elif extracted is not None:
            readj.append((run, i))
        else:
            r["adjudication_superseded"] = None      # nothing new to show; keep it
        print(f"regrade {run[-6:]} s{r['step']:<2} {r['id']}: key={r['answer']!r} "
              f"bare={extracted!r} -> {'agree' if agree else 'still disagrees'}", flush=True)
    for run in rows:
        save_run(files, rows, run)
    with ThreadPoolExecutor(a.concurrency) as pool:
        futs = {pool.submit(adjudicate, rows[run][i], rows[run][i]["solver_answer"], a):
                (run, i) for run, i in readj}
        for fut in as_completed(futs):
            run, i = futs[fut]
            r = rows[run][i]
            res, teacher_label = fut.result()
            if not res["ok"]:
                print(f"re-adjudication failed {run[-6:]} s{r['step']} {r['id']}", flush=True)
                continue
            verdict = parse_verdict(res["text"])
            r["adjudication"] = {**{k: v for k, v in res.items() if k != "ok"},
                                 "teacher_label": teacher_label, "verdict": verdict}
            r["category"] = category(verdict, teacher_label)
            save_run(files, rows, run)
            print(f"re-adjudicated {run[-6:]} s{r['step']:<2} {r['id']}: {r['category']}",
                  flush=True)


def run_recheck(a):
    probe = probe_sandbox()
    print(f"sandbox probe: {probe}", flush=True)
    if probe.get("bwrap --unshare-all") == "NONET":
        mode = "bwrap --unshare-all"
    elif probe.get("unshare -rn") == "NONET":
        mode = "unshare -rn"
    else:
        sys.exit("no working network isolation; refusing to run model-written code")
    files, rows = load_all()
    todo = [(run, i) for run, rs in rows.items() for i, r in enumerate(rs)
            if r["solver_model"] == a.model and needs_recheck(r)]
    print(f"{len(todo)} rows to recheck (sandbox={mode}, concurrency={a.concurrency})",
          flush=True)
    lock = threading.Lock()
    t_start = time.time()

    with ThreadPoolExecutor(a.concurrency) as pool:
        futs = {pool.submit(recheck_one, rows[run][i], a, mode): (run, i) for run, i in todo}
        for fut in as_completed(futs):
            run, i = futs[fut]
            r = rows[run][i]
            rc = fut.result()
            if rc is None:
                print(f"SKIP recheck {run[-6:]} s{r['step']} {r['id']}", flush=True)
                continue
            rc["adjudication_category"] = r["category"]
            # an unsolved row had no adjudication to overturn; the recheck is the
            # only independent check its key gets
            rc["overturned"] = (r["category"] != "unsolved" and rc["category"] is not None
                                and rc["category"] != r["category"])
            with lock:
                r["recheck"] = rc
                save_run(files, rows, run)
            print(f"recheck {run[-6:]} s{r['step']:<2} {r['id']}: adjudication={r['category']} "
                  f"recheck={rc['status']}/{rc['category']}"
                  f"{'  OVERTURNED' if rc['overturned'] else ''}", flush=True)
    print(f"recheck done in {time.time() - t_start:.0f}s wall; spend now ${SPEND.spent:.2f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, help="stratified sample of this many problems")
    g.add_argument("--all", action="store_true", help="every training problem (960)")
    g.add_argument("--summary", action="store_true", help="summarise results on disk")
    g.add_argument("--mark-unsolved", action="store_true",
                   help="record every still-missing problem as unsolved (no calls)")
    g.add_argument("--regrade", action="store_true",
                   help="re-extract bare final answers on rows written before grade_solver")
    g.add_argument("--recheck", action="store_true",
                   help="Python recheck of disagreements already on disk (see needs_recheck)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5",
                    help="solver and adjudicator; must not be the teacher's model")
    ap.add_argument("--effort", default="", help="--effort passed to claude (default: CLI's)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per call")
    ap.add_argument("--max-budget", type=float, default=3.0, help="USD cap per call")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop starting calls once cumulative spend (all rows on disk "
                         "plus discarded calls) reaches this")
    ap.add_argument("--unsolved-reason", default="solve timed out on every attempt",
                    help="with --mark-unsolved: recorded on each row")
    ap.add_argument("--selection", default=None,
                    help="with --summary: only rows from this selection label")
    a = ap.parse_args()

    if a.summary:
        summary(a.model, a.selection)
        return
    if a.model.split("[")[0] == TEACHER_MODEL:
        sys.exit(f"--model {a.model} is the teacher's model; the check must be independent")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPEND.load(a.budget_usd)
    if a.mark_unsolved:
        mark_unsolved(a)
        return
    if a.regrade:
        run_regrade(a)
        return
    if a.recheck:
        run_recheck(a)
        return
    rows = load_rows()
    assert len(rows) == 320 * len(RUNS), f"expected 960 rows, found {len(rows)}"
    if a.all:
        a.selection = "all"
    else:
        rows = stratified_sample(rows, a.sample, a.seed)
        a.selection = f"sample{a.sample}_seed{a.seed}"
    run_check(rows, a)


if __name__ == "__main__":
    main()
