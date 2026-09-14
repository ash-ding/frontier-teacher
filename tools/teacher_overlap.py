#!/usr/bin/env python3
"""Did the teacher write its problems, or copy benchmark items? Measured, not read.

    python tools/teacher_overlap.py              # -> outputs/analysis/teacher_overlap.json
    python tools/teacher_overlap.py --check 200  # pruned nearest-neighbour search == naive
    python tools/teacher_overlap.py --contamination  # -> outputs/analysis/benchmark_contamination.json

§8 says the 5,152 problems the teacher wrote across the three open-ended runs
are "written, not retrieved", from reading three files. The objection that
leaves open is the cheapest one against the whole result: a benchmark problem in
a training curriculum is a leaked evaluation item, whether the teacher recalled
it or found it on disk. This measures it.

Every `step_*/train/data.jsonl` and `step_*/eval_*/data.jsonl` row is compared
with every problem of MATH-500, AIME 2020-2024 and the three HMMT sets -- the
full files the eval configs read, not the reference fifths -- and with the
11,996-problem MATH training pool. Pool items leak no evaluation, but they say
whether the teacher retrieves known problems at all. Per (statement, set):

  exact       normalised statement equality, twice: goal_audit.py's normaliser
              (lowercase, non-alphanumerics to spaces, collapse whitespace), and
              the same after dropping [asy] blocks and typesetting-only LaTeX
              (\\left, \\text, \\dfrac -> frac) on both sides, so markup or a missing
              figure cannot hide one. `contained` catches one statement embedded
              in the other. Either counts as a copy only if the answers also agree
              (the normaliser drops signs, so z^4 - z^2 + 1 "equals" z^4 + z^2 + 1)
  13-grams    shared 13-word spans (goal_audit.py's test, on statements with
              [asy] code removed), how a paraphrase shows up once the wording
              has drifted
  nearest     the most similar statement in the set by difflib character ratio,
              over EVERY pair -- no candidate prefilter, so nothing is missed.
              difflib's real_quick_ratio / quick_ratio are upper bounds on
              ratio, which prunes most pairs exactly; --check confirms it
  answers     whether the nearest pair's answers agree, by the repository's
              grader (eval/verifiers/extract.py). A copied problem keeps its
              answer; a same-form exercise with new numbers usually does not

Two flags decide what a match would mean. `phase` is `train` (the statement
reached the student; every later checkpoint is downstream of it) or `eval_k` (a
probe the teacher graded and never trained on). `nearest_in_ref` marks a
benchmark item in the reference fifth the teacher was allowed to see
(`data/benchmark/*_ref*.manifest.json`): copying one of those contaminates the
seen fifth, copying any other contaminates the held-out four fifths §8 reports.

A similarity number is not a verdict -- a teacher drilling the right machinery
writes problems of the same form as benchmark items. Every flagged pair is
written out with both statements and answers so it can be read. Two things the
text alone cannot show are recorded beside it: `provenance`, the teacher's own
tool calls that opened benchmark or pool files, the main checkout, or the
network; and `leak_effect`, for any benchmark item copied verbatim into a
curriculum, the student's score on exactly those items before and after, and
the held-out headline with them removed.

The teacher's scratch drafts (other *.jsonl in eval dirs) are matched too and
reported apart; they reached neither the student nor an evaluation.

--contamination turns the same machinery on what every arm TRAINED on -- the
section 7 band subsets (verified byte for byte against what verl read) and the
teacher runs' curricula -- and writes one exclusion set per (model, benchmark),
so all arms of a model can be re-scored on the same clean held-out ids. See the
block above contamination_main for how the training files are established.
"""
import argparse
import difflib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import pass_at_k  # noqa: E402

RUNS = ["run_20260902_072619", "run_20260902_072856", "run_20260902_072857"]
BENCH = {
    "math500": ["data/benchmark/math500.jsonl"],
    "aime": [f"data/benchmark/aime_{y}.jsonl" for y in range(2020, 2025)],
    "hmmt": ["data/benchmark/hmmt_feb_2025.jsonl", "data/benchmark/hmmt_nov_2025.jsonl",
             "data/benchmark/hmmt_feb_2026.jsonl"],
    "pool": ["data/training_set/math_train_orig_train.jsonl",
             "data/training_set/math_train_orig_test.jsonl"],
}
REF_MANIFEST = {"math500": "math500_ref100", "aime": "aime_ref30", "hmmt": "hmmt_ref19"}
HEADLINE_K = {"math500": 1, "aime": 4, "hmmt": 4}
FINAL_STEPS = (4, 9, 14, 19)
N_GRAM = 13
THRESH = (0.9, 0.8, 0.7, 0.6, 0.5)
LIST_RATIO = 0.8            # benchmark pairs at or above this are written out in full
SOURCE_ID = re.compile(r"(aime-\d{4}|math500-\d{4}|hmmt-(feb|nov)-\d{4}|mathtrain-\d{5})", re.I)
PROVENANCE = {
    "benchmark_files": re.compile(r"data/benchmark|aime_20\d\d\.jsonl|hmmt_(feb|nov)_|math500(_ref100)?\.jsonl"),
    "pool_files": re.compile(r"data/training_set|math_train_orig|further_improve"),
    "main_checkout": re.compile(r"/code/frontier-teacher\b"),
    "network": re.compile(r"\b(curl|wget|urllib|requests\.get|httpx|huggingface|hf_hub|load_dataset|"
                          r"snapshot_download|https?://)", re.I),
}


# ---------------------------------------------------------------- normalising
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


TEX_DROP = {"left", "right", "displaystyle", "text", "textbf", "textit", "mathrm", "mathbf",
            "mathit", "operatorname", "quad", "qquad", "cdot", "cdots", "ldots", "dots", "dotsb"}
TEX_SYN = {"dfrac": "frac", "tfrac": "frac", "le": "leq", "ge": "geq", "dbinom": "binom",
           "tbinom": "binom", "lvert": "vert", "rvert": "vert"}


def strip_asy(s):
    """[asy] figure code: shared drawing boilerplate is not shared wording."""
    return re.sub(r"\[asy\].*?\[/asy\]", " ", str(s), flags=re.S)


def norm_tex(s):
    """norm() after removing [asy] blocks and typesetting-only commands. Commands that
    carry meaning (\\cos, \\sqrt, \\binom) are kept as words, so dropping LaTeX cannot
    make "Compute \\cos 120" equal "Compute \\tan 120"."""
    s = strip_asy(s)
    s = re.sub(r"\\([a-zA-Z]+)",
               lambda m: " " if m.group(1) in TEX_DROP else f" {TEX_SYN.get(m.group(1), m.group(1))} ", s)
    return norm(s)


def grams(text, n):
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def ans_str(a):
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", str(a))
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\^\\circ|\^\{\\circ\}|\\%|%|\\!|\\,|\\left|\\right|[\s$]", "", s)
    return s.rstrip(".")


def answers_agree(ta, ba, integer):
    if ans_str(ta) == ans_str(ba):
        return True
    try:
        from verifiers.extract import grade
        return bool(grade(f"\\boxed{{{ta}}}", str(ba), integer_answer=integer)[0])
    except Exception:
        return False


# ---------------------------------------------------------------- loading
def load_teacher():
    rows, scratch, superseded, provenance = [], [], [], []
    for run in RUNS:
        rd = ROOT / "outputs/frontier-model" / run
        model = json.loads((rd / "step_0/config.json").read_text())["student"]["grpo_preset"]
        for f in sorted(rd.glob("step_*/*/*.jsonl")):
            phase = f.parent.name
            if not (phase == "train" or phase.startswith("eval_")) or \
                    f.name in ("records.jsonl", "generations.jsonl", "train.verl.jsonl"):
                continue
            sink = rows if f.name == "data.jsonl" else scratch
            for i, line in enumerate(f.open()):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sink.append({"run": run, "model": model,
                             "step": int(f.parent.parent.name.split("_")[1]), "phase": phase,
                             "file": str(f.relative_to(ROOT)), "line": i, "id": str(r.get("id")),
                             "problem": str(r.get("problem", "")), "answer": str(r.get("answer", ""))})
        sup = [p for p in rd.glob("pipeline.superseded_*/**/*") if p.is_file()]
        superseded.append({"run": run,
                           "dirs": sorted({p.relative_to(rd).parts[0] for p in sup}),
                           "files": len(sup),
                           "problem_files": [str(p.relative_to(ROOT)) for p in sup if p.suffix == ".jsonl"]})
        provenance += scan_tool_calls(rd, run, model)
    return rows, scratch, superseded, provenance


def scan_tool_calls(rd, run, model):
    """The teacher's own tool calls that touched benchmark/pool files, the checkout, or the net."""
    out = []
    for log in sorted(rd.glob("step_*/*/teacher.log")):
        hits = defaultdict(list)
        for line in log.open():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") != "assistant":
                continue
            for b in o.get("message", {}).get("content", []):
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    continue
                s = json.dumps(b.get("input", {}))
                if b.get("name") in ("WebFetch", "WebSearch"):
                    hits["network"].append(f"[{b['name']}] {s[:300]}")
                for cat, rx in PROVENANCE.items():
                    if rx.search(s):
                        hits[cat].append(f"[{b.get('name')}] {s[:300]}")
        if hits:
            out.append({"run": run, "model": model, "step": int(log.parent.parent.name.split("_")[1]),
                        "phase": log.parent.name, "log": str(log.relative_to(ROOT)),
                        "calls": {k: v for k, v in hits.items()}})
    return out


def load_refs():
    refs = {}
    for name, files in BENCH.items():
        items = [json.loads(l) for f in files for l in (ROOT / f).open()]
        ref_ids = set()
        if name in REF_MANIFEST:
            ref_ids = set(json.loads((ROOT / f"data/benchmark/{REF_MANIFEST[name]}.manifest.json")
                                     .read_text())["ids"])
            assert ref_ids <= {it["id"] for it in items}, name
        tex = [norm_tex(it["problem"]) for it in items]
        g13t, g13p = defaultdict(list), defaultdict(list)
        for j, it in enumerate(items):
            for g in grams(tex[j], N_GRAM):
                g13t[g].append(j)
            for g in grams(norm(strip_asy(it["problem"])), N_GRAM):
                g13p[g].append(j)
        by_plain, by_tex = {}, {}
        for it, tx in zip(items, tex):
            by_plain.setdefault(norm(it["problem"]), it["id"])
            by_tex.setdefault(tx, it["id"])
        refs[name] = {"files": files, "items": items, "ref_ids": ref_ids, "tex": tex,
                      "len": np.array([len(t) for t in tex]), "g13t": g13t, "g13p": g13p,
                      "by_plain": by_plain, "by_tex": by_tex}
    return refs


# ---------------------------------------------------------------- matching
G = {}  # filled before fork


def nearest(tt, R, prune=True):
    """Exact argmax of SequenceMatcher(bench, teacher).ratio() over the set.

    The teacher string is seq2, so its index is built once per statement.
    real_quick_ratio and quick_ratio bound ratio from above, so a pair whose
    bound is below the best so far cannot be the argmax and is skipped."""
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(tt)
    order = np.argsort(np.abs(R["len"] - len(tt)), kind="stable") if prune else range(len(R["tex"]))
    best, arg = -1.0, 0
    for j in order:
        sm.set_seq1(R["tex"][j])
        if prune and (sm.real_quick_ratio() <= best or sm.quick_ratio() <= best):
            continue
        r = sm.ratio()
        if r > best:
            best, arg = r, int(j)
    return best, arg


def match_one(u):
    t = G["uniq"][u]
    tt, tp = norm_tex(t["problem"]), norm(t["problem"])
    t13t, t13p, t5 = grams(tt, N_GRAM), grams(norm(strip_asy(t["problem"])), N_GRAM), grams(tt, 5)
    out = {}
    for name, R in G["refs"].items():
        c13 = Counter(j for g in t13t for j in R["g13t"].get(g, ()))
        c13p = Counter(j for g in t13p for j in R["g13p"].get(g, ()))
        both = c13 + c13p
        r, j = nearest(tt, R)
        b = R["items"][j]
        bt = R["tex"][j]
        out[name] = {
            "exact_plain": R["by_plain"].get(tp), "exact_tex": R["by_tex"].get(tt),
            "n13_items": len(set(c13) | set(c13p)),
            "n13_max": max(max(c13.values(), default=0), max(c13p.values(), default=0)),
            "n13_id": R["items"][both.most_common(1)[0][0]]["id"] if both else None,
            "nearest_id": b["id"], "ratio": round(r, 4),
            "shared_5grams": len(t5 & grams(bt, 5)),
            "contained": len(min(tt, bt, key=len)) >= 40 and (tt in bt or bt in tt),
            "nearest_in_ref": (b["id"] in R["ref_ids"]) if R["ref_ids"] else None,
            "answer_match": answers_agree(t["answer"], b.get("answer", ""), name == "aime"),
        }
    return u, out


def check_one(u):
    tt = norm_tex(G["uniq"][u]["problem"])
    return u, {n: (nearest(tt, R)[0], nearest(tt, R, prune=False)[0])
               for n, R in G["refs"].items() if n != "pool"}


def is_exact(f):
    return f["exact_plain"] is not None or f["exact_tex"] is not None


def is_copy(f):
    """Same statement: normalised-equal or one embedded in the other, AND the same answer.
    Containment alone is not enough -- "five-digit palindromes" sits inside "five-digit
    palindromes divisible by 11", a new problem. Neither is normalised equality: the
    normaliser drops signs, so z^4 - z^2 + 1 (pool, answer 12) "equals" z^4 + z^2 + 1
    (math500-0046, answer 6). An equal statement with a different answer is read instead."""
    return (is_exact(f) or f["contained"]) and f["answer_match"]


# ---------------------------------------------------------------- reporting
def summarise(rows, flags, refs, keyfn):
    groups = defaultdict(list)
    for r in rows:
        for k in keyfn(r):
            groups[k].append(r)
    out = {}
    for name in refs:
        per = {}
        for k, rs in sorted(groups.items()):
            fl = [flags[r["uniq"]][name] for r in rs]
            ratios = np.array([flags[u][name]["ratio"] for u in {r["uniq"] for r in rs}])
            d = {"rows": len(rs), "unique_statements": len({r["uniq"] for r in rs}),
                 "exact_rows": sum(is_exact(f) for f in fl),
                 "exact_answer_mismatch_rows": sum(is_exact(f) and not f["answer_match"] for f in fl),
                 "copy_rows": sum(is_copy(f) for f in fl),
                 "n13_rows": sum(f["n13_items"] > 0 for f in fl),
                 "n13_max": max((f["n13_max"] for f in fl), default=0),
                 "nearest_ratio_quantiles_unique": {str(q): round(float(np.quantile(ratios, q)), 4)
                                                    for q in (0.5, 0.9, 0.99, 1.0)}}
            if refs[name]["ref_ids"]:
                d["copy_rows_ref_fifth"] = sum(is_copy(f) and f["nearest_in_ref"] for f in fl)
                d["copy_rows_held_out"] = sum(is_copy(f) and not f["nearest_in_ref"] for f in fl)
            for th in THRESH:
                hit = [f for f in fl if f["ratio"] >= th]
                e = {"rows": len(hit), "answer_match_rows": sum(f["answer_match"] for f in hit)}
                if refs[name]["ref_ids"]:
                    e["ref_fifth_rows"] = sum(bool(f["nearest_in_ref"]) for f in hit)
                    e["held_out_rows"] = sum(f["nearest_in_ref"] is False for f in hit)
                d[f"ratio_ge_{th}"] = e
            per[k] = d
        out[name] = per
    return out


def pair_record(u, occ, flags, refs, name):
    f = flags[u][name]
    b = next(it for it in refs[name]["items"] if it["id"] == f["nearest_id"])
    return {"ratio": f["ratio"], "n13_max": f["n13_max"], "exact": is_exact(f), "copy": is_copy(f),
            "contained": f["contained"], "answer_match": f["answer_match"],
            "nearest_in_ref": f["nearest_in_ref"],
            "reached_student": any(o["phase"] == "train" for o in occ),
            "teacher": {"problem": occ[0]["problem"], "answer": occ[0]["answer"],
                        "occurrences": [{k: o[k] for k in ("model", "run", "step", "phase", "id", "file", "line")}
                                        for o in occ]},
            "benchmark": {"id": b["id"], "problem": b["problem"], "answer": b.get("answer")}}


def pairs(rows, flags, refs, name, top=20):
    by_u = defaultdict(list)
    for r in rows:
        by_u[r["uniq"]].append(r)
    rank = sorted(by_u, key=lambda u: (-flags[u][name]["ratio"], -flags[u][name]["n13_max"]))
    top_list = [pair_record(u, by_u[u], flags, refs, name) for u in rank[:top]]
    if name == "pool":
        flagged = [u for u in rank if is_copy(flags[u][name])]
    else:
        flagged = [u for u in rank if is_copy(flags[u][name]) or flags[u][name]["n13_max"] > 0
                   or flags[u][name]["ratio"] >= LIST_RATIO]
    return top_list, [pair_record(u, by_u[u], flags, refs, name) for u in flagged]


def leak_effect(rows, flags, refs):
    """For benchmark items copied into a curriculum: the student's score on them, then and after."""
    out = []
    trained = defaultdict(lambda: defaultdict(set))  # (run, model, task) -> bench id -> train steps
    for r in rows:
        if r["phase"] != "train":
            continue
        for name in HEADLINE_K:
            f = flags[r["uniq"]][name]
            if is_copy(f):
                trained[(r["run"], r["model"], name)][f["nearest_id"]].add(r["step"])
    for (run, model, task), ids in sorted(trained.items()):
        ref = refs[task]["ref_ids"]
        k = HEADLINE_K[task]
        first = min(min(s) for s in ids.values())

        def score(path):
            p = ROOT / path / "records.jsonl"
            if not p.exists():
                return None
            recs = {x["id"]: x for x in map(json.loads, p.open()) if "c" in x}
            val = lambda sel: (round(100 * float(np.mean([pass_at_k(recs[i]["n"], recs[i]["c"], k)
                                                          for i in sel])), 2), len(sel)) if sel else None
            held = [i for i in recs if i not in ref]
            return {"copied": val([i for i in ids if i in recs]),
                    "copied_held_out": val([i for i in ids if i in recs and i not in ref]),
                    "held_out_all": val(held),
                    "held_out_without_copied": val([i for i in held if i not in ids])}
        ck = {"base": score(f"outputs/benchmarks/{model}__{task}")}
        for s in FINAL_STEPS:
            ck[f"step{s}"] = score(f"outputs/frontier-model/{run}/final_eval/{model}__teacher__step{s}__{task}")
        arms = {"(base model)": ck["base"], "TEACHER": ck["step19"]}
        for dd in sorted((ROOT / "outputs/grpo").glob(f"{model}__pass1_*__step20__{task}")):
            arms[dd.name[len(model) + 2:].split("__step")[0]] = score(dd.relative_to(ROOT))
        out.append({"run": run, "model": model, "task": task, "metric": f"pass@{k}",
                    "copied_ids": {i: sorted(s) for i, s in sorted(ids.items())},
                    "in_ref_fifth": sorted(i for i in ids if i in ref),
                    "first_train_step": first,
                    "downstream_final_checkpoints": [s for s in FINAL_STEPS if s >= first],
                    "scores": ck,
                    "arms_at_10240_rollouts": arms})
    return out


# ---------------------------------------------------------------- --contamination
# Every arm's training data, scanned the same way, turned into one exclusion set per
# (model, benchmark) so all arms of a model can be re-scored on the same clean ids.
#
# Which file an arm trained on is not inferred from its name. train/grpo/run_grpo.sh
# resolves `data/further_improve/<cfg>/<cfg>__<band>__n*.jsonl` (TAG=__g32 changes
# only the experiment name) and hands verl the `.verl.jsonl` conversion. Each run's
# log (logs/grpo__<exp>.log, node-local, gitignored) prints that `train_files` path
# and verl's `dataset len`; the sha256 below is of the .verl.jsonl on the node that
# ran it, read 2026-09-14. The tool re-converts the tracked subset with
# to_verl_dataset.convert and requires the same bytes, so the scan is of exactly
# what verl read, not of a file with the same name.
CONTAMINATION_OUT = "outputs/analysis/benchmark_contamination.json"
ARM_RECORD = {  # experiment: (node holding the log, verl dataset len, sha256 of the .verl.jsonl)
    "llama32-3b__pass1_eq_0": ("lumen2", 800, "722839cc0cc188442df473a5d41adc0528d0bbaf5c66904b3bba299e74ae6b84"),
    "llama32-3b__pass1_05-15pct": ("lumen1", 400, "2b0e57634dffa0688bb4909fd311d67472a32c5170167cf644915f6f8e37ad51"),
    "llama32-3b__pass1_40-60pct": ("lumen1", 500, "5fffe842c3fd506d417f6680c9987e8552b7d7b698b83fe92424389396b8d4aa"),
    "llama32-3b__pass1_40-60pct__g32": ("lumen3", 500, "5fffe842c3fd506d417f6680c9987e8552b7d7b698b83fe92424389396b8d4aa"),
    "llama32-3b__pass1_85-95pct": ("lumen1", 400, "e96999f83b0651da36715abfe480806f9372ae1ee2063027c2c1e4519a4baba7"),
    "llama32-3b__pass1_eq_1": ("lumen2", 100, "4676b41ec0330d7611b7e38963264a3fb52bc238ff2de1a697267f7cb1b44dd9"),
    "qwen3-4b-nothink__pass1_eq_0": ("lumen2", 698, "46a1a8e601e3da14715b946b0cbd298f5f5c0440626741aabf5727bb86d5039b"),
    "qwen3-4b-nothink__pass1_12-25pct": ("lumen2", 600, "fbd6462699cb9a4ee49c894832ef1bbc0e63300075997fc8359bdc70a452883d"),
    "qwen3-4b-nothink__pass1_37-62pct": ("lumen2", 1000, "db294ca97d9f438ef1a554f4f1c0466e469fc6a6b8d3d7694e254c4852941e4c"),
    "qwen3-4b-nothink__pass1_37-62pct__g32": ("lumen2", 1000, "db294ca97d9f438ef1a554f4f1c0466e469fc6a6b8d3d7694e254c4852941e4c"),
    "qwen3-4b-nothink__pass1_75-87pct": ("lumen2", 1000, "bdd0dcbbb1e743ad0517472b49f42ecf2f7bc672aa4dcf43e143b131cab85842"),
    "qwen3-4b-nothink__pass1_eq_1": ("lumen2", 1000, "e76a4a8d9d381cef9137301d679b5a7192e3e0e21bf56f210ea63906b58a35b8"),
    "qwen3-4b-think__pass1_eq_0": ("lumen1", 199, "e1a55a999571c7bc76a581cd3fe197400dfb9eb58c9d631e336fa52fb4346a45"),
    "qwen3-4b-think__pass1_12-25pct": ("lumen3", 100, "c9337cb6964edcef083ea76d3f279c8d87f45e88d93e7efca70b5acf896159a2"),
    "qwen3-4b-think__pass1_37-62pct": ("lumen2", 200, "9789d60bb49e086539588da6d5ca53dedfafd30d249d4489ec0c3a0376e55e89"),
    "qwen3-4b-think__pass1_37-62pct__g32": ("lumen3", 200, "9789d60bb49e086539588da6d5ca53dedfafd30d249d4489ec0c3a0376e55e89"),
    "qwen3-4b-think__pass1_75-87pct": ("lumen1", 400, "b395c20533e427d7dfab416a9adc4b53c45f924664cfb5100687d2a9bff3d781"),
    "qwen3-4b-think__pass1_eq_1": ("lumen2", 500, "bdfd5e50ea1d55aeb233cd487797f436487d0f944f0c7579344e53c0432ac040"),
}
BENCHMARKS = ("math500", "aime", "hmmt")
# A pair is put in front of a reader when any of these hold; exact copies need no reader.
REVIEW = {"ratio": 0.8, "ratio_if_answer_matches": 0.6}
# Read by hand. Key: (source, benchmark id), where source is the pool id for a band
# subset row and run/step/phase/id for a teacher row. Only "paraphrase" enters an
# exclusion set; "shared-setup" enters only the strict set; "same-form" is a reviewed exercise of
# the same shape with a new problem and enters neither.
JUDGED = {
    ("run_20260902_072619/step_12/train/s12-08", "math500-0215"): (
        "paraphrase", "'If x^2-x-1=0, find the value of x^3-2x+1' = 0215 reworded, same answer 2"),
    ("run_20260902_072619/step_13/train/s13_08", "math500-0215"): (
        "paraphrase", "same statement as s12-08, trained again at step 13"),
    ("run_20260902_072619/step_12/train/s12-09", "math500-0186"): (
        "paraphrase", "'For how many integers n>1 is 2^24 a perfect nth power' = 0186 minus 'positive', answer 7"),
    ("run_20260902_072856/step_5/train/h7", "math500-0118"): (
        "paraphrase", "'ordered pairs of integers (x,y) with x^2+y^2 <= 25' is 0118's |a+bi| <= 5 restated; "
                      "same lattice-point count, answer 81. Borderline: same problem, not same wording"),
    ("mathtrain-02027", "aime-2020-28"): (
        "paraphrase", "the MATH pool's copy of AIME 2020 II #7 (two cones, sphere): word-level typo differences, "
                      "answer 298"),
    ("mathtrain-01903", "math500-0386"): ("paraphrase", "'How many integers are there in the solution set of "
                                                        "|x-2| <= 5.6?' = 0386, answer 11"),
    ("mathtrain-05768", "math500-0339"): ("paraphrase", "parabola reflected about y=k; 'in terms of k, what is' vs "
                                                        "'express ... in terms of k', answer 2k"),
    ("mathtrain-03207", "math500-0237"): ("paraphrase", "xy=24, xz=48, yz=72, find x+y+z; answer 22"),
    ("mathtrain-00215", "math500-0378"): ("paraphrase", "two-digit primes with digit sum 8; answer 3"),
    # same setup, different question: not the benchmark problem, but a model trained on the
    # setup has seen half of it. Listed so a stricter re-score can drop them too.
    ("mathtrain-08491", "math500-0189"): ("shared-setup", "hot-air balloon over O, ropes HC=150, HD=130: "
                                                          "height of balloon vs rope saved"),
    ("mathtrain-06254", "math500-0301"): ("shared-setup", "AMC8 2002 #8-10 stamp table, a different part"),
    ("mathtrain-03552", "math500-0301"): ("shared-setup", "AMC8 2002 #8-10 stamp table, a different part"),
    ("mathtrain-10816", "math500-0298"): ("shared-setup", "same three cross products given, different expression"),
    ("mathtrain-08545", "math500-0152"): ("shared-setup", "same centroid/parallel-line figure, different area "
                                                          "given and asked"),
    ("mathtrain-00922", "math500-0320"): ("shared-setup", "same diagram and sin RPQ = 7/25; sin RPS vs cos RPS"),
    ("mathtrain-09897", "math500-0177"): ("shared-setup", "same four-circles diagram; perimeter vs smallest angle"),
    ("mathtrain-07301", "math500-0229"): ("shared-setup", "Yann and Camille, 10 dishes; repeats allowed vs not"),
}
# Every other pair the review rule lists was read on 2026-09-14 and judged same-form. That
# verdict is only valid for the list that was read: if the listed pairs change (new data,
# new thresholds), they come back UNREVIEWED and the run says so.
REVIEWED_SET_SHA256 = "33daeca87fdd4c59c99431dae90ca3e987ba6c19ce8843b1438738e01c7afe04"


def band_arms():
    """[(experiment, model, subset path)] for every section 7 arm that has results."""
    exps = sorted({d.name.split("__step")[0] for d in (ROOT / "outputs/grpo").glob("*__pass1_*__step20__*")})
    out = []
    for exp in exps:
        model, band = exp.split("__", 1)
        band = band.removesuffix("__g32")
        subs = sorted((ROOT / "data/further_improve" / model).glob(f"{model}__{band}__n*.jsonl"))
        subs = [s for s in subs if not s.name.endswith(".verl.jsonl")]
        assert subs, f"no subset for {exp}"
        assert exp in ARM_RECORD, f"{exp} has results but no recorded training log"
        out.append((exp, model, subs[0]))
    missing = set(ARM_RECORD) - {e for e, _, _ in out}
    assert not missing, f"recorded arms without results: {missing}"
    return out


def verl_bytes(rows):
    sys.path.insert(0, str(ROOT / "train/grpo"))
    from to_verl_dataset import convert
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in convert(rows)).encode()


def contamination_main(a):
    import hashlib
    t0 = time.time()
    refs = load_refs()
    arms, rows = {}, []
    for exp, model, sub in band_arms():
        raw = [json.loads(l) for l in sub.open()]
        kept = [(i, r) for i, r in enumerate(raw) if str(r.get("answer", "")).strip()]  # as to_verl_dataset
        sha = hashlib.sha256(verl_bytes([r for _, r in kept])).hexdigest()
        node, dlen, rec = ARM_RECORD[exp]
        g = 8 if re.search(r"37-62pct|40-60pct", exp) and not exp.endswith("__g32") else 32
        tb = 512 // g
        arms[exp] = {"model": model, "kind": "band", "subset": str(sub.relative_to(ROOT)),
                     "verl_file_sha256": sha, "log": f"{node}:~/code/frontier-teacher/logs/grpo__{exp}.log",
                     "verified_against_node_copy": sha == rec and len(kept) == dlen,
                     "problems": len(kept), "group_size": g, "problem_instances_20_steps": 20 * tb,
                     "note": "verl data.seed is null (unseeded shuffle): which rows a 20-step run drew "
                             "cannot be reconstructed, so every row of the file counts as trained on"}
        assert arms[exp]["verified_against_node_copy"], f"{exp}: tracked subset != what verl read"
        for i, r in kept:
            rows.append({"model": model, "arm": exp, "file": str(sub.relative_to(ROOT)), "line": i,
                         "source": r["id"], "id": r["id"], "problem": r["problem"], "answer": str(r["answer"])})
    for run in RUNS:
        rd = ROOT / "outputs/frontier-model" / run
        model = json.loads((rd / "step_0/config.json").read_text())["student"]["grpo_preset"]
        exp = f"{model}__teacher__{run}"
        n = 0
        for f in sorted(rd.glob("step_*/train/data.jsonl")):
            step = f.parent.parent.name
            for i, line in enumerate(f.open()):
                r = json.loads(line)
                rows.append({"model": model, "arm": exp, "file": str(f.relative_to(ROOT)), "line": i,
                             "source": f"{run}/{step}/train/{r.get('id')}", "id": str(r.get("id")),
                             "step": int(step.split("_")[1]),
                             "problem": str(r.get("problem", "")), "answer": str(r.get("answer", ""))})
                n += 1
        arms[exp] = {"model": model, "kind": "teacher", "run": run, "problems": n,
                     "note": "training curricula only; evaluation probes never reach the student"}

    uniq, key = [], {}
    for r in rows:
        k = (r["problem"], r["answer"])
        if k not in key:
            key[k] = len(uniq)
            uniq.append({"problem": r["problem"], "answer": r["answer"]})
        r["uniq"] = key[k]
    print(f"{len(arms)} arms, {len(rows)} training rows, {len(uniq)} unique statements")
    G.update(uniq=uniq, refs={n: refs[n] for n in BENCHMARKS})
    flags = [None] * len(uniq)
    with mp.get_context("fork").Pool(a.procs) as pool:
        for i, (u, res) in enumerate(pool.imap_unordered(match_one, range(len(uniq)), chunksize=4)):
            flags[u] = res
            if i % 1000 == 0:
                print(f"  matched {i}/{len(uniq)}  {time.time() - t0:.0f}s", flush=True)

    # contamination hits and pairs for a reader
    hits, review = [], []
    for r in rows:
        for b in BENCHMARKS:
            f = flags[r["uniq"]][b]
            R = refs[b]
            cands = {f["nearest_id"]} | ({f["n13_id"]} if f["n13_id"] else set())
            for bid in sorted(cands):
                j = JUDGED.get((r["source"], bid))
                copy = is_copy(f) and bid == (f["exact_tex"] or f["exact_plain"] or f["nearest_id"])
                if copy:
                    kind = "exact"
                elif j and j[0] in ("paraphrase", "shared-setup"):
                    kind = j[0]
                else:
                    kind = None
                if kind:
                    hits.append({"model": r["model"], "arm": r["arm"], "benchmark": b, "id": bid,
                                 "in_ref_fifth": bid in R["ref_ids"], "kind": kind,
                                 "answer_match": f["answer_match"] if bid == f["nearest_id"] else None,
                                 "file": r["file"], "line": r["line"], "source": r["source"],
                                 "judgement": j[1] if j else None})
                flagged = bid == f["nearest_id"] and (
                    f["ratio"] >= REVIEW["ratio"] or f["contained"] or is_exact(f)
                    or (f["answer_match"] and f["ratio"] >= REVIEW["ratio_if_answer_matches"]))
                flagged = flagged or (bid == f["n13_id"] and f["n13_max"] > 0)
                if flagged and not copy:
                    bench = next(it for it in R["items"] if it["id"] == bid)
                    review.append({"model": r["model"], "arm": r["arm"], "benchmark": b, "id": bid,
                                   "in_ref_fifth": bid in R["ref_ids"], "source": r["source"],
                                   "file": r["file"], "line": r["line"],
                                   "ratio": f["ratio"] if bid == f["nearest_id"] else None,
                                   "n13_max": f["n13_max"] if bid == f["n13_id"] else 0,
                                   "answer_match": f["answer_match"] if bid == f["nearest_id"] else None,
                                   "verdict": j[0] if j else "UNREVIEWED", "note": j[1] if j else None,
                                   "train": {"problem": r["problem"], "answer": r["answer"]},
                                   "benchmark_item": {"problem": bench["problem"], "answer": bench.get("answer")}})

    # per-arm table
    for exp, arm in arms.items():
        mine = [h for h in hits if h["arm"] == exp]
        arm["contamination"] = {b: sorted({(h["id"], h["kind"], h["in_ref_fifth"], h["answer_match"])
                                           for h in mine if h["benchmark"] == b}) for b in BENCHMARKS}
        arm["contamination"] = {b: [dict(zip(("id", "kind", "in_ref_fifth", "answer_match"), t)) for t in v]
                                for b, v in arm["contamination"].items()}
    # exclusion sets
    exclusion = {}
    for model in sorted({r["model"] for r in rows}):
        exclusion[model] = {}
        for b in BENCHMARKS:
            src = defaultdict(list)
            for h in hits:
                if h["model"] == model and h["benchmark"] == b:
                    src[h["id"]].append({k: h[k] for k in ("arm", "file", "line", "source", "kind",
                                                            "answer_match", "judgement")})
            ref = refs[b]["ref_ids"]
            ids = sorted(i for i, v in src.items() if any(x["kind"] in ("exact", "paraphrase") for x in v))
            strict = sorted(src)
            exclusion[model][b] = {
                "ids": ids, "n": len(ids),
                "held_out": [i for i in ids if i not in ref],
                "reference_fifth": [i for i in ids if i in ref],
                "clean_held_out_n": sum(it["id"] not in ref and it["id"] not in ids for it in refs[b]["items"]),
                "clean_reference_fifth_n": sum(i not in ids for i in ref),
                "strict_ids": strict, "strict_n": len(strict),
                "strict_held_out": [i for i in strict if i not in ref],
                "sources": dict(sorted(src.items()))}

    # where the pool's own benchmark items sit, and which subsets picked them up
    subsets = {}
    for s in sorted((ROOT / "data/further_improve").glob("*/*__n*.jsonl")):
        if not s.name.endswith(".verl.jsonl"):
            subsets[str(s.relative_to(ROOT))] = {json.loads(l)["id"] for l in s.open()}
    pool_in_bench = []
    for it in refs["pool"]["items"]:
        tt, tp = norm_tex(it["problem"]), norm(it["problem"])
        for b in BENCHMARKS:
            bid = refs[b]["by_tex"].get(tt) or refs[b]["by_plain"].get(tp)
            if bid:
                bench = next(x for x in refs[b]["items"] if x["id"] == bid)
                pool_in_bench.append({
                    "pool_id": it["id"], "split_origin": it.get("split_origin"), "subject": it.get("subject"),
                    "level": it.get("level"), "benchmark": b, "id": bid, "in_ref_fifth": bid in refs[b]["ref_ids"],
                    "answer_match": answers_agree(it["answer"], bench.get("answer", ""), b == "aime"),
                    "note": None if answers_agree(it["answer"], bench.get("answer", ""), b == "aime") else
                    "normalised-equal only: the normaliser drops signs; a different problem",
                    "in_subsets": [f for f, s in subsets.items() if it["id"] in s]})

    review_sha = hashlib.sha256("\n".join(sorted({f"{x['source']}|{x['id']}" for x in review})).encode()).hexdigest()
    if review_sha == REVIEWED_SET_SHA256:
        for x in review:
            if x["verdict"] == "UNREVIEWED":
                x["verdict"], x["note"] = "same-form", "read 2026-09-14"
    unreviewed = [x for x in review if x["verdict"] == "UNREVIEWED"]
    doc = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": {
            "training_files": "section 7 arms: the subset run_grpo.sh resolves for the experiment, converted with "
                              "train/grpo/to_verl_dataset.convert and required to equal, byte for byte, the "
                              ".verl.jsonl named by train_files in the run's log (sha256 in arms[*]). Teacher arms: "
                              "step_*/train/data.jsonl of the three open-ended runs; eval probes excluded",
            "benchmarks": {b: refs[b]["files"] for b in BENCHMARKS},
            "reference_fifth": {b: f"data/benchmark/{REF_MANIFEST[b]}.manifest.json" for b in BENCHMARKS},
            "exact": "normalised statement equality (goal_audit.py normaliser, and again with [asy] and "
                     "typesetting-only LaTeX dropped), or one statement (>= 40 chars) contained in the other; "
                     "either with matching answers. Equal statements with different answers go to review",
            "similarity": "exhaustive difflib ratio against every benchmark item, plus shared 13-word spans",
            "review_rule": f"a non-exact pair is read by hand when ratio >= {REVIEW['ratio']}, or ratio >= "
                           f"{REVIEW['ratio_if_answer_matches']} with matching answers, or containment, or any "
                           "shared 13-gram",
            "judgement_rule": "paraphrase = the same problem (same objects, same numbers, same question) reworded, "
                              "normally with the same answer; it enters the exclusion set. shared-setup = the "
                              "same setup or figure with a different question; strict set only. same-form = a new "
                              "problem of the same shape (numbers changed); neither. Most 13-gram-only hits are "
                              "answer-format boilerplate ('m and n are relatively prime positive integers')",
            "exclusion": "ids: union over every arm of a model, bands and teacher alike, of exact copies and "
                         "judged paraphrases. strict_ids adds judged shared-setup items. The whole subset file "
                         "counts because verl's shuffle was unseeded",
            "reviewed_set_sha256": review_sha,
            "unreviewed_pairs": len(unreviewed)},
        "arms": arms,
        "exclusion": exclusion,
        "pool_items_equal_to_benchmark_items": pool_in_bench,
        "reviewed_pairs": review,
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out)

    print(f"\n{'arm':42s}" + "".join(f"{b:>22s}" for b in BENCHMARKS))
    for exp, arm in arms.items():
        cells = []
        for b in BENCHMARKS:
            c = arm["contamination"][b]
            cells.append(f"{len(c)} ({sum(not x['in_ref_fifth'] for x in c)} held-out)" if c else "-")
        print(f"{exp:42s}" + "".join(f"{x:>22s}" for x in cells))
    print("\nexclusion sets:")
    for model, per in exclusion.items():
        print(f"  {model:18s} " + "  ".join(f"{b} {v['n']} ({len(v['held_out'])} held-out; strict {v['strict_n']}"
                                           f"/{len(v['strict_held_out'])})" for b, v in per.items()))
    print(f"\npool items equal to a benchmark item: {len(pool_in_bench)}")
    print(f"pairs for review: {len(review)}, unreviewed: {len(unreviewed)}  (set sha256 {review_sha})")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/analysis/teacher_overlap.json")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--check", type=int, default=0,
                    help="re-run N random statements without pruning and require identical ratios")
    ap.add_argument("--contamination", action="store_true",
                    help="scan every arm's TRAINING data (the section 7 band subsets and the teacher "
                         "runs' curricula) and write per-model benchmark exclusion sets")
    a = ap.parse_args()
    if a.contamination:
        if a.out == ap.get_default("out"):
            a.out = CONTAMINATION_OUT
        return contamination_main(a)
    t0 = time.time()

    rows, scratch, superseded, provenance = load_teacher()
    refs = load_refs()
    counts = defaultdict(lambda: {"train": 0, "eval": 0})
    for r in rows:
        counts[r["model"]]["train" if r["phase"] == "train" else "eval"] += 1
    print(f"teacher rows: {len(rows)}   scratch rows: {len(scratch)}")
    for m, c in counts.items():
        print(f"  {m:18s} train={c['train']} eval={c['eval']} total={c['train'] + c['eval']}")
    for s in superseded:
        print(f"  {s['run']} superseded {s['dirs']}: {s['files']} files, {len(s['problem_files'])} problem files")

    uniq, key = [], {}
    for r in rows + scratch:
        k = (r["problem"], r["answer"])
        if k not in key:
            key[k] = len(uniq)
            uniq.append({"problem": r["problem"], "answer": r["answer"]})
        r["uniq"] = key[k]
    print(f"unique (problem, answer): {len(uniq)};  " +
          ", ".join(f"{n} {len(R['items'])} ({len(R['ref_ids'])} ref)" for n, R in refs.items()))

    G.update(uniq=uniq, refs=refs)
    flags = [None] * len(uniq)
    with mp.get_context("fork").Pool(a.procs) as pool:
        for i, (u, res) in enumerate(pool.imap_unordered(match_one, range(len(uniq)), chunksize=4)):
            flags[u] = res
            if i % 500 == 0:
                print(f"  matched {i}/{len(uniq)}  {time.time() - t0:.0f}s", flush=True)
        check = None
        if a.check:
            sample = np.random.default_rng(0).choice(len(uniq), min(a.check, len(uniq)), replace=False)
            res = pool.map(check_one, sample.tolist())
            bad = [(u, n, p, q) for u, d in res for n, (p, q) in d.items() if abs(p - q) > 1e-12]
            check = {"sampled": len(sample), "sets": ["math500", "aime", "hmmt"], "mismatches": len(bad)}
            print(f"pruning check: {check}")
            assert not bad, bad[:5]

    scope = lambda r: ["all", "train" if r["phase"] == "train" else "eval_probe", f"{r['model']}/all",
                       f"{r['model']}/" + ("train" if r["phase"] == "train" else "eval_probe")]
    summary = summarise(rows, flags, refs, scope)
    listed = {n: dict(zip(("top20", "flagged"), pairs(rows, flags, refs, n))) for n in refs}
    effect = leak_effect(rows, flags, refs)
    id_claims = [{k: r[k] for k in ("model", "run", "step", "phase", "id")} for r in rows
                 if SOURCE_ID.search(r["id"])]

    doc = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": {"normalise": "lowercase; non-alphanumerics -> space; collapse (goal_audit.py). "
                                "_tex variants also drop \\commands and [asy] blocks, both sides",
                   "ngram": N_GRAM,
                   "nearest": "exact argmax over all pairs of difflib.SequenceMatcher(None, bench_tex, "
                              "teacher_tex, autojunk=False).ratio(), pruned by quick_ratio upper bounds",
                   "copy": "(exact_plain or exact_tex or contained -- the shorter normalised statement, "
                           ">= 40 chars, a substring of the other) and matching answers",
                   "answer_match": "normalised string equality, else eval/verifiers/extract.grade "
                                   "(integer compare for AIME)",
                   "pruning_check": check},
        "benchmarks": {k: {"files": v["files"], "n": len(v["items"]),
                           "reference_fifth_ids": sorted(v["ref_ids"])} for k, v in refs.items()},
        "counts": {"rows": len(rows), "per_model": counts, "unique_statements_incl_scratch": len(uniq),
                   "scratch_rows": len(scratch), "scratch_files": sorted({r["file"] for r in scratch}),
                   "superseded": superseded},
        "summary": summary,
        "scratch_summary": summarise(scratch, flags, refs, lambda r: ["scratch"]) if scratch else {},
        "leak_effect": effect,
        "provenance": provenance,
        "ids_naming_a_source": id_claims,
        "pairs": listed,
        "items": [{**{k: r[k] for k in ("model", "run", "step", "phase", "file", "line", "id",
                                         "problem", "answer")},
                   "scratch": not r["file"].endswith("/data.jsonl"), "match": flags[r["uniq"]]}
                  for r in rows + scratch],
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out)

    print(f"\n{'set/scope':22s}{'rows':>6s}{'copy':>6s}{'ref':>5s}{'13g':>5s}{'max13':>6s}"
          + "".join(f"{'>=' + str(t):>7s}" for t in THRESH))
    for name in refs:
        for sc in ("all", "train", "eval_probe"):
            d = summary[name][sc]
            print(f"{name + '/' + sc:22s}{d['rows']:>6d}{d['copy_rows']:>6d}"
                  f"{d.get('copy_rows_ref_fifth', 0):>5d}{d['n13_rows']:>5d}{d['n13_max']:>6d}"
                  + "".join(f"{d[f'ratio_ge_{t}']['rows']:>7d}" for t in THRESH))
    print(f"\nprovenance: {len(provenance)} teacher turns touched " +
          ", ".join(f"{c} {sum(c in p['calls'] for p in provenance)}" for c in PROVENANCE))
    print(f"rows whose id names a source: {len(id_claims)}")
    for e in effect:
        print(f"leak: {e['model']} {e['task']} {len(e['copied_ids'])} copied into training "
              f"(first step {e['first_train_step']}, {len(e['in_ref_fifth'])} in ref fifth)")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
