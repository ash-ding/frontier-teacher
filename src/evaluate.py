"""Run one (model-config, task) evaluation with vLLM and score it.

MATH-500 -> pass@1   (mean accuracy over n samples, + stderr)
AIME      -> pass@4  (unbiased estimator from n samples, Chen et al. 2021)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from grading import grade  # noqa: E402
from summarize import summarize  # noqa: E402

PROMPT = "Solve the following math problem. Reason step by step, and put your final answer within \\boxed{{}}.\n\n{problem}"


def answer_segment(text: str) -> str:
    """For thinking models, grade only what follows </think>."""
    tag = "</think>"
    return text.split(tag, 1)[1] if tag in text else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, help="a key under `tasks:` in the config")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N problems")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--save-generations", default="", help="dir for full raw model output")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.task not in cfg["tasks"]:
        raise SystemExit(f"task {args.task!r} not in {args.config}; have {sorted(cfg['tasks'])}")
    task = cfg["tasks"][args.task]
    tag = f"{cfg['name']}__{args.task}"
    if args.num_shards > 1:
        tag += f"__s{args.shard}of{args.num_shards}"

    # a task reads one file (data_file) or several concatenated in order
    # (data_files) - the latter lets a multi-year benchmark stay split on disk
    names = task.get("data_files") or [task.get("data_file", f"benchmark/{args.task}.jsonl")]
    data_files = [ROOT / "data" / n for n in names]
    for p_ in data_files:
        if not p_.exists():
            raise SystemExit(f"missing dataset {p_}; run the builders in data/ first")
    rows = [json.loads(l) for p_ in data_files for l in open(p_)]
    if args.limit:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard :: args.num_shards]   # round-robin keeps shards difficulty-balanced
    integer_answer = task.get("integer_answer", args.task == "aime")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    kw = {}
    if cfg.get("enable_thinking") is not None:
        kw["enable_thinking"] = cfg["enable_thinking"]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(problem=r["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
            **kw,
        )
        for r in rows
    ]

    n = task["n"]
    sp = SamplingParams(
        n=n,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg.get("top_k", -1),
        max_tokens=task["max_tokens"],
        seed=cfg.get("seed", 1234),
    )

    llm = LLM(
        model=cfg["model"],
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
        max_model_len=task["max_model_len"],
        dtype="bfloat16",
        seed=cfg.get("seed", 1234),
        enforce_eager=False,
        trust_remote_code=True,
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    genf = None
    if args.save_generations:
        gdir = Path(args.save_generations)
        gdir.mkdir(parents=True, exist_ok=True)
        genf = (gdir / f"{tag}.generations.jsonl").open("w")

    records, n_trunc, n_noext, gen_tokens = [], 0, 0, 0
    for row, out in zip(rows, outs):
        per_sample = []
        for si, comp in enumerate(out.outputs):
            ntok = len(comp.token_ids)
            gen_tokens += ntok
            truncated = comp.finish_reason == "length"
            n_trunc += truncated
            ok, extracted = grade(answer_segment(comp.text), row["answer"], integer_answer)
            n_noext += extracted is None
            per_sample.append({"correct": bool(ok), "extracted": extracted,
                               "truncated": truncated, "n_tokens": ntok})
            if genf is not None:
                genf.write(json.dumps({"id": row["id"], "sample": si, "correct": bool(ok),
                                       "extracted": extracted, "truncated": truncated,
                                       "n_tokens": ntok, "text": comp.text},
                                      ensure_ascii=False) + "\n")
        c = sum(s["correct"] for s in per_sample)
        records.append({**row, "n": n, "c": c, "samples": per_sample})
    if genf is not None:
        genf.close()

    summary = summarize(records, cfg, args.task, task, extra={
        "mean_gen_tokens": gen_tokens / (len(records) * n),
        "gen_seconds": gen_s,
        "shard": args.shard, "num_shards": args.num_shards,
    })
    summary["tag"] = tag

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    with (outdir / f"{tag}.records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
