"""Evaluate one model on one benchmark: generate, grade, persist everything.

Interfaces are explicit and overridable, because every one of them has needed to
vary: the data, the model, where results land, how many samples, how long a
response may be, whether the model reasons out loud, and which verifier decides
correctness. A config file supplies defaults; the command line overrides any of
them, so a checkpoint sweep does not need 36 near-identical config files.

Three things are deliberately not optional:

  Generations are always saved. `--save-generations` was a flag before and was
  forgotten on every checkpoint evaluation in the study, so 141 runs kept only
  verdicts. The checkpoints were deleted afterwards, which made the traces
  unrecoverable. Whether to keep them is not a decision worth re-making.

  Every pass@k the sample count supports is reported. n samples produce the
  whole ladder for free, and one k hides the shape: a problem solved once in
  sixteen and one solved every time both score pass@16 = 1.

  The verifier is named, never inferred. Previously `integer_answer` defaulted
  to `args.task == "aime"` inside the loop, so "unset" meant different things
  for different tasks and only the source revealed which.

  python eval/evaluate.py --config configs/eval/llama32-3b.yaml --task math500
  python eval/evaluate.py --config ... --task aime --model <ckpt> --name <tag>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from metrics import summarize                      # noqa: E402
from verifiers import get_verifier                 # noqa: E402
from verifiers.extract import answer_segment       # noqa: E402

PROMPT = ("Solve the following math problem. Reason step by step, and put your "
          "final answer within \\boxed{{}}.\n\n{problem}")


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="configs/eval/<model>.yaml")
    ap.add_argument("--task", required=True, help="a key under `tasks:` in the config")

    g = ap.add_argument_group("paths - each overridable so one config serves many runs")
    g.add_argument("--data", nargs="*", default=None,
                   help="dataset JSONL(s), relative to data/ or absolute; "
                        "default: the task's own data_files")
    g.add_argument("--out", default=None, help="output directory (default: outputs/)")
    g.add_argument("--model", default=None, help="model id or checkpoint directory")
    g.add_argument("--name", default=None,
                   help="output tag; MUST differ per checkpoint or a run overwrites "
                        "the base model's results")

    g = ap.add_argument_group("sampling - model-specific, defaults from the config")
    g.add_argument("--samples", type=int, default=None,
                   help="samples per problem; pass@1..pass@N are all reported")
    g.add_argument("--max-tokens", type=int, default=None)
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--thinking", choices=["true", "false"], default=None,
                   help="Qwen3 only; Llama has no such mode and must not be given one")

    g = ap.add_argument_group("grading")
    g.add_argument("--verifier", choices=["exact_integer", "symbolic"], default=None,
                   help="default: the task's `verifier` key")

    g = ap.add_argument_group("execution")
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--num-shards", type=int, default=1)
    g.add_argument("--gpu-memory-utilization", type=float, default=None)
    g.add_argument("--limit", type=int, default=0, help="debug: first N problems only")
    return ap.parse_args()


def resolve(args):
    """Config supplies defaults; the command line wins. Returns one flat dict."""
    cfg = yaml.safe_load(open(args.config))
    if args.task not in cfg.get("tasks", {}):
        raise SystemExit(f"task {args.task!r} not in {args.config}; "
                         f"have {sorted(cfg.get('tasks', {}))}")
    task = cfg["tasks"][args.task]

    def pick(cli, *chain, required=True, default=None):
        for v in (cli, *chain):
            if v is not None:
                return v
        if required:
            raise SystemExit(f"missing setting for --{chain and 'see help' or ''}")
        return default

    r = {
        "model": pick(args.model, cfg.get("model")),
        "name": pick(args.name, cfg.get("name")),
        "task": args.task,
        "samples": int(pick(args.samples, task.get("n"))),
        "max_tokens": int(pick(args.max_tokens, task.get("max_tokens"), cfg.get("max_tokens"))),
        "max_model_len": int(pick(args.max_model_len, task.get("max_model_len"),
                                  cfg.get("max_model_len"))),
        "temperature": float(pick(args.temperature, cfg.get("temperature"))),
        "top_p": float(pick(args.top_p, cfg.get("top_p"))),
        "top_k": int(pick(args.top_k, cfg.get("top_k"), -1)),
        "seed": cfg.get("seed", 1234),
        "tensor_parallel_size": cfg.get("tensor_parallel_size", 1),
        "gpu_memory_utilization": float(pick(args.gpu_memory_utilization,
                                             cfg.get("gpu_memory_utilization"), 0.90)),
        "verifier": pick(args.verifier, task.get("verifier")),
        "out_subdir": task.get("out_subdir", ""),
        "headline_metric": task.get("headline_metric"),
    }

    # enable_thinking is Qwen3-only. Passing it to a template that does not take
    # it raises; omitting it for Qwen3 silently picks the template's default,
    # which is not the same as choosing. So it is tri-state: absent means the
    # model has no such mode.
    th = args.thinking if args.thinking is not None else cfg.get("enable_thinking")
    r["enable_thinking"] = None if th is None else (
        th if isinstance(th, bool) else th == "true")

    names = args.data or task.get("data_files") or [task.get("data_file", "")]
    if not names or not names[0]:
        raise SystemExit(f"task {args.task!r} declares no data_file(s) and --data was not given")
    r["data_files"] = [Path(n) if Path(n).is_absolute() else ROOT / "data" / n for n in names]
    return r


def main():
    args = build_args()
    R = resolve(args)

    for p in R["data_files"]:
        if not p.exists():
            raise SystemExit(f"missing dataset {p}; build it with the scripts in data/")
    rows = [json.loads(l) for p in R["data_files"] for l in open(p)]
    if args.limit:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        # round-robin, so every shard sees the same difficulty mix
        rows = rows[args.shard :: args.num_shards]
    if not rows:
        raise SystemExit("no problems selected")

    tag = f"{R['name']}__{R['task']}"
    if args.num_shards > 1:
        tag += f"__s{args.shard}of{args.num_shards}"
    verifier = get_verifier(R["verifier"])

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(R["model"])
    tkw = {} if R["enable_thinking"] is None else {"enable_thinking": R["enable_thinking"]}
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(problem=r["problem"])}],
        tokenize=False, add_generation_prompt=True, **tkw) for r in rows]

    llm = LLM(model=R["model"], tensor_parallel_size=R["tensor_parallel_size"],
              gpu_memory_utilization=R["gpu_memory_utilization"],
              max_model_len=R["max_model_len"], dtype="bfloat16", seed=R["seed"],
              enforce_eager=False, trust_remote_code=True)
    sp = SamplingParams(n=R["samples"], temperature=R["temperature"], top_p=R["top_p"],
                        top_k=R["top_k"], max_tokens=R["max_tokens"], seed=R["seed"])

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    out_root = Path(args.out) if args.out else ROOT / "outputs"
    outdir = out_root / R["out_subdir"]
    gendir = out_root / "generations"
    outdir.mkdir(parents=True, exist_ok=True)
    gendir.mkdir(parents=True, exist_ok=True)

    records = []
    # Generations are written as they are produced, not buffered to the end: a
    # shard killed mid-run has cost a 12%-complete evaluation before.
    with (gendir / f"{tag}.generations.jsonl").open("w") as genf:
        for row, out in zip(rows, outs):
            per_sample = []
            for si, comp in enumerate(out.outputs):
                v = verifier.grade(answer_segment(comp.text), row["answer"])
                s = {"correct": v.correct, "extracted": v.extracted,
                     "truncated": comp.finish_reason == "length",
                     "n_tokens": len(comp.token_ids)}
                per_sample.append(s)
                genf.write(json.dumps({"id": row["id"], "sample": si, **s,
                                       "text": comp.text}, ensure_ascii=False) + "\n")
            records.append({**row, "n": R["samples"],
                            "c": sum(s["correct"] for s in per_sample),
                            "samples": per_sample})

    summary = summarize(records, extra={
        "tag": tag, "task": R["task"], "model": R["model"], "config": R["name"],
        "verifier": R["verifier"], "gen_seconds": gen_s,
        "sampling": {k: R[k] for k in ("temperature", "top_p", "top_k",
                                       "max_tokens", "seed")} |
                    {"enable_thinking": R["enable_thinking"]},
        "shard": args.shard, "num_shards": args.num_shards,
        "data_files": [str(p) for p in R["data_files"]],
        "headline_metric": R["headline_metric"],
    })
    (outdir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    with (outdir / f"{tag}.records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    head = R["headline_metric"] or "pass@1"
    ks = [k for k in summary if k.startswith("pass@") and not k.endswith("_stderr")]
    print(f"{tag}  n={R['samples']}  verifier={R['verifier']}")
    print("  " + "  ".join(f"{k}={100*summary[k]:.1f}" for k in ks))
    print(f"  headline {head}={100*summary.get(head, float('nan')):.1f}"
          f"  truncated={100*summary['truncation_rate']:.1f}%"
          f"  no_answer={100*summary['no_answer_rate']:.1f}%")
    print(f"  -> {outdir}/{tag}.summary.json + .records.jsonl")
    print(f"  -> {gendir}/{tag}.generations.jsonl")


if __name__ == "__main__":
    main()
