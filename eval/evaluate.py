"""Evaluate one model on one benchmark: generate, grade, persist everything.

Interfaces are explicit and overridable, because every one of them has needed to
vary: the data, the model, where results land, how many samples, how long a
response may be, whether the model reasons out loud, and which verifier decides
correctness. A config file supplies defaults; the command line overrides any of
them, so a checkpoint sweep does not need 36 near-identical config files.

Three things are deliberately not optional:

  Generations are always saved, beside the summary and records rather than in a
  tree of their own. `--save-generations` was a flag before and was forgotten on
  every checkpoint evaluation in the study, so 141 runs kept only verdicts; the
  checkpoints were then deleted, which made the traces unrecoverable.

  Every pass@k the sample count supports is reported. n samples produce the
  whole ladder for free, and one k hides the shape: a problem solved once in
  sixteen and one solved every time both score pass@16 = 1.

  The verifier is named, never inferred. Previously `integer_answer` defaulted
  to `args.task == "aime"` inside the loop, so "unset" meant different things
  for different tasks and only the source revealed which.

  python eval/evaluate.py --config configs/eval/llama32-3b.yaml --task math500
  python eval/evaluate.py --config ... --task aime --weights <ckpt-dir>
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

# Every field's fallback lives here, in the script. A config or the command line
# may set a field; when neither does, this is what applies. Nothing falls back
# through the config to reach a default, because a three-level chain is exactly
# the ambiguity that makes "why did it use 0.9?" hard to answer.
DEFAULTS = {
    "samples": 4,
    "max_tokens": 4096,
    "max_model_len": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
    "seed": 1234,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "verifier": "symbolic",       # AIME is the only benchmark that is not
    "enable_thinking": None,      # absent, not false: Llama's template rejects it
    "out_subdir": "",
    "headline_metric": None,
}

# Fields a config may set. Anything here that is ALSO given on the command line
# is an error rather than a silent precedence decision.
CONFIG_FIELDS = set(DEFAULTS) | {"model"}


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
    g.add_argument("--weights", default=None,
                   help="evaluate THESE weights using the config's profile - a "
                        "checkpoint directory. Distinct from the config's `model`, "
                        "which names the model the profile describes, so the two "
                        "never conflict. Omit to evaluate the config's own model.")
    g.add_argument("--name", default=None,
                   help="output tag. Derived from the model and the weights when "
                        "omitted, so a checkpoint can never overwrite a baseline.")

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


def resolve_name(args, cfg):
    """The output tag. Explicit --name wins; otherwise it follows the weights.

    `name` forms the filename, so two runs sharing one name overwrite each
    other. That is a real hazard rather than a theoretical one: evaluating a
    checkpoint with the base model's config and forgetting --name silently
    replaces that model's baseline - the rollout-0 anchor every curve and every
    paired comparison is measured against - with a trained checkpoint's score,
    and nothing warns you.

    So a run that points --weights somewhere gets a name derived from where:

        .local_checkpoints/llama32-3b__pass1_05-15pct/global_step_20/actor/huggingface
        -> llama32-3b__pass1_05-15pct__step20

    A run without --weights is evaluating the config's own model and keeps the
    config's name, which is what the baselines are called.
    """
    if args.name:
        return args.name
    if not args.weights:
        return cfg["name"]

    parts = Path(args.weights).resolve().parts
    step = next((p for p in reversed(parts) if p.startswith("global_step_")), None)
    if step:
        exp = parts[parts.index(step) - 1]
        return f"{exp}__step{step.rsplit('_', 1)[1]}"
    # Some other directory of weights: fall back to its own name, which at least
    # cannot collide with a baseline.
    return f"{cfg['name']}__{Path(args.weights).name}"


def resolve(args):
    """Merge script defaults, the config, and the command line into one dict.

    Config and command line are two ways of saying the same thing, not layers:
    a field set in both is an error, not a precedence decision. That rules out
    the class of surprise where a config value is quietly ignored because some
    orchestration script also passed the flag - which is how a run ends up
    sampling at a temperature nobody chose.

    The one pair that looks like a conflict and is not: the config's `model`
    names the model whose profile this is, and --weights points at weights to
    evaluate with that profile. Different questions, different fields.
    """
    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    if args.config and args.task not in cfg.get("tasks", {}):
        raise SystemExit(f"task {args.task!r} not in {args.config}; "
                         f"have {sorted(cfg.get('tasks', {}))}")
    task = cfg.get("tasks", {}).get(args.task, {})

    cli = {
        "samples": args.samples, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "verifier": args.verifier,
        "enable_thinking": None if args.thinking is None else args.thinking == "true",
    }
    cli = {k: v for k, v in cli.items() if v is not None}

    # A task's settings are part of the config, so a task-level `n` collides
    # with --samples exactly as a top-level one would.
    from_cfg = {k: v for k, v in {**cfg, **task}.items()
                if k in CONFIG_FIELDS and k != "model"}
    if task.get("n") is not None:
        from_cfg["samples"] = task["n"]

    clash = sorted(set(cli) & set(from_cfg))
    if clash:
        raise SystemExit(
            "these are set in both " + args.config + " and on the command line: "
            + ", ".join(clash) + ".\nPick one. Config and command line are two "
            "ways to specify a run, not a precedence chain.")

    R = {**DEFAULTS, **from_cfg, **cli}
    R["task"] = args.task
    R["model"] = args.weights or cfg.get("model")
    if not R["model"]:
        raise SystemExit("no model: give --weights, or set `model:` in the config.")
    R["name"] = resolve_name(args, cfg)
    R["profile_model"] = cfg.get("model")
    for k in ("samples", "max_tokens", "max_model_len", "top_k", "seed",
              "tensor_parallel_size"):
        R[k] = int(R[k])
    for k in ("temperature", "top_p", "gpu_memory_utilization"):
        R[k] = float(R[k])

    names = args.data or task.get("data_files") or ([task["data_file"]]
                                                    if task.get("data_file") else [])
    if not names:
        raise SystemExit(f"no dataset: give --data, or declare data_files under "
                         f"task {args.task!r} in the config.")
    R["data_files"] = [Path(n) if Path(n).is_absolute() else ROOT / "data" / n
                       for n in names]
    return R


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

    # Summary, per-problem records and raw generations all land in one place.
    # A separate generations/ tree meant three directories had to be kept in
    # step by hand, and the traces for a run were one directory away from the
    # numbers that summarise them.
    outdir = (Path(args.out) if args.out else ROOT / "outputs") / R["out_subdir"]
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    # Generations are written as they are produced, not buffered to the end: a
    # shard killed mid-run has cost a 12%-complete evaluation before.
    with (outdir / f"{tag}.generations.jsonl").open("w") as genf:
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
        "tag": tag, "task": R["task"], "model": R["model"], "profile": R["profile_model"], "config": R["name"],
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
    print(f"  -> {outdir}/{tag}.generations.jsonl")


if __name__ == "__main__":
    main()
