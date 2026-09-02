"""Evaluate one model on one benchmark: generate, grade, persist everything.

Every setting has needed to vary at some point: the data, the model, how many
samples, how long a response may be, whether the model reasons out loud, which
verifier decides correctness, where results land. So all of them are settable
two ways, and for each field independently the order is:

    config, if it names the field -> command line -> the script's default.

The config wins, which is the opposite of the usual convention, so a run prints
which command-line values it ignored rather than letting one quietly do nothing.

Three things are deliberately not optional:

  Generations are always saved, beside the summary and records rather than in a
  tree of their own. `--save-generations` was a flag before and was forgotten on
  every checkpoint evaluation in the study, so 141 runs kept only verdicts; the
  checkpoints were then deleted, which made the traces unrecoverable.

  Every pass@k the sample count supports is reported. n samples produce the
  whole ladder for free, and one k hides the shape: a problem solved once in
  sixteen and one solved every time both score pass@16 = 1.

  The verifier is named, never inferred. `integer_answer` used to default to
  `args.task == "aime"` inside the loop, so "unset" meant different things for
  different benchmarks and only the source revealed which.

There is no --task. A benchmark is not a name to look up in a catalogue, it is a
data file plus how to sample and grade it - which is exactly what a config holds.
Removing the lookup is what lets a pipeline evaluate data it has just written:
--data takes any path, resolved against the working directory rather than data/,
so nothing has to be registered first.

  python eval/evaluate.py --config configs/eval/llama32-3b__aime.yaml \
                          --output-path outputs/benchmarks/llama32-3b__aime
  python eval/evaluate.py --config configs/eval/llama32-3b__aime.yaml \
                          --model <ckpt-dir> --output-path outputs/grpo/<exp>__step20__aime
  python eval/evaluate.py --model Qwen/Qwen3-4B --data /abs/step7/data.jsonl \
                          --verifier symbolic --samples 4 --output-path <step-dir>

Output is three files in --output-path: summary.json, records.jsonl,
generations.jsonl. One run, one directory. Whether two runs collide is decided
by the directory the caller names, not by a filename the script derives - the
latter meant a forgotten flag could overwrite a baseline with a checkpoint.
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
# may name a field; when neither does, this applies. There is no chain through
# the config to reach a default.
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
    "headline_metric": None,
}


PROMPT = ("Solve the following math problem. Reason step by step, and put your "
          "final answer within \\boxed{{}}.\n\n{problem}")


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None,
                    help="a config describing one evaluation; see configs/eval/. "
                         "Optional - every setting can be given on the command line.")

    g = ap.add_argument_group("paths - each overridable so one config serves many runs")
    g.add_argument("--data", nargs="*", default=None,
                   help="dataset JSONL(s). Any path: absolute, or relative to the "
                        "working directory. Nothing is resolved under data/, so a "
                        "pipeline evaluating data it just generated needs no "
                        "registration and no symlink.")
    g.add_argument("--output-path", default=None,
                   help="directory for this run's three files. One run, one "
                        "directory: naming is the caller's, so nothing can "
                        "collide by accident.")
    g.add_argument("--label", default=None,
                   help="what is being evaluated, e.g. aime. Recorded in the "
                        "summary; defaults to the first data file's stem.")
    g.add_argument("--model", default=None,
                   help="what to evaluate: a HuggingFace id, or a checkpoint "
                        "directory. A checkpoint is a model; there is no separate "
                        "flag for one.")

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
    """Settle every setting. For each field independently:

        command line  ->  config  ->  the script's default in DEFAULTS

    A config is a starting point, not a contract: pointing --model at a
    checkpoint and keeping everything else from the model's benchmark config is
    the normal way to evaluate one, and that only works if the command line
    wins.

    Precedence is a rule you would otherwise have to hold in your head, so every
    run prints each setting with the source it came from.

    `--out` is not a setting and does not take part; it says where to write.
    """
    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    cfg = {k: v for k, v in cfg.items() if v is not None}

    cli = {
        "samples": args.samples, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "verifier": args.verifier, "model": args.model, "label": args.label,
        "data": args.data,
        "enable_thinking": None if args.thinking is None else args.thinking == "true",
    }
    cli = {k: v for k, v in cli.items() if v is not None}

    R = {**DEFAULTS, **cfg, **cli}
    for req in ("model", "data"):
        if not R.get(req):
            raise SystemExit(
                f"{req!r} is not set. Give --{req} on the command line, or set "
                f"{req}: in a config. See configs/eval/ for a complete example.")

    R["data_files"] = [Path(p).expanduser().resolve()
                       for p in ([R["data"]] if isinstance(R["data"], str) else R["data"])]
    if not R.get("label"):
        R["label"] = R["data_files"][0].stem
    R["_sources"] = {k: ("command line" if k in cli else
                         "config" if k in cfg else "default")
                     for k in sorted(set(DEFAULTS) | set(cfg) | set(cli))}
    for k in ("samples", "max_tokens", "max_model_len", "top_k", "seed",
              "tensor_parallel_size"):
        R[k] = int(R[k])
    for k in ("temperature", "top_p", "gpu_memory_utilization"):
        R[k] = float(R[k])
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

    # One run, one directory, three fixed names. The caller chooses the
    # directory, which is the only thing that decides whether two runs collide -
    # a decision that used to be made by deriving a filename, where forgetting a
    # flag could overwrite a baseline with a checkpoint's score.
    outdir = Path(args.output_path) if args.output_path else ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "" if args.num_shards == 1 else f".s{args.shard}of{args.num_shards}"

    records = []
    # Generations are written as they are produced, not buffered to the end: a
    # shard killed mid-run has cost a 12%-complete evaluation before.
    with (outdir / f"generations{stem}.jsonl").open("w") as genf:
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
        "label": R["label"], "model": R["model"],
        "profile_model": R["model"],
        "verifier": R["verifier"], "gen_seconds": gen_s,
        "sampling": {k: R[k] for k in ("temperature", "top_p", "top_k",
                                       "max_tokens", "seed")} |
                    {"enable_thinking": R["enable_thinking"]},
        "shard": args.shard, "num_shards": args.num_shards,
        "data_files": [str(p) for p in R["data_files"]],
        "headline_metric": R["headline_metric"],
    })
    (outdir / f"summary{stem}.json").write_text(json.dumps(summary, indent=2))
    with (outdir / f"records{stem}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    head = R["headline_metric"] or "pass@1"
    ks = [k for k in summary if k.startswith("pass@") and not k.endswith("_stderr")]
    # Where every setting came from. Precedence is a rule you would otherwise
    # have to remember; printing the resolution means you do not.
    src = R.get("_sources", {})
    shown = ("model", "label", "samples", "verifier", "temperature", "top_p",
             "top_k", "max_tokens", "max_model_len", "enable_thinking")
    print(f"{outdir}  ({len(records)} problems)")
    for k in shown:
        if k in R:
            print(f"    {k:18} {str(R[k]):<44} [{src.get(k, 'default')}]")
    print("  " + "  ".join(f"{k}={100*summary[k]:.1f}" for k in ks))
    print(f"  headline {head}={100*summary.get(head, float('nan')):.1f}"
          f"  truncated={100*summary['truncation_rate']:.1f}%"
          f"  no_answer={100*summary['no_answer_rate']:.1f}%")
    print(f"  -> summary{stem}.json  records{stem}.jsonl  generations{stem}.jsonl")


if __name__ == "__main__":
    main()
