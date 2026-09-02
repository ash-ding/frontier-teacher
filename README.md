# Frontier Teacher

Testing whether frontier models can serve as teachers to better train smaller
models on mathematics.

Everything currently in this repository is the **control**: plain GRPO, no
teacher, which any teacher-side method has to beat. `train/frontier_model/` is
where the teacher pipeline starts.

**Results:** [Which Problems Teach](https://claude.ai/code/artifact/96d42a23-04d8-488e-9e70-a764cc971780) —
fifteen GRPO runs plus three matched controls across three models and five
difficulty bands. `docs/experiment.md` is the full record, including the
failures worth knowing about; `docs/plan.md` is what is left.

---

## Setup

One environment for training and evaluation. Splitting them means training and
evaluation can grade differently, and then a benchmark gain cannot be attributed
to the model.

```bash
conda env create -f environment.yml     # python 3.12 only; everything else is pip
conda activate frontier-teacher
pip install -r requirements.lock.txt    # exact resolved set, from a verified node
```

vLLM 0.27.1, verl 0.9.0, flash-attn 2.8.3, `math-verify[antlr4_9_3]`. The antlr4
extra matters: verl's hydra pins antlr4 to 4.9.\*, and `math-verify` defaults to
13.2, so the default extra produces an environment that imports but grades
nothing.

verl's source is pinned, not vendored:

```bash
scripts/fetch_verl_source.sh            # fetch the pinned commit into package/
scripts/fetch_verl_source.sh --verify   # check it against the installed package
```

Run `--verify` after any environment change. It compares all 362 files by hash
and fails if what is installed is not what `package/verl.lock` claims.

---

## Layout

```
configs/
  eval/         One YAML per (model, benchmark) pair - 21 of them. Each fully
                describes one evaluation and doubles as a template to copy.
  grpo/         Training-side configs.
data/
  build_*.py    Dataset builders. Every one asserts on its output; re-running
                them reproduces the committed files byte for byte.
  benchmark/    MATH-500, AIME 2020-2024, HMMT (Feb 2025 / Nov 2025 / Feb 2026).
  training_set/ The 11,996-problem MATH pool, split by original provenance.
  further_improve/  Subsets cut by measured pass@1, with a manifest recording
                the decoding settings each was measured under.
eval/           Everything that measures.
  evaluate.py   Generate with vLLM, grade, persist. One run, one benchmark.
  metrics.py    pass@k for every k the sample count supports, plus the
                truncation and no-answer counters that say whether a score is a
                model result or a harness artefact.
  verifiers/    Two, named by the config and never inferred: exact_integer
                (AIME - answers are integers 0-999 by the competition's rules)
                and symbolic (everything else). extract.py holds the \boxed{}
                extraction and </think> split they share.
  merge_shards.py, calibrate.py, check_baselines.py
  run_all.sh, run_checkpoints.sh, run_sharded.sh
train/
  grpo/         The no-teacher baseline. verl_reward.py wraps eval/verifiers, so
                training and evaluation cannot grade differently.
  frontier_model/  The teacher-in-the-loop pipeline.
tools/          Off the critical path: subset sampling, difficulty reports,
                curve collection, paired statistics, the report renderer.
scripts/        setup_env.sh, fetch_verl_source.sh, ckpt_janitor.sh.
package/        verl.lock pins the exact verl commit; the source is fetched on
                demand and gitignored.
outputs/        A symlink to ~/data/frontier-teacher/outputs on the shared
                bucket, so all three nodes read and write one set of results.
                A run's summary, records and raw generations land together.
  benchmarks/     Baselines - the rollout-0 point of every curve.
  grpo/           Checkpoint evaluations - the rest of every curve.
  math_profiling/ Per-problem results over the training pool: the difficulty
                  measurement the subsets are cut from.
  analysis/       curves.json, paired_stats.json. Both regenerable.
.local_checkpoints/  Node-local weights, gitignored. 26 GB per run, written and
                deleted within one node's evaluation.
```

---

## Running things

### Evaluation

A config and the command line are two ways to describe one run. For each field
the command line wins, then the config, then the script's default — and the run
prints where every value came from, so the rule never has to be remembered.

```bash
# fully described by its config
python eval/evaluate.py --config configs/eval/llama32-3b__math500.yaml

# a checkpoint, with everything else from the model's benchmark config. The
# output tag follows the weights, so this cannot overwrite the baseline.
python eval/evaluate.py --config configs/eval/llama32-3b__aime.yaml \
  --model .local_checkpoints/<exp>/global_step_20/actor/huggingface

# no config: data written a moment ago, at any path. Nothing to register.
python eval/evaluate.py --model Qwen/Qwen3-4B --model-label qwen3-4b-think \
  --data /abs/path/step7/data.jsonl --label teacher_step7 \
  --verifier symbolic --samples 4 --thinking true
```

Three things are not optional: generations are always saved, every pass@k the
sample count supports is reported, and the verifier is named rather than
inferred.

Across a run's checkpoints:

```bash
eval/run_checkpoints.sh <config-name> <band> 8   # one job per GPU
eval/run_sharded.sh     <config-name> <band> 8   # one job across all GPUs
```

Use the sharded form for thinking models. A single HMMT job there is 5.5 hours
on one card — longer than any sane timeout — while a wave of eight runs only as
fast as its slowest member.

### GRPO baseline

```bash
train/grpo/run_grpo.sh <config-name> <band> [n_gpus]   # train only
train/grpo/run_one.sh  <config-name> <band>            # train, then evaluate
```

Bands are the subsets in `data/further_improve/<model>/`. Group size follows the
band's measured pass@1: a group whose rollouts all fail contributes no gradient,
which is 48% of groups at pass@1 8.7% with G=8 but 5% with G=32. Batch sizes are
set against G to hold 512 rollouts per step, so every curve shares an x-axis.

`GROUP_SIZE=32` overrides the band's group size; `TAG=__g32` suffixes the
experiment name so a re-run lands beside the original.

### Frontier teacher

```bash
python train/frontier_model/teacher/orchestrator.py \
    --config configs/grpo/llama32-3b-teacher.yaml \
    --output-path outputs/frontier-model \
    --reference-data /abs/path/to/some.jsonl        # optional
python train/frontier_model/teacher/orchestrator.py \
    --config configs/grpo/llama32-3b-teacher.yaml --dry-run   # no-GPU self-test
```

One step is one GRPO update. Within a step the teacher may evaluate the current
checkpoint any number of times; the step closes only when it chooses to train,
and reaching `max_evals_per_step` without training halts the run rather than
fabricating one. Everything lands in `<output-path>/run_<timestamp>/`.

Settings resolve the same way evaluation's do — command line, then the config
file, then the script's default — so `--steps`, `--max-evals-per-step`,
`--output-path` and `--reference-data` each override the config field of the
same name. `--reference-data` takes one absolute file path and copies it into
the run directory, so what the teacher read is part of the run's record; omit it
and the teacher gets no reference data at all.

Each step evaluates with `eval/evaluate.py` unmodified — a plain command line
against the data the teacher just wrote, with the sub-action directory as
`--output-path`. `run_teacher_step.sh` is the single-GRPO-update executor the
loop calls; it is not run by hand.

### Analysis

```bash
python tools/collect_curves.py      # summaries   -> outputs/analysis/curves.json
python tools/paired_stats.py        # bootstrapped intervals, problems + generations
python tools/render_report.py --out report.html      # 3x3 figure, no plotting deps
```

### Checks worth running

```bash
python eval/calibrate.py --data data/benchmark/hmmt_feb_2025.jsonl   # grading is sound
python eval/check_baselines.py                # baseline ids still match the harness
python tools/geomcheck.py report.html         # marks in bounds, labels not colliding
scripts/fetch_verl_source.sh --verify         # installed verl == the pin
```

Each exists because the failure it catches happened and was not obvious: a
grader that silently returns True, a baseline whose record ids drifted so the
paired analysis found an empty intersection, ten overlapping labels in a
published figure, an environment no longer matching its lock.

---

Datasets are committed. Pinning the exact problem set is what makes a score
comparable across runs and across time; the builders stay as the reproducibility
check.
