# Frontier Teacher

Testing whether frontier models can serve as teachers to better train smaller
models on mathematics. Everything currently in the repository is the **control**
— plain GRPO, no teacher — which any teacher-side method has to beat. No teacher
has been used yet; `docs/plan.md` §2 is where that starts.

`docs/experiment.md` is the experimental record: what ran, what it measured, what
the numbers mean, and the failures worth knowing about. Read it before proposing
an experiment — several obvious ones have already been run or ruled out.

## Layout

| Path | |
|---|---|
| `configs/*.yaml` | one per model configuration; `tasks:` defines every benchmark, its sample count and headline metric |
| `data/benchmark/` | MATH-500, AIME 2020–2024, HMMT (Feb 2025 / Nov 2025 / Feb 2026) |
| `data/training_set/` | the 11,996-problem MATH pool, split by original provenance |
| `data/further_improve/<model>/` | curated subsets by measured pass@1, with a `manifest.json` recording the decoding settings each was measured under |
| `src/evaluate.py` | the only evaluation path. `PROMPT` at the top is the single source of the prompt — training must use the same one |
| `src/grading.py` | `grade()`; the reward function wraps this so training and evaluation cannot disagree |
| `outputs/` | summaries and per-problem records are tracked in git; generations and profiling bulk are not |

## Environment

**One environment for everything** — `conda activate frontier-teacher` on
`lumen-1`, `lumen-2`, `lumen-3`. vLLM 0.27.1, verl 0.9.0, flash-attn 2.8.3,
`math-verify[antlr4_9_3]`. Do not create a second environment for RL: the
apparent antlr4 conflict with verl's hydra pinning dissolves under the
`antlr4_9_3` extra, and a split environment means training and evaluation can
grade differently.

Do not add matplotlib. Figures are hand-written SVG (`src/render_report.py`)
precisely so rendering never touches an environment that running jobs depend on.

## verl's source

The environment runs verl 0.9.0 from a PyPI wheel, which records no commit. It is
tag `v0.9.0`, commit `483b8a0` of `verl-project/verl` — verified rather than
assumed, all 362 `.py` files sha256-identical to the installed package.

`package/verl.lock` holds that pin and is tracked. The source is not: 8.3 MB
across 362 files would bury this repository's own history in `git log` and
`git grep`, and the pin reconstructs it exactly.

```bash
scripts/fetch_verl_source.sh            # fetch into package/ (gitignored), then verify
scripts/fetch_verl_source.sh --verify   # verify only
```

The fetched tree is a **read-only snapshot**. Python imports verl from
site-packages, so editing it changes nothing — the failure mode is patching it
and wondering why nothing happens. To modify verl: patch, reinstall, update the
pin, re-run `--verify`. That check is the point of the pin; run it after any
environment change.

## Running things

```bash
scripts/run_grpo.sh <config> <band> [n_gpus]     # one GRPO run
scripts/run_one.sh  <config> <band>              # train then evaluate
scripts/eval_checkpoints.sh <config> <band> 8    # Llama / non-thinking: one job per GPU
scripts/eval_sharded.sh     <config> <band> 8    # thinking: one job at a time, sharded across GPUs
python src/collect_curves.py && python src/paired_stats.py && python src/render_report.py
python src/check_baselines.py && python src/geomcheck.py outputs/report.html
```

`GROUP_SIZE=32` forces the group size; `TAG=__g32` suffixes the experiment name
so a re-run lands beside the original instead of overwriting it.

## Things that have bitten us

Each of these cost hours. They are in `docs/experiment.md` with the evidence.

- **Never `pkill -f` or `pgrep -f` a pattern that appears in your own command
  line.** It has killed the ssh session running it — twice — and orphaned a vLLM
  process holding 80 GB. Resolve to PIDs first, then kill by PID. A ray job has
  ~190 child processes; pattern matching over that tree is not controllable.
- **verl's `trainer.resume_mode` defaults to `auto`.** Re-running a band does not
  re-run it; it silently continues an aborted attempt from an unknown step.
  `run_grpo.sh` sets `disable` and wipes stale checkpoints. Any new entry point
  inherits the bad default.
- **A response cap is part of the reward, not a tuning knob.** A truncated
  rollout has no `\boxed{}` and scores 0. Training capped below what evaluation
  allows measures where the cap fell. Length also grows with difficulty, so a cap
  bites hardest on the hardest band — the one you most want to measure.
- **Baselines drift between nodes and fail silently.** A stale copy has record
  ids `aime-0000` where the harness now emits `aime-2020-00`. Scores match, so
  nothing looks wrong; the paired analysis just finds an empty intersection and
  drops the cell. Run `src/check_baselines.py` before trusting any comparison.
- **Pushing a fix is not delivering it.** Orchestrators `git checkout` once at
  startup and never re-sync. A converter fix reached the repo 40 minutes before
  the run it would have saved, and that run still died on the old code.
- **`git checkout origin/main -- <paths>` does not move HEAD.** It is the right
  way to sync a node mid-run — it touches only the paths you name, leaving
  `outputs/` alone — but the node's `git log` then reports a commit from before
  the sync, and any path you never named is simply absent. All three nodes ended
  up 24–33 commits behind with three different versions of one file. Bring a node
  forward with `git reset origin/main` (mixed) plus a checkout of the non-output
  paths; **never `git reset --hard`**, which reverts the 440 result files in
  `outputs/` — that is how a set of Llama baselines was lost once already.
- **Difficulty bands are measured at n=8 or n=32, and training samples G=32
  fresh each step.** "Never solved in 8 samples" is not "never solved": at a true
  rate of 4.7%, eight samples miss it 68% of the time. Measured, the p=0 bands
  train — reward 1.4% → 6.3% (Llama), 4.7% → 9.6% (non-thinking) — and about half
  their groups carry gradient. Any claim of the form "N% of the pool is
  untrainable" that rests on n=8 is an overestimate, including the 92.9% figure
  this repository quotes for thinking.
- **p=1 is not a null band either.** Llama gains +3.7 on MATH-500 training only
  on problems it already solved 32/32, and nothing out of domain. Training on
  what a model already knows buys consistency on that distribution, not
  capability — worth remembering before reading any in-domain gain as learning.
- **Sharded evaluation writes `{tag}__{task}__sNofM.summary.json` per shard**
  until it merges. Those match the checkpoint filename pattern; anything counting
  or collecting summaries must exclude them or a job in flight reads as finished.
- **verl's per-step metrics line is ~4 KB and stdout is block-buffered** into the
  log, so a young run can have completed steps with nothing written. tqdm goes to
  stderr unbuffered — use the progress bar as the fallback.
- **The nodes' clock runs ahead of the local one.** Node logs, checkpoint mtimes
  and progress-bar elapsed times are all server time; quote that, not local.
- **Disk.** A thinking checkpoint is 31 GB (15 GB HF export + 16 GB FSDP shards).
  A cron job strips the shards, which also removes the only way to resume — a
  deliberate trade, given `resume_mode=disable`. ZFS accounting lags deletions by
  minutes, so a `df` right after `rm` is not the truth.

## Reporting conventions

- MATH-500 headlines pass@1; AIME and HMMT headline pass@4 with pass@1 shown
  beside it. HMMT resolves nothing for a 3B model — 1.8% pass@4 is 1.7 problems
  of 93 — and is reported for completeness, not as evidence.
- **Intervals bootstrap over problems _and_ generations.** Re-running one
  identical evaluation of one identical model returned AIME pass@4 15.3% and
  12.0%; resampling problems alone understates intervals by about a third.
- A run that reached fewer than 20 steps does not have the 10,240-rollout
  endpoint the other cells report. Say so rather than plotting its last point as
  if it were comparable.
