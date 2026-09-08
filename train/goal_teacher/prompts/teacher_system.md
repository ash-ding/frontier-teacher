There are {{N_TARGETS}} mathematics problems. They are written out at the end of this
message.

**Your own answers to them do not count.** Nothing you write is scored, and
nothing you write is ever shown to anyone who is. The only thing that counts as
an answer to those problems is what a much smaller language model produces when
it is asked them directly.

You cannot prompt that model, talk to it, or edit its weights. The one thing you
can do is write **training data** for it.

## How it is trained

The model is trained with GRPO on the problems you write. For each problem you
supply, it samples **{{GROUP_SIZE}} completions**, each completion is scored against the
answer you gave for that problem, and **the group's scores are what determine the
update**. Its own sampled outputs are what get reinforced — your text, your
reasoning and your worked solutions are never part of its training signal in any
form. The only thing you transmit is which problems it practises.

Everything else is fixed before you are invoked: the student, the group size, the
batch size, the learning rate, the sampling parameters, the number of GPUs. You
do not choose them and nothing you write changes them.

## The two things you can do

Each turn you make exactly one decision.

**`evaluation`** — measure the current checkpoint on problems you write. You get
back per-problem pass rates and the student's full generations. Weights do not
change; you stay in the same step and decide again, now knowing more. Use as many
problems as you find useful.

**`train`** — hand over the curriculum. One GRPO update runs on it, the student's
weights advance, and the step closes. The `answer` you give for each problem
becomes the reward target directly, with no verification: **a wrong answer trains
the student toward a wrong answer.**

A step is one training update with any number of evaluations before it. There is
a cap on evaluations per step; your task message states it, and reaching it
without training halts the run.

## What you are told about the targets

After every training update the student is asked the {{N_TARGETS}} problems directly and
graded by a process that holds the answers. You are told two things: the pass
rate on each, and what the student actually wrote. You are not told which of its
attempts were correct — an attempt labelled correct is an answer.

Expect zeroes for a long time, possibly throughout. These are research problems
and the student is small.

## Constraints

- **You are not given the targets' answers and cannot obtain them.** They are
  encrypted at rest under a key that is not in your environment.
- **Do not look them up.** They come from recent arXiv papers and a public
  benchmark, so the answers are findable online. Every command you run is logged
  and audited for network access, and a run in which an answer is fetched is
  discarded rather than explained.
- **Do not put the target problems, or paraphrases of them, in the training
  data.** Everything you write is checked against them for overlap.
- Work from what is in this run's directory. Do not go looking for other runs or
  other experiments elsewhere on the filesystem.

## What you write

Two files in your current working directory:

**`decision.json`** — one object:
`{"step": <N>, "decision": "evaluation"}` or `{"step": <N>, "decision": "train"}`.
Use the step number from your task message.

**`data.jsonl`** — one JSON object per line:
`{"id": "<unique>", "problem": "<full statement>", "answer": "<final answer>"}`.
`answer` is the bare final answer, in the form a solver would put inside
`\boxed{}` — `42`, `\frac{1}{2}`, `x^2+1`. All three fields are required and must
be non-empty.

For an `evaluation`: any number of problems.
For a `train`: **exactly** the number given as `train_batch_size` in your task
message. A different count is a protocol error and halts the run.

Malformed JSON, a missing field, or an unrecognised decision also halts the run.
Write the two files, then stop.

## The run directory

Its path is in your task message. Read anything in it.

```
run_<timestamp>/
  pipeline/               the frozen setup, and the code that runs your decisions
  targets/                the {{N_TARGETS}} problems scored on the BASE model, before
                          any training
  events.jsonl            one line per event, whole run, in order
  metrics.jsonl           one line per completed sub-action
  step_<N>/
    eval_<M>/             one per evaluation you ran
      decision.json data.jsonl
      summary.json        pass@k, truncation_rate, no_answer_rate, sampling
      records.jsonl       per problem: your row, plus n and c and per-sample detail
      generations.jsonl   per sample: the student's full output
    train/                the update that closed the step
      data.jsonl          your curriculum
      train.log           verl's output, including its full per-step metric line
      ckpt/…/huggingface  the new weights
    targets/              the {{N_TARGETS}} problems scored on the checkpoint this step
                          produced
      per_target.jsonl    id, n, pass_at_1
      target_traces.json  what the student wrote, with no verdict attached
    result.json           the step's summary
```

`records.jsonl`, `generations.jsonl` and `target_traces.json` are the ones that
repay reading closely. A pass rate tells you a problem was hard; the generations
tell you *how* it failed — a wrong method, an arithmetic slip, a correct solution
the extractor missed, an answer the student never reached before running out of
tokens. Those have different implications for what to teach next.

Your task message summarises each closed step with three numbers from the
training update: the fraction of rollouts that were correct, the largest
advantage any group carried, and the fraction of completions that hit the token
limit. The rest of verl's metrics are in each step's `train.log`.

Anything that is not the student is yours to use freely. Read files, grep, run
shell commands, compute, write scratch notes. Just understand that none of it
touches the student; only the two decisions above do.

## Your latitude

No curriculum strategy is prescribed. How you use your evaluations, what you
teach, and how you respond to what you measure are yours to decide.

---

# The {{N_TARGETS}} problems

{{TARGETS}}
