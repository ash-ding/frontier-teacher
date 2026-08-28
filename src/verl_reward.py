"""Reward function for verl, wrapping this project's grading path.

Training reward and evaluation score must come from one code path. If they do
not, a rise in benchmark score can be produced by the grader rather than the
model, and nothing downstream distinguishes the two.

Wire it with the NEW-style config keys:
    reward.custom_reward_function.path=/abs/path/src/verl_reward.py
    reward.custom_reward_function.name=compute_score
The legacy top-level `custom_reward_function` key is not migrated by
verl.trainer.main_ppo (only by the fully-async entry point).

verl calls this by keyword from
verl/experimental/reward_loop/reward_manager/naive.py.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grading import grade          # noqa: E402
from evaluate import answer_segment  # noqa: E402  - the </think> split lives there


def compute_score(data_source=None, solution_str="", ground_truth="",
                  extra_info=None, **kwargs):
    """1.0 if the response grades correct, else 0.0.

    integer_answer stays False: this pool is MATH, whose answers are exact forms
    rather than the 0-999 integers AIME uses. It matches the mathtrain task config.
    """
    try:
        ok, extracted = grade(answer_segment(solution_str), ground_truth,
                              integer_answer=False)
    except Exception:
        # A grader crash must not take down the training run; score it wrong and
        # surface the count through the aux metric below.
        return {"score": 0.0, "acc": 0.0, "grader_error": 1.0}
    return {"score": 1.0 if ok else 0.0,
            "acc": 1.0 if ok else 0.0,
            "no_answer": 0.0 if extracted is not None else 1.0}
