"""Verifiers: given a model's raw text and a gold answer, is it correct?

Two of them, because the benchmarks split two ways and no further:

  exact_integer   AIME. Every answer is an integer 0-999 by the competition's
                  own rules, so integer comparison is exact and sidesteps
                  symbolic-equivalence ambiguity entirely.
  symbolic        MATH-500, HMMT, the MATH training pool. Answers are exact
                  forms - fractions, radicals, intervals, tuples - where
                  `\\frac{1}{2}`, `0.5` and `1/2` are the same answer and only a
                  symbolic comparator can say so.

They are passed in, not looked up by benchmark name. A task declares which one
it uses; nothing infers it from a filename. The previous arrangement inferred
`integer_answer` from `args.task == "aime"` buried in the evaluation loop, so
"unset" meant different things for different tasks and only reading the source
revealed which.
"""
from .exact_integer import ExactIntegerVerifier
from .symbolic import SymbolicVerifier

VERIFIERS = {
    "exact_integer": ExactIntegerVerifier,
    "symbolic": SymbolicVerifier,
}


def get_verifier(name: str):
    """Resolve a verifier by name, failing loudly on an unknown one."""
    if name not in VERIFIERS:
        raise ValueError(
            f"unknown verifier {name!r}; available: {sorted(VERIFIERS)}. "
            "A task must name its verifier explicitly - see configs/eval/*.yaml."
        )
    return VERIFIERS[name]()
