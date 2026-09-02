"""AIME: the answer is an integer 0-999, so compare integers.

This is not a shortcut. The competition constrains answers to that range, which
makes exact comparison both correct and immune to the symbolic-equivalence edge
cases the other verifier has to handle. It falls back to symbolic comparison
when a model writes something that is not a bare integer - "\\frac{204}{1}" is a
legitimate way to write 204 - so nothing is scored wrong for formatting alone.
"""
from .base import Verdict, Verifier
from .extract import _int_or_none, extract_answer, grade as symbolic_grade


class ExactIntegerVerifier(Verifier):
    name = "exact_integer"

    def grade(self, text: str, gold: str) -> Verdict:
        pred = extract_answer(text)
        if pred is None:
            return Verdict(False, None)
        pi, gi = _int_or_none(pred), _int_or_none(gold)
        if pi is not None and gi is not None:
            return Verdict(pi == gi, pred)
        ok, extracted = symbolic_grade(text, gold, integer_answer=False)
        return Verdict(bool(ok), extracted)
