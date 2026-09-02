"""MATH-500, HMMT, and the MATH training pool: compare exact forms symbolically.

Answers here are fractions, radicals, intervals and tuples, where `\\frac{1}{2}`,
`0.5` and `1/2` are one answer written three ways. math_verify does the
comparison; this class only chooses it and reports the extraction separately.

One documented looseness: math_verify compares numerically within a tolerance,
so a truncated decimal matches an exact form. That makes grading more permissive
- equally so for every model - and is recorded rather than fixed.
"""
from .base import Verdict, Verifier
from .extract import grade as symbolic_grade


class SymbolicVerifier(Verifier):
    name = "symbolic"

    def grade(self, text: str, gold: str) -> Verdict:
        ok, extracted = symbolic_grade(text, gold, integer_answer=False)
        return Verdict(bool(ok), extracted)
