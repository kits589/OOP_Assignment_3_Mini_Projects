"""Money: exact shilling amounts, backed by Decimal.

Rounds half away from zero to whole shillings, which is what URA's published
tables do, and the shilling has no circulating subdivision so a fraction is
not payable anyway. Rounding happens in one place, quantize().

float never appears in a calculation. float_is_unsafe() returns the concrete
failures that justify that rule; the report works through them.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any, Union

__all__ = ["CURRENCY", "EXPONENT", "Money", "ROUNDING", "float_is_unsafe"]

CURRENCY = "UGX"

# Whole shillings: the smallest unit anyone can actually pay.
EXPONENT = Decimal("1")

# Half away from zero, matching URA's published tables.
ROUNDING = ROUND_HALF_UP

Numeric = Union[int, str, Decimal, "Money"]


class Money:
    """An exact amount of Ugandan shillings.

    >>> Money("335000") + Money(1)
    Money('335001')
    >>> Money("1000") * Decimal("0.3")
    Money('300')
    """

    __slots__ = ("_amount",)

    def __init__(self, amount: Numeric = 0) -> None:
        self._amount = self._coerce(amount)

    @staticmethod
    def _coerce(value: Numeric) -> Decimal:
        if isinstance(value, Money):
            return value._amount
        if isinstance(value, float):
            raise TypeError(
                "Money refuses float: 0.1 + 0.2 != 0.3 in binary floating point, "
                "and the error compounds through a progressive calculation. "
                "Pass an int, a str or a Decimal.")
        if isinstance(value, bool):
            raise TypeError("Money refuses bool")
        try:
            return Decimal(value).quantize(EXPONENT, rounding=ROUNDING)
        except (InvalidOperation, ValueError, TypeError) as problem:
            raise ValueError(f"cannot read {value!r} as an amount") from problem

    @classmethod
    def zero(cls) -> "Money":
        """The additive identity, named so totals start readably."""
        return cls(0)

    @classmethod
    def of(cls, value: Numeric) -> "Money":
        """Alias for the constructor, for readability at call sites."""
        return cls(value)

    @property
    def amount(self) -> Decimal:
        """The exact value, as a Decimal."""
        return self._amount

    def quantize(self) -> "Money":
        """Re-apply the rounding policy. Idempotent by construction."""
        return Money(self._amount)

    def is_zero(self) -> bool:
        return self._amount == 0

    def is_negative(self) -> bool:
        return self._amount < 0

    def __add__(self, other: Numeric) -> "Money":
        return Money(self._amount + Money._coerce(other))

    __radd__ = __add__

    def __sub__(self, other: Numeric) -> "Money":
        return Money(self._amount - Money._coerce(other))

    def __rsub__(self, other: Numeric) -> "Money":
        return Money(Money._coerce(other) - self._amount)

    def __mul__(self, factor: int | str | Decimal) -> "Money":
        """Multiply by a rate. Shillings times shillings has no meaning, so Money
        times Money is a TypeError.

        Done at extended precision and rounded once, so a chain of rates does
        not accumulate error.
        """
        if isinstance(factor, Money):
            raise TypeError("multiplying Money by Money is meaningless")
        if isinstance(factor, float):
            raise TypeError("Money refuses float rates; use Decimal('0.3')")
        with localcontext() as context:
            context.prec = 28
            return Money(Decimal(factor) * self._amount)

    __rmul__ = __mul__

    def __neg__(self) -> "Money":
        return Money(-self._amount)

    def __abs__(self) -> "Money":
        return Money(abs(self._amount))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (Money, int, str, Decimal)):
            try:
                return self._amount == Money._coerce(other)
            except (TypeError, ValueError):
                return NotImplemented
        return NotImplemented

    def __lt__(self, other: Numeric) -> bool:
        return self._amount < Money._coerce(other)

    def __le__(self, other: Numeric) -> bool:
        return self._amount <= Money._coerce(other)

    def __gt__(self, other: Numeric) -> bool:
        return self._amount > Money._coerce(other)

    def __ge__(self, other: Numeric) -> bool:
        return self._amount >= Money._coerce(other)

    def __hash__(self) -> int:
        return hash((CURRENCY, self._amount))

    def __bool__(self) -> bool:
        return self._amount != 0

    def __repr__(self) -> str:
        return f"Money('{self._amount}')"

    def __str__(self) -> str:
        return f"{CURRENCY} {self._amount:,}"

    def __format__(self, spec: str) -> str:
        return format(self._amount, spec) if spec else str(self)


def float_is_unsafe(rate: str = "0.2", shillings: int = 335_001,
                    repeats: int = 10_000) -> dict[str, Any]:
    """Three ways binary floating point gets tax arithmetic wrong.

    All three were checked to fail; none is hypothetical.

    1. round(2.675, 2) gives 2.67, not 2.68, because the nearest double to
       2.675 sits just below it. Half-up on a float therefore rounds half
       down whenever the stored value falls on the low side, and the decimal
       digits give no warning which way it will go.
    2. (0.1 + 0.2) + 0.3 and 0.1 + (0.2 + 0.3) are different doubles, so a
       payslip's total would depend on the order its lines were summed.
    3. Applying the same rate repeatedly drifts, and the drift grows with
       the number of operations.
    """
    binary_total, exact_total = 0.0, Decimal(0)
    step, exact_step = float(rate), Decimal(rate)
    for _ in range(int(repeats)):
        binary_total += shillings * step
        exact_total += Decimal(shillings) * exact_step
    drift = Decimal(repr(binary_total)) - exact_total

    left = (0.1 + 0.2) + 0.3
    right = 0.1 + (0.2 + 0.3)

    return {
        # 1. the rounding trap
        "round_input": "2.675",
        "round_float": round(2.675, 2),
        "round_exact": Decimal("2.675").quantize(Decimal("0.01"),
                                                 rounding=ROUND_HALF_UP),
        "round_wrong": round(2.675, 2) != float(
            Decimal("2.675").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        "round_stored_as": Decimal(2.675),

        # 2. order dependence
        "assoc_left": left,
        "assoc_right": right,
        "assoc_equal": left == right,

        # 3. accumulation
        "float_total": binary_total,
        "decimal_total": exact_total,
        "drift": drift,
        "repeats": int(repeats),

        # the one-liner everyone knows
        "classic": f"0.1 + 0.2 == 0.3 is {0.1 + 0.2 == 0.3}",
        "classic_value": repr(0.1 + 0.2),

        "note": (f"round(2.675, 2) -> {round(2.675, 2)} where the exact answer is "
                 f"2.68; float addition is not associative; and {repeats:,} "
                 f"applications of {rate} to {shillings:,} drift by {drift} UGX."),
    }
