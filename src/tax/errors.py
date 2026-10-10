"""Exceptions for the tax engine.

One base so a caller can catch TaxError and be done with it, subclasses so it
can be specific. Nothing here returns a sentinel or a zero to signal failure.
"""
from __future__ import annotations

import datetime as dt

__all__ = ["InvalidIncomeError", "InvalidRateError", "NoScheduleForDateError",
           "TaxError", "UnknownSupplyError"]


class TaxError(Exception):
    """Base class for every error this package raises."""


class InvalidIncomeError(TaxError):
    """Raised when an amount cannot be a chargeable income.

    Negative pay is the usual cause, and it almost always means a sign was lost
    upstream, so flooring it at zero would hide a real bug.
    """

    def __init__(self, amount: object, reason: str = "income cannot be negative"):
        self.amount = amount
        self.reason = reason
        super().__init__(f"{reason}: got {amount}")


class InvalidRateError(TaxError):
    """Raised when a rate is outside a sensible range."""

    def __init__(self, rate: object, reason: str = "rate must lie in 0..1"):
        self.rate = rate
        self.reason = reason
        super().__init__(f"{reason}: got {rate}")


class NoScheduleForDateError(TaxError):
    """Raised when no rate schedule covers the date being calculated.

    Schedules are effective-dated, and a date with no schedule is a gap in the
    data, not a cue to reach for the nearest one.
    """

    def __init__(self, when: dt.date, residency: str = "",
                 known: list[str] | None = None) -> None:
        self.when = when
        self.residency = residency
        self.known = known or []
        detail = f" for {residency}" if residency else ""
        coverage = f"; schedules on file: {', '.join(self.known)}" if self.known else ""
        super().__init__(f"no tax schedule{detail} covers {when}{coverage}")


class UnknownSupplyError(TaxError):
    """Raised when an invoice carries a supply type the VAT module cannot rate."""

    def __init__(self, supply: object) -> None:
        self.supply = supply
        super().__init__(f"unknown supply type {supply!r}")
