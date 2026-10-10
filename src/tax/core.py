"""The Tax contract: a context in, a TaxResult out.

PAYE, VAT and corporate tax share no arithmetic, so the base class holds
none of it. All it fixes is the signature, which is what lets a payroll run
or a report treat them interchangeably.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterator, Mapping

from src.tax.errors import InvalidIncomeError
from src.tax.money import Money

__all__ = ["Residency", "Tax", "TaxContext", "TaxResult", "Workings"]


class Residency(str):
    """Residency status, as a string subclass so it prints and compares plainly."""

    RESIDENT = "resident"
    NON_RESIDENT = "non-resident"

    @classmethod
    def check(cls, value: str) -> str:
        allowed = {cls.RESIDENT, cls.NON_RESIDENT}
        text = str(value).strip().lower()
        if text not in allowed:
            raise ValueError(f"residency must be one of {sorted(allowed)}, "
                             f"got {value!r}")
        return text


@dataclass(frozen=True)
class TaxContext:
    """Everything a tax needs in order to calculate itself.

    One context type for all of them, so the Tax signature stays uniform. Each
    tax ignores the fields that do not apply to it: VAT has no use for
    residency, PAYE none for revenue.
    """

    period: dt.date
    gross: Money = field(default_factory=Money.zero)
    allowances: Money = field(default_factory=Money.zero)
    residency: str = Residency.RESIDENT
    revenue: Money = field(default_factory=Money.zero)
    deductions: Money = field(default_factory=Money.zero)
    invoices: tuple = ()
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.period, dt.date):
            raise TypeError("period must be a datetime.date")
        object.__setattr__(self, "residency", Residency.check(self.residency))
        for name in ("gross", "allowances", "revenue", "deductions"):
            value = getattr(self, name)
            if not isinstance(value, Money):
                object.__setattr__(self, name, Money(value))
            if getattr(self, name).is_negative():
                raise InvalidIncomeError(getattr(self, name),
                                         f"{name} cannot be negative")

    @property
    def chargeable_employment_income(self) -> Money:
        """Gross plus taxable allowances, before any deduction."""
        return self.gross + self.allowances

    def replace(self, **changes: Any) -> "TaxContext":
        """A copy with some fields changed; the context itself stays frozen."""
        fields = {"period": self.period, "gross": self.gross,
                  "allowances": self.allowances, "residency": self.residency,
                  "revenue": self.revenue, "deductions": self.deductions,
                  "invoices": self.invoices, "extras": self.extras}
        fields.update(changes)
        return TaxContext(**fields)


@dataclass(frozen=True)
class Workings:
    """One line of the arithmetic behind a result, so a figure can explain itself."""

    label: str
    base: Money
    rate: Decimal
    amount: Money

    def __str__(self) -> str:
        return (f"{self.label}: {self.base} x {self.rate} = {self.amount}")


@dataclass(frozen=True)
class TaxResult:
    """What a tax computed, and how."""

    tax: str
    chargeable: Money
    payable: Money
    workings: tuple[Workings, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.payable.is_negative():
            raise InvalidIncomeError(self.payable, "tax payable cannot be negative")

    @property
    def effective_rate(self) -> Decimal:
        """Tax as a share of the chargeable amount; zero when nothing is charged."""
        if self.chargeable.is_zero():
            return Decimal("0")
        return (self.payable.amount / self.chargeable.amount).quantize(
            Decimal("0.000001"))

    def __iter__(self) -> Iterator[Workings]:
        return iter(self.workings)

    def __len__(self) -> int:
        return len(self.workings)

    def __repr__(self) -> str:
        return (f"TaxResult({self.tax}, chargeable={self.chargeable}, "
                f"payable={self.payable}, rate={self.effective_rate:.2%})")

    def explain(self) -> str:
        """The full arithmetic, one line per band."""
        lines = [f"{self.tax} on {self.chargeable}"]
        lines.extend(f"  {w}" for w in self.workings)
        lines.append(f"  = {self.payable}  "
                     f"(effective rate {self.effective_rate:.2%})")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


class Tax(ABC):
    """The abstract base every tax implements."""

    name: str = "tax"

    @abstractmethod
    def calculate(self, ctx: TaxContext) -> TaxResult:
        """Compute the tax for one context."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
