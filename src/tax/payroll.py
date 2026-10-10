"""The payroll pipeline: LST, PAYE, NSSF, net pay.

Order of deductions, and why it is this order:

1. gross = basic pay + taxable allowances
2. LST, which is an allowable deduction against employment income, so it
   comes off before PAYE. The brief says "applied before PAYE" and the Local
   Governments Act treatment agrees.
3. PAYE on gross less LST
4. NSSF employee 5% on gross, not on gross-less-LST, and not deductible
   against PAYE. This is the step most often got wrong: Uganda does not allow
   the employee contribution as a deduction against employment income the way
   several neighbours do, so PAYE and NSSF sit on overlapping bases and
   neither reduces the other.
5. net = gross - LST - PAYE - NSSF employee
6. NSSF employer 10% never touches net pay. It sits on top of the wage and is
   reported as part of the cost of employment.

Each step is a Deduction applied in sequence, so the order is data you can
print and test rather than control flow inside a function.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterator, Sequence

import pandas as pd

from src.tax.core import Tax, TaxContext, TaxResult
from src.tax.errors import InvalidRateError
from src.tax.money import Money
from src.tax.paye import PayeTax

__all__ = ["Deduction", "LocalServiceTax", "NssfContribution", "PayeDeduction",
           "Payslip", "PayrollRun", "STANDARD_LST_BANDS", "lst_for_income"]

# Employee and employer NSSF rates, from the brief.
NSSF_EMPLOYEE_RATE = Decimal("0.05")
NSSF_EMPLOYER_RATE = Decimal("0.10")

# Indicative annual Local Service Tax by monthly income, deducted in four
# equal instalments between July and October. LST is set by each local
# government, so these are a documented default, not a national rate.
STANDARD_LST_BANDS: tuple[tuple[Money, Money], ...] = (
    (Money("100000"), Money("5000")),
    (Money("200000"), Money("10000")),
    (Money("300000"), Money("20000")),
    (Money("400000"), Money("30000")),
    (Money("500000"), Money("40000")),
    (Money("1000000"), Money("60000")),
    (Money("99999999999"), Money("100000")),
)


def lst_for_income(monthly_gross: Money,
                   bands: Sequence[tuple[Money, Money]] = STANDARD_LST_BANDS,
                   instalments: int = 4) -> Money:
    """The monthly Local Service Tax instalment for a given gross.

    The annual charge is banded by income and collected in equal instalments;
    this returns the instalment, not the annual figure.
    """
    if monthly_gross.is_negative():
        raise ValueError("gross cannot be negative")
    if instalments < 1:
        raise ValueError("instalments must be at least 1")
    for ceiling, annual in bands:
        if monthly_gross <= ceiling:
            return annual * (Decimal(1) / Decimal(instalments))
    return bands[-1][1] * (Decimal(1) / Decimal(instalments))


class Deduction(ABC):
    """One line on a payslip."""

    # Whether this line reduces the employee's take-home pay.
    reduces_net: bool = True

    # Whether PAYE is charged on income *after* this line.
    before_paye: bool = False

    def __init__(self, label: str) -> None:
        self.label = label

    @abstractmethod
    def amount(self, ctx: TaxContext, running: Money) -> Money:
        """How much this line takes, given the context and the running base."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label!r})"


class LocalServiceTax(Deduction):
    """Local Service Tax, deducted before PAYE.

    Construct with enabled=False for an employee who is not liable, or pass a
    fixed monthly amount: LST is set locally and a district need not use the
    indicative bands.
    """

    before_paye = True

    def __init__(self, enabled: bool = True, monthly: Money | None = None,
                 bands: Sequence[tuple[Money, Money]] = STANDARD_LST_BANDS,
                 instalments: int = 4) -> None:
        super().__init__("Local Service Tax")
        self.enabled = bool(enabled)
        self.monthly = monthly
        self.bands = tuple(bands)
        self.instalments = int(instalments)

    def amount(self, ctx: TaxContext, running: Money) -> Money:
        if not self.enabled:
            return Money.zero()
        if self.monthly is not None:
            return Money(self.monthly)
        return lst_for_income(ctx.chargeable_employment_income, self.bands,
                              self.instalments)


class NssfContribution(Deduction):
    """NSSF, charged on gross and not deductible against PAYE."""

    def __init__(self, rate: Decimal = NSSF_EMPLOYEE_RATE,
                 borne_by_employee: bool = True) -> None:
        if not isinstance(rate, Decimal):
            raise InvalidRateError(rate, "NSSF rate must be a Decimal")
        if not Decimal("0") <= rate <= Decimal("1"):
            raise InvalidRateError(rate)
        who = "employee" if borne_by_employee else "employer"
        super().__init__(f"NSSF {who} ({rate:.0%})")
        self.rate = rate
        self.reduces_net = bool(borne_by_employee)

    def amount(self, ctx: TaxContext, running: Money) -> Money:
        """Charged on gross plus allowances, deliberately not on running."""
        return ctx.chargeable_employment_income * self.rate


class PayeDeduction(Deduction):
    """PAYE, charged on whatever is left after the before-PAYE lines."""

    def __init__(self, tax: Tax | None = None) -> None:
        super().__init__("PAYE")
        self.tax = tax or PayeTax()
        self.last_result: TaxResult | None = None

    def amount(self, ctx: TaxContext, running: Money) -> Money:
        already = ctx.chargeable_employment_income - running
        charged = ctx.replace(extras={**dict(ctx.extras),
                                      "pre_paye_deductions": already})
        self.last_result = self.tax.calculate(charged)
        return self.last_result.payable


@dataclass(frozen=True)
class Payslip:
    """The outcome of one payroll run."""

    period: dt.date
    gross: Money
    lines: tuple[tuple[str, Money], ...]
    paye: TaxResult
    net: Money
    employer_cost: Money

    def __iter__(self) -> Iterator[tuple[str, Money]]:
        return iter(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __repr__(self) -> str:
        return (f"Payslip({self.period.isoformat()}, gross={self.gross}, "
                f"net={self.net})")

    @property
    def total_deductions(self) -> Money:
        """Everything taken off the employee's gross."""
        return self.gross - self.net

    @property
    def take_home_rate(self) -> Decimal:
        if self.gross.is_zero():
            return Decimal("0")
        return (self.net.amount / self.gross.amount).quantize(Decimal("0.0001"))

    def to_frame(self) -> pd.DataFrame:
        """The payslip as a table, in the order the deductions were applied."""
        rows = [{"line": "Gross pay", "amount_ugx": self.gross.amount}]
        rows += [{"line": label, "amount_ugx": -amount.amount}
                 for label, amount in self.lines]
        rows.append({"line": "Net pay", "amount_ugx": self.net.amount})
        rows.append({"line": "Employer NSSF (not deducted)",
                     "amount_ugx": (self.employer_cost - self.gross).amount})
        return pd.DataFrame(rows)

    def explain(self) -> str:
        parts = [f"Payroll for {self.period.isoformat()}",
                 f"  Gross pay                     {self.gross:>18}"]
        for label, amount in self.lines:
            parts.append(f"  less {label:<24} {-amount:>18}")
        parts.append(f"  Net pay                       {self.net:>18}")
        parts.append(f"  (take-home {self.take_home_rate:.2%} of gross)")
        parts.append(f"  Employer total cost           {self.employer_cost:>18}")
        return "\n".join(parts)


class PayrollRun:
    """Applies the deductions in order and produces a Payslip."""

    def __init__(self, deductions: Sequence[Deduction] | None = None,
                 employer_nssf: NssfContribution | None = None) -> None:
        self.deductions = tuple(deductions) if deductions is not None else (
            LocalServiceTax(), PayeDeduction(), NssfContribution())
        self.employer_nssf = employer_nssf or NssfContribution(
            NSSF_EMPLOYER_RATE, borne_by_employee=False)

    def __repr__(self) -> str:
        return f"PayrollRun({[d.label for d in self.deductions]})"

    def order(self) -> pd.DataFrame:
        """The deduction order as a table, so it can be read and justified."""
        return pd.DataFrame([
            {"step": index + 1, "line": d.label,
             "charged before PAYE": d.before_paye,
             "reduces net pay": d.reduces_net}
            for index, d in enumerate(self.deductions)])

    def run(self, ctx: TaxContext) -> Payslip:
        """Compute one month's payslip."""
        gross = ctx.chargeable_employment_income
        running = gross
        lines: list[tuple[str, Money]] = []
        paye_result: TaxResult | None = None

        for deduction in self.deductions:
            amount = deduction.amount(ctx, running)
            lines.append((deduction.label, amount))
            if deduction.before_paye:
                running = running - amount
            if isinstance(deduction, PayeDeduction):
                paye_result = deduction.last_result

        taken = Money.zero()
        for _, amount in lines:
            taken = taken + amount
        net = gross - taken
        employer = gross + self.employer_nssf.amount(ctx, gross)

        if paye_result is None:
            paye_result = PayeTax().calculate(ctx)
        return Payslip(period=ctx.period, gross=gross, lines=tuple(lines),
                       paye=paye_result, net=net, employer_cost=employer)
