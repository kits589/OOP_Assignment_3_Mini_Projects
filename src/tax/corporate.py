"""Corporate income tax, plus extension A (withholding and rental tax).

Flat 30% on chargeable income, which is revenue less allowable deductions.
A disallowed expense still reduces book profit but not taxable profit, so the
two figures diverge; Expense carries the flag and the result reports the
add-back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Iterator, Sequence

import pandas as pd

from src.tax.core import Tax, TaxContext, TaxResult, Workings
from src.tax.errors import InvalidIncomeError, InvalidRateError
from src.tax.money import Money

__all__ = ["CORPORATE_RATE", "CorporateIncomeTax", "Expense", "PaymentType",
           "RENTAL_ALLOWANCE", "RENTAL_RATE", "RENTAL_THRESHOLD",
           "RentalIncomeTax", "WITHHOLDING_RATES", "WithholdingTax"]

# Resident company rate, from the brief.
CORPORATE_RATE = Decimal("0.30")


@dataclass(frozen=True)
class Expense:
    """One cost, flagged by whether the Act allows it as a deduction."""

    description: str
    amount: Money
    allowable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Money):
            object.__setattr__(self, "amount", Money(self.amount))
        if self.amount.is_negative():
            raise InvalidIncomeError(self.amount, "an expense cannot be negative")

    def __str__(self) -> str:
        flag = "allowable" if self.allowable else "DISALLOWED"
        return f"{self.description}: {self.amount} ({flag})"


class CorporateIncomeTax(Tax):
    """Corporate income tax at a flat rate on chargeable income."""

    name = "Corporate income tax"

    def __init__(self, rate: Decimal = CORPORATE_RATE) -> None:
        if not isinstance(rate, Decimal):
            raise InvalidRateError(rate, "the corporate rate must be a Decimal")
        if not Decimal("0") <= rate <= Decimal("1"):
            raise InvalidRateError(rate)
        self.rate = rate

    @staticmethod
    def chargeable_income(revenue: Money,
                          expenses: Sequence[Expense]) -> tuple[Money, Money]:
        """Chargeable income and the total disallowed amount added back."""
        allowed, disallowed = Money.zero(), Money.zero()
        for expense in expenses:
            if expense.allowable:
                allowed = allowed + expense.amount
            else:
                disallowed = disallowed + expense.amount
        chargeable = revenue - allowed
        return (Money.zero() if chargeable.is_negative() else chargeable), disallowed

    def calculate(self, ctx: TaxContext) -> TaxResult:
        """Apply the rate to revenue less allowable deductions.

        Expenses can arrive as Expense objects in extras['expenses'], in which
        case disallowed ones are added back. Otherwise ctx.deductions is taken
        as already allowable.
        """
        expenses = ctx.extras.get("expenses")
        if expenses:
            chargeable, disallowed = self.chargeable_income(ctx.revenue, expenses)
            note = (f"{disallowed} of expenses disallowed and added back"
                    if disallowed else "all expenses allowable")
        else:
            chargeable = ctx.revenue - ctx.deductions
            if chargeable.is_negative():
                chargeable = Money.zero()
            disallowed, note = Money.zero(), "deductions taken as allowable"

        payable = chargeable * self.rate
        workings = (
            Workings("revenue less allowable deductions", chargeable,
                     self.rate, payable),)
        notes = (note, "a loss carries forward; this period's charge is nil"
                 ) if chargeable.is_zero() else (note,)
        return TaxResult(tax=self.name, chargeable=chargeable, payable=payable,
                         workings=workings, notes=notes)


# ----------------------------------------------------------------------
# Extension A: withholding tax and rental income tax
# ----------------------------------------------------------------------
class PaymentType(Enum):
    """Payments that attract withholding tax, with their rates."""

    GOODS_AND_SERVICES = "goods and services"
    PROFESSIONAL_FEES = "professional fees"
    DIVIDENDS = "dividends"
    INTEREST = "interest"
    ROYALTIES = "royalties"
    RENT_TO_RESIDENT = "rent paid to a resident"

    def __str__(self) -> str:
        return self.value


# Indicative resident withholding rates (Income Tax Act Cap 340, Part VIII).
# Rates differ for non-residents and under double-taxation treaties; those
# cases are out of scope here, and the engine says so instead of guessing.
WITHHOLDING_RATES: dict[PaymentType, Decimal] = {
    PaymentType.GOODS_AND_SERVICES: Decimal("0.06"),
    PaymentType.PROFESSIONAL_FEES: Decimal("0.06"),
    PaymentType.DIVIDENDS: Decimal("0.15"),
    PaymentType.INTEREST: Decimal("0.15"),
    PaymentType.ROYALTIES: Decimal("0.15"),
    PaymentType.RENT_TO_RESIDENT: Decimal("0.10"),
}

# Withholding on goods and services only bites above this threshold.
WHT_THRESHOLD = Money("1000000")


class WithholdingTax(Tax):
    """Tax withheld at source on a payment."""

    name = "Withholding tax"

    def __init__(self, payment_type: PaymentType = PaymentType.GOODS_AND_SERVICES,
                 rates: dict[PaymentType, Decimal] | None = None,
                 threshold: Money = WHT_THRESHOLD) -> None:
        if not isinstance(payment_type, PaymentType):
            raise ValueError(f"unknown payment type {payment_type!r}")
        self.payment_type = payment_type
        self.rates = dict(rates or WITHHOLDING_RATES)
        self.threshold = threshold

    @property
    def rate(self) -> Decimal:
        return self.rates[self.payment_type]

    def calculate(self, ctx: TaxContext) -> TaxResult:
        """Withhold on the gross payment in ctx.revenue."""
        gross = ctx.revenue
        applies = (self.payment_type is not PaymentType.GOODS_AND_SERVICES
                   or gross >= self.threshold)
        payable = gross * self.rate if applies else Money.zero()
        workings = (Workings(f"withholding on {self.payment_type}", gross,
                             self.rate if applies else Decimal("0"), payable),)
        notes = [f"{self.payment_type} withheld at {self.rate:.0%}"]
        if not applies:
            notes.append(f"below the {self.threshold} threshold, so nothing withheld")
        notes.append("resident rates only; non-resident and treaty rates differ")
        return TaxResult(tax=self.name, chargeable=gross, payable=payable,
                         workings=workings, notes=tuple(notes))


# Rental income tax for an individual: a 20% allowance for expenses, a
# threshold below which nothing is charged, and a 12% rate on the balance.
RENTAL_ALLOWANCE = Decimal("0.20")
RENTAL_THRESHOLD = Money("2820000")
RENTAL_RATE = Decimal("0.12")


class RentalIncomeTax(Tax):
    """Rental income tax for an individual landlord.

    Chargeable income is gross rent less a flat 20% expense allowance and an
    annual threshold, with the balance at 12%. Rental income is its own
    chargeable head, not another employment band, so this does not inherit from
    PayeTax even though both are charges on an individual.
    """

    name = "Rental income tax"

    def __init__(self, allowance: Decimal = RENTAL_ALLOWANCE,
                 threshold: Money = RENTAL_THRESHOLD,
                 rate: Decimal = RENTAL_RATE) -> None:
        for value in (allowance, rate):
            if not isinstance(value, Decimal):
                raise InvalidRateError(value, "rental rates must be Decimal")
            if not Decimal("0") <= value <= Decimal("1"):
                raise InvalidRateError(value)
        self.allowance = allowance
        self.threshold = threshold
        self.rate = rate

    def calculate(self, ctx: TaxContext) -> TaxResult:
        """Charge on annual gross rent supplied as ctx.revenue."""
        gross = ctx.revenue
        if gross.is_negative():
            raise InvalidIncomeError(gross, "rent cannot be negative")
        allowed = gross * self.allowance
        after_expenses = gross - allowed
        chargeable = after_expenses - self.threshold
        if chargeable.is_negative():
            chargeable = Money.zero()
        payable = chargeable * self.rate

        workings = (
            Workings(f"expense allowance at {self.allowance:.0%}", gross,
                     self.allowance, allowed),
            Workings(f"less threshold {self.threshold}", after_expenses,
                     Decimal("1"), chargeable),
            Workings("rental income tax", chargeable, self.rate, payable),
        )
        notes = [f"individual landlord: {self.allowance:.0%} allowance, "
                 f"{self.threshold} threshold, {self.rate:.0%} rate"]
        if chargeable.is_zero():
            notes.append("below the threshold, so no tax is due")
        return TaxResult(tax=self.name, chargeable=chargeable, payable=payable,
                         workings=workings, notes=tuple(notes))
