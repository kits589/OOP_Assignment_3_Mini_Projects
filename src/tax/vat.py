"""VAT: standard-rated, zero-rated and exempt supplies.

Zero-rated and exempt both produce no output VAT, which makes them look
identical on a sales report. They are not. Zero-rated is a taxable supply
charged at 0%, so input VAT incurred making it stays recoverable; exempt is
outside the VAT net, so input VAT attributable to it is blocked and becomes a
cost. Conflating them always overstates the refund, so recoverable and
irrecoverable input are tracked separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Iterator, Sequence

import pandas as pd

from src.tax.core import Tax, TaxContext, TaxResult, Workings
from src.tax.errors import InvalidRateError, UnknownSupplyError
from src.tax.money import Money

__all__ = ["Invoice", "STANDARD_RATE", "SupplyType", "VatReturn", "VatTax"]

# The standard rate, from the brief.
STANDARD_RATE = Decimal("0.18")


class SupplyType(Enum):
    """How a supply is treated for VAT."""

    STANDARD = "standard-rated"
    ZERO_RATED = "zero-rated"
    EXEMPT = "exempt"

    @property
    def output_rate(self) -> Decimal:
        """The rate charged on a sale of this type."""
        return STANDARD_RATE if self is SupplyType.STANDARD else Decimal("0")

    @property
    def input_recoverable(self) -> bool:
        """True for standard and zero-rated supplies, False for exempt ones."""
        return self is not SupplyType.EXEMPT

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Invoice:
    """One sale or purchase."""

    reference: str
    net: Money
    supply: SupplyType = SupplyType.STANDARD
    is_purchase: bool = False
    # Defaults to the computed amount. Set it when a supplier has charged
    # something unexpected, which is when a VAT return goes wrong.
    vat_charged: Money | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.supply, SupplyType):
            raise UnknownSupplyError(self.supply)
        if not isinstance(self.net, Money):
            object.__setattr__(self, "net", Money(self.net))
        if self.net.is_negative():
            raise ValueError(f"{self.reference}: a net amount cannot be negative")

    @property
    def vat(self) -> Money:
        """VAT on this invoice."""
        if self.vat_charged is not None:
            return Money(self.vat_charged)
        return self.net * self.supply.output_rate

    @property
    def gross(self) -> Money:
        return self.net + self.vat

    def __str__(self) -> str:
        side = "purchase" if self.is_purchase else "sale"
        return (f"{self.reference} ({side}, {self.supply}): "
                f"{self.net} + {self.vat} VAT")


@dataclass(frozen=True)
class VatReturn:
    """A completed VAT return for a period."""

    output_vat: Money
    input_vat_recoverable: Money
    input_vat_blocked: Money
    net_payable: Money
    by_supply: pd.DataFrame = field(repr=False, default=None)

    @property
    def is_refund(self) -> bool:
        """True when more input VAT was recovered than output VAT charged."""
        return self.net_payable.is_negative()

    def __repr__(self) -> str:
        direction = "refund due" if self.is_refund else "payable"
        return (f"VatReturn(output={self.output_vat}, "
                f"recoverable={self.input_vat_recoverable}, "
                f"{direction}={abs(self.net_payable)})")

    def explain(self) -> str:
        return "\n".join([
            f"  Output VAT on sales              {self.output_vat:>18}",
            f"  less recoverable input VAT       {-self.input_vat_recoverable:>18}",
            f"  = net VAT "
            f"{'refundable' if self.is_refund else 'payable':<22}"
            f"{abs(self.net_payable):>18}",
            f"  (input VAT blocked as attributable to exempt supplies: "
            f"{self.input_vat_blocked})",
        ])


class VatTax(Tax):
    """Value Added Tax over a list of invoices."""

    name = "VAT"

    def __init__(self, rate: Decimal = STANDARD_RATE) -> None:
        if not isinstance(rate, Decimal):
            raise InvalidRateError(rate, "the VAT rate must be a Decimal")
        if not Decimal("0") <= rate <= Decimal("1"):
            raise InvalidRateError(rate)
        self.rate = rate

    def output_vat(self, invoices: Sequence[Invoice]) -> Money:
        """VAT charged on sales."""
        total = Money.zero()
        for invoice in invoices:
            if not invoice.is_purchase:
                total = total + invoice.vat
        return total

    def input_vat(self, invoices: Sequence[Invoice]) -> tuple[Money, Money]:
        """Input VAT split into (recoverable, blocked).

        Blocked VAT is the part attributable to exempt supplies, which the
        business absorbs as a cost.
        """
        recoverable, blocked = Money.zero(), Money.zero()
        for invoice in invoices:
            if not invoice.is_purchase:
                continue
            if invoice.supply.input_recoverable:
                recoverable = recoverable + invoice.vat
            else:
                blocked = blocked + invoice.vat
        return recoverable, blocked

    def prepare_return(self, invoices: Sequence[Invoice]) -> VatReturn:
        """Build the full return from a list of invoices."""
        output = self.output_vat(invoices)
        recoverable, blocked = self.input_vat(invoices)
        rows = []
        for supply in SupplyType:
            sales = [i for i in invoices if not i.is_purchase and i.supply is supply]
            buys = [i for i in invoices if i.is_purchase and i.supply is supply]
            net_sales = Money.zero()
            for invoice in sales:
                net_sales = net_sales + invoice.net
            vat_sales = Money.zero()
            for invoice in sales:
                vat_sales = vat_sales + invoice.vat
            vat_buys = Money.zero()
            for invoice in buys:
                vat_buys = vat_buys + invoice.vat
            rows.append({
                "supply": str(supply),
                "net sales": net_sales.amount,
                "output VAT": vat_sales.amount,
                "input VAT": vat_buys.amount,
                "input recoverable": supply.input_recoverable,
            })
        return VatReturn(output_vat=output, input_vat_recoverable=recoverable,
                         input_vat_blocked=blocked,
                         net_payable=output - recoverable,
                         by_supply=pd.DataFrame(rows))

    def calculate(self, ctx: TaxContext) -> TaxResult:
        """The Tax interface: VAT for the invoices on the context."""
        invoices = tuple(ctx.invoices)
        completed = self.prepare_return(invoices)
        taxable_sales = Money.zero()
        for invoice in invoices:
            if not invoice.is_purchase and invoice.supply is SupplyType.STANDARD:
                taxable_sales = taxable_sales + invoice.net

        workings = (
            Workings("output VAT on standard-rated sales", taxable_sales,
                     self.rate, completed.output_vat),
            Workings("recoverable input VAT", completed.input_vat_recoverable,
                     Decimal("-1"), -completed.input_vat_recoverable),
        )
        notes = [f"input VAT of {completed.input_vat_blocked} is blocked as "
                 f"attributable to exempt supplies"]
        if completed.is_refund:
            notes.append("net position is a refund, not a payment")
        # A refund is a negative payable; TaxResult forbids that, so report the
        # magnitude and record the direction in the notes.
        return TaxResult(tax=self.name, chargeable=taxable_sales,
                         payable=abs(completed.net_payable),
                         workings=workings, notes=tuple(notes))
