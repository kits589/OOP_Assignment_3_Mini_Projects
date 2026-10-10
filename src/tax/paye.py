"""PAYE: progressive bands, resident and non-resident, effective-dated.

The resident monthly schedule is the brief's. The non-resident one is not in
the brief and comes from URA's published monthly tables: no nil band, 10%
from the first shilling. Both carry the extra 10% above UGX 10,000,000, which
puts the top marginal rate at 40% rather than 30%.

Schedules are dated because rates change. A date no schedule covers raises
NoScheduleForDateError rather than falling back to the nearest one.
"""
from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterator, Sequence

from src.tax.core import Residency, Tax, TaxContext, TaxResult, Workings
from src.tax.errors import InvalidIncomeError, InvalidRateError, NoScheduleForDateError
from src.tax.money import Money

__all__ = ["Band", "PayeSchedule", "PayeTax", "SCHEDULES", "ScheduleRegistry",
           "SURCHARGE_RATE", "SURCHARGE_THRESHOLD"]

# The additional charge above UGX 10,000,000 per month.
SURCHARGE_THRESHOLD = Money("10000000")
SURCHARGE_RATE = Decimal("0.10")


@dataclass(frozen=True)
class Band:
    """One progressive band: a rate applying above lower up to upper."""

    lower: Money
    upper: Money | None
    rate: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.rate, Decimal):
            raise InvalidRateError(self.rate, "band rate must be a Decimal")
        if not Decimal("0") <= self.rate <= Decimal("1"):
            raise InvalidRateError(self.rate)
        if self.upper is not None and self.upper <= self.lower:
            raise ValueError(f"band upper {self.upper} must exceed lower {self.lower}")

    @property
    def width(self) -> Money | None:
        """How much income the band spans; None when open-ended."""
        return None if self.upper is None else self.upper - self.lower

    def slice_of(self, income: Money) -> Money:
        """How much of income falls inside this band."""
        if income <= self.lower:
            return Money.zero()
        top = income if self.upper is None else min(income, self.upper)
        return top - self.lower

    def charge(self, income: Money) -> Money:
        """Tax contributed by this band."""
        return self.slice_of(income) * self.rate

    def __str__(self) -> str:
        ceiling = "upwards" if self.upper is None else f"to {self.upper}"
        return f"{self.lower} {ceiling} at {self.rate:.0%}"


@dataclass(frozen=True)
class PayeSchedule:
    """A complete set of bands, valid from a date."""

    residency: str
    effective_from: dt.date
    bands: tuple[Band, ...]
    source: str = ""
    surcharge_threshold: Money = SURCHARGE_THRESHOLD
    surcharge_rate: Decimal = SURCHARGE_RATE

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("a schedule needs at least one band")
        if self.bands[0].lower != Money.zero():
            raise ValueError("the first band must start at zero")
        for earlier, later in zip(self.bands, self.bands[1:]):
            if earlier.upper is None:
                raise ValueError("only the final band may be open-ended")
            if earlier.upper != later.lower:
                raise ValueError(
                    f"bands must be contiguous: {earlier.upper} != {later.lower}")
        if self.bands[-1].upper is not None:
            raise ValueError("the final band must be open-ended")

    def __iter__(self) -> Iterator[Band]:
        return iter(self.bands)

    def __len__(self) -> int:
        return len(self.bands)

    def __str__(self) -> str:
        return (f"{self.residency} schedule from "
                f"{self.effective_from.isoformat()} ({len(self)} bands)")

    def charge(self, income: Money) -> tuple[Money, tuple[Workings, ...]]:
        """Total tax and the band-by-band workings.

        Each band is applied to its own slice, so the total is a sum of
        marginal charges and not a lookup in a cumulative table. Both give the
        same answer, but this way the workings explain themselves and editing
        one band cannot invalidate a hard-coded base figure.
        """
        if income.is_negative():
            raise InvalidIncomeError(income)
        total, workings = Money.zero(), []
        for band in self.bands:
            slice_ = band.slice_of(income)
            if slice_.is_zero():
                continue
            amount = slice_ * band.rate
            total = total + amount
            workings.append(Workings(f"band {band}", slice_, band.rate, amount))

        if income > self.surcharge_threshold:
            excess = income - self.surcharge_threshold
            extra = excess * self.surcharge_rate
            total = total + extra
            workings.append(Workings(
                f"additional charge above {self.surcharge_threshold}",
                excess, self.surcharge_rate, extra))
        return total, tuple(workings)


def _band(lower: str, upper: str | None, rate: str) -> Band:
    return Band(Money(lower), None if upper is None else Money(upper), Decimal(rate))


URA_RESIDENT = (
    "URA monthly PAYE rates for resident individuals, as set out in the "
    "assignment brief and published at ura.go.ug (Income Tax Act Cap 340, "
    "Third Schedule). Rates confirmed against the brief's baseline table.")

URA_NON_RESIDENT = (
    "URA monthly PAYE rates for non-resident individuals: no nil band, 10% "
    "from the first shilling (Income Tax Act Cap 340, Third Schedule Part II). "
    "Not supplied in the brief; sourced from URA's published monthly tables.")

# The schedules this engine knows about, newest last.
SCHEDULES: tuple[PayeSchedule, ...] = (
    PayeSchedule(
        residency=Residency.RESIDENT,
        effective_from=dt.date(2012, 7, 1),
        bands=(_band("0", "335000", "0.00"),
               _band("335000", "410000", "0.20"),
               _band("410000", "485000", "0.25"),
               _band("485000", None, "0.30")),
        source=URA_RESIDENT),
    PayeSchedule(
        residency=Residency.NON_RESIDENT,
        effective_from=dt.date(2012, 7, 1),
        bands=(_band("0", "335000", "0.10"),
               _band("335000", "410000", "0.20"),
               _band("410000", "485000", "0.25"),
               _band("485000", None, "0.30")),
        source=URA_NON_RESIDENT),
)


class ScheduleRegistry:
    """Finds the schedule in force for a residency on a date.

    Held per residency, sorted by effective date, selected with bisect_right.
    Adding a 2027 schedule is a data change, not a code change.
    """

    def __init__(self, schedules: Sequence[PayeSchedule] = SCHEDULES) -> None:
        if not schedules:
            raise ValueError("the registry needs at least one schedule")
        self._by_residency: dict[str, list[PayeSchedule]] = {}
        for schedule in schedules:
            self._by_residency.setdefault(schedule.residency, []).append(schedule)
        for group in self._by_residency.values():
            group.sort(key=lambda s: s.effective_from)

    def __repr__(self) -> str:
        counts = {k: len(v) for k, v in self._by_residency.items()}
        return f"ScheduleRegistry({counts})"

    def __len__(self) -> int:
        return sum(len(group) for group in self._by_residency.values())

    def known_dates(self, residency: str) -> list[str]:
        """Effective dates on file, for the error message when none matches."""
        return [s.effective_from.isoformat()
                for s in self._by_residency.get(residency, [])]

    def for_date(self, when: dt.date, residency: str) -> PayeSchedule:
        """The schedule in force on when."""
        residency = Residency.check(residency)
        group = self._by_residency.get(residency)
        if not group:
            raise NoScheduleForDateError(when, residency,
                                         sorted(self._by_residency))
        starts = [s.effective_from for s in group]
        index = bisect.bisect_right(starts, when) - 1
        if index < 0:
            raise NoScheduleForDateError(when, residency,
                                         self.known_dates(residency))
        return group[index]


class PayeTax(Tax):
    """Pay As You Earn on monthly employment income."""

    name = "PAYE"

    def __init__(self, registry: ScheduleRegistry | None = None) -> None:
        self.registry = registry or ScheduleRegistry()

    def calculate(self, ctx: TaxContext) -> TaxResult:
        """PAYE for one month.

        Chargeable is gross plus allowances, less anything the caller already
        took off and passed in extras['pre_paye_deductions']. That is how Local
        Service Tax reaches this calculation without PAYE needing to know what
        LST is.
        """
        schedule = self.registry.for_date(ctx.period, ctx.residency)
        chargeable = ctx.chargeable_employment_income
        pre = ctx.extras.get("pre_paye_deductions")
        if pre is not None:
            chargeable = chargeable - Money(pre)
            if chargeable.is_negative():
                chargeable = Money.zero()

        payable, workings = schedule.charge(chargeable)
        notes = [str(schedule), schedule.source]
        if pre is not None:
            notes.insert(0, f"{Money(pre)} deducted before PAYE")
        return TaxResult(tax=self.name, chargeable=chargeable, payable=payable,
                         workings=workings, notes=tuple(notes))
