# %% [markdown]
# # Mini-Project 2: Uganda Tax Engine
#
# **What this is.** An extensible, testable calculator for the taxes URA
# administers: PAYE, NSSF, Local Service Tax, VAT and corporate income tax,
# plus withholding and rental income tax as the extension.
#
# **Where the code lives.** `src/tax/`: `money.py` (the `Money` value object),
# `core.py` (the abstract `Tax`, `TaxContext` and `TaxResult`), `paye.py`,
# `payroll.py`, `vat.py` and `corporate.py`. This notebook exercises them.
#
# **Rates used.** The baseline table in the brief, cross-checked against the
# Income Tax Act Cap 340 Third Schedule as published by URA. Every schedule
# carries its own `source` string, printed below, and schedules are
# effective-dated so a figure can always be traced to the rules in force on
# the day it was calculated.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import datetime as dt
from decimal import Decimal

import pandas as pd
import matplotlib.pyplot as plt

from src.weather.display import PALETTE, house_style, save
from src.tax.core import Residency, Tax, TaxContext, TaxResult
from src.tax.corporate import (CorporateIncomeTax, Expense, PaymentType,
                               RentalIncomeTax, WithholdingTax, WITHHOLDING_RATES)
from src.tax.errors import (InvalidIncomeError, InvalidRateError,
                            NoScheduleForDateError, TaxError, UnknownSupplyError)
from src.tax.money import Money, float_is_unsafe
from src.tax.paye import PayeTax, SCHEDULES, ScheduleRegistry
from src.tax.payroll import (LocalServiceTax, NssfContribution, PayeDeduction,
                             PayrollRun, lst_for_income)
from src.tax.vat import Invoice, SupplyType, VatTax

house_style()
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.2f}".format)

PERIOD = dt.date(2026, 7, 1)
print(f"calculating for the month beginning {PERIOD.isoformat()}")

# %% [markdown]
# ## The rate schedules, and their sources
#
# Rates are data, not constants buried in a function. Each schedule records
# where it came from and when it took effect.

# %%
for schedule in SCHEDULES:
    print(f"{schedule}")
    for band in schedule:
        print(f"    {band}")
    print(f"    plus {schedule.surcharge_rate:.0%} on income above "
          f"{schedule.surcharge_threshold}")
    print(f"    source: {schedule.source}\n")

# %% [markdown]
# ## Why `float` is unsafe here
#
# The brief asks for a concrete failing example. There are three, and each one
# below was checked to actually fail rather than being asserted.

# %%
evidence = float_is_unsafe()

print("1. ROUNDING GOES THE WRONG WAY")
print(f"   round(2.675, 2)  -> {evidence['round_float']}")
print(f"   the exact answer -> {evidence['round_exact']}")
print(f"   wrong?           -> {evidence['round_wrong']}")
print(f"   because 2.675 is actually stored as:")
print(f"     {evidence['round_stored_as']}")
print("   The stored double sits just BELOW 2.675, so round-half-up rounds it")
print("   DOWN. For a tax figure that is a wrong shilling with no diagnosable")
print("   cause: the decimal digits say .675 and the answer says .67.")

print("\n2. ADDITION IS NOT ASSOCIATIVE")
print(f"   (0.1 + 0.2) + 0.3 = {evidence['assoc_left']!r}")
print(f"   0.1 + (0.2 + 0.3) = {evidence['assoc_right']!r}")
print(f"   equal?             {evidence['assoc_equal']}")
print("   A payslip sums several deductions. In float the total depends on the")
print("   ORDER they are added, so two correct implementations can disagree and")
print("   neither is wrong.")

print("\n3. ERROR ACCUMULATES")
print(f"   {evidence['repeats']:,} applications of a rate:")
print(f"   float   -> {evidence['float_total']!r}")
print(f"   Decimal -> {evidence['decimal_total']}")
print(f"   drift   -> {evidence['drift']} UGX")

print(f"\n   and the classic: {evidence['classic']}  ({evidence['classic_value']})")

# %%
print("So Money refuses float at the door, loudly:")
for attempt, label in ((lambda: Money(335000.50), "Money(335000.50)"),
                       (lambda: Money("1000") * 0.3, "Money('1000') * 0.3"),
                       (lambda: Money("1000") * Money("2"), "Money * Money")):
    try:
        attempt()
        print(f"  {label:<24} -> unexpectedly allowed")
    except TypeError as problem:
        print(f"  {label:<24} -> TypeError: {str(problem)[:70]}")

print("\nand accepts only exact representations:")
print(f"  Money(335000) == Money('335000') == Money(Decimal('335000')): "
      f"{Money(335000) == Money('335000') == Money(Decimal('335000'))}")

# %% [markdown]
# ## 1. PAYE calculator
#
# ### The band edges the brief names

# %%
paye = PayeTax()
edges = [335_000, 335_001, 410_000, 485_000, 10_000_000, 10_000_001]
rows = []
for gross in edges:
    result = paye.calculate(TaxContext(period=PERIOD, gross=Money(gross)))
    rows.append({"monthly gross": gross,
                 "PAYE": int(result.payable.amount),
                 "effective rate": float(result.effective_rate)})
boundary = pd.DataFrame(rows).set_index("monthly gross")
boundary["effective rate"] = boundary["effective rate"].map("{:.4%}".format)
boundary

# %% [markdown]
# Two of these deserve a word.
#
# **335,001 charges nothing.** Twenty per cent of one shilling is 0.2, which
# rounds to zero. That is correct, not a bug: the nil band ends at 335,000 and
# the charge grows continuously from there.
#
# **10,000,001 charges the same as 10,000,000.** The additional 10% applies to
# one shilling of excess, which is 10 cents, which rounds away. Again correct —
# and the check that matters is that the *marginal rate* above the threshold
# really is 40%, not 30%.

# %%
at_threshold = paye.calculate(TaxContext(period=PERIOD,
                                         gross=Money(10_000_000))).payable
above = paye.calculate(TaxContext(period=PERIOD, gross=Money(11_000_000))).payable
extra = above - at_threshold
print(f"tax at 10,000,000      : {at_threshold}")
print(f"tax at 11,000,000      : {above}")
print(f"extra on the 1,000,000 : {extra}")
print(f"marginal rate          : {extra.amount / 1_000_000:.0%}  "
      f"(30% band + 10% additional charge)")
assert extra == Money("400000")

# %% [markdown]
# ### The arithmetic, shown

# %%
print(paye.calculate(TaxContext(period=PERIOD, gross=Money(12_000_000))).explain())

# %% [markdown]
# ### Resident against non-resident
#
# The brief supplies only the resident schedule. The non-resident one comes
# from URA's published monthly tables: no nil band, and 10% from the first
# shilling. Both carry the additional charge above 10,000,000.

# %%
incomes = [200_000, 335_000, 500_000, 1_000_000, 3_000_000, 12_000_000]
comparison = pd.DataFrame({
    "resident": [int(paye.calculate(TaxContext(period=PERIOD, gross=Money(g)))
                     .payable.amount) for g in incomes],
    "non-resident": [int(paye.calculate(TaxContext(
        period=PERIOD, gross=Money(g),
        residency=Residency.NON_RESIDENT)).payable.amount) for g in incomes],
}, index=incomes)
comparison.index.name = "monthly gross"
comparison["difference"] = comparison["non-resident"] - comparison["resident"]
comparison

# %%
fine = list(range(0, 15_000_001, 50_000))
curves = {}
for status in (Residency.RESIDENT, Residency.NON_RESIDENT):
    curves[status] = [float(paye.calculate(TaxContext(
        period=PERIOD, gross=Money(g), residency=status)).effective_rate)
        for g in fine]

figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for index, (status, rates) in enumerate(curves.items()):
    axes[0].plot([g / 1e6 for g in fine], [r * 100 for r in rates],
                 color=PALETTE[index], lw=1.8, label=status)
axes[0].axvline(10, color="0.4", ls="--", lw=1, label="10m threshold")
axes[0].set_xlabel("monthly gross (UGX millions)")
axes[0].set_ylabel("effective rate (%)")
axes[0].set_title("Effective rate rises towards, but never reaches, the margin")
axes[0].legend(fontsize=8)

marginal = [(curves[Residency.RESIDENT][i + 1] * fine[i + 1]
             - curves[Residency.RESIDENT][i] * fine[i]) / 50_000
            for i in range(len(fine) - 1)]
axes[1].step([g / 1e6 for g in fine[1:]], [m * 100 for m in marginal],
             color=PALETTE[2], lw=1.6, where="pre")
axes[1].axvline(10, color="0.4", ls="--", lw=1)
axes[1].set_xlabel("monthly gross (UGX millions)")
axes[1].set_ylabel("marginal rate (%)")
axes[1].set_title("Marginal rate: 0, 20, 25, 30, then 40 above 10m")
axes[1].set_ylim(-2, 45)

figure.tight_layout()
print("saved:", save(figure, "mp2_paye_rates.png").name)
plt.show()

# %% [markdown]
# ### Effective-dated schedules
#
# `NoScheduleForDateError` exists because rates change. Asking for a date no
# schedule covers is a gap in the data, and the engine says so instead of
# quietly applying whichever schedule happens to be nearest.

# %%
try:
    paye.calculate(TaxContext(period=dt.date(2000, 1, 1), gross=Money("500000")))
except NoScheduleForDateError as problem:
    print(f"{type(problem).__name__}: {problem}")

print("\nAdding a future schedule is a data change, not a code change:")
registry = ScheduleRegistry()
print(f"  {registry!r}")
print(f"  schedule in force on 2026-07-01: "
      f"{registry.for_date(PERIOD, Residency.RESIDENT)}")

# %% [markdown]
# ## 2. Payroll pipeline
#
# ### The order of deductions, and the justification
#
# | Step | Line | Charged before PAYE? | Reduces net pay? |
# |---|---|---|---|
# | 1 | Local Service Tax | **yes** | yes |
# | 2 | PAYE | – | yes |
# | 3 | NSSF employee 5% | no | yes |
# | – | NSSF employer 10% | no | **no** |
#
# **Why this order.** LST is an allowable deduction against employment income,
# so PAYE is charged on gross *less* LST — the brief states this directly and
# it matches the Local Governments Act treatment. NSSF is the step most often
# got wrong: the employee's 5% is computed on **gross**, and Uganda does *not*
# allow it as a deduction against employment income the way several
# neighbouring jurisdictions do. PAYE and NSSF are therefore charged on
# overlapping bases and neither reduces the other. The employer's 10% never
# touches net pay at all; it is a cost of employment borne on top of the wage,
# so it is reported separately rather than deducted.

# %%
run = PayrollRun()
run.order()

# %%
ctx = TaxContext(period=PERIOD, gross=Money("2500000"),
                 allowances=Money("300000"))
slip = run.run(ctx)
print(slip.explain())

# %%
slip.to_frame()

# %% [markdown]
# ### What the ordering is worth
#
# Applying LST before PAYE instead of after is not a rounding detail.

# %%
without_lst = PayrollRun(deductions=[PayeDeduction(), NssfContribution()]).run(ctx)
print(f"PAYE with LST deducted first : {slip.paye.payable}")
print(f"PAYE if LST were ignored     : {without_lst.paye.payable}")
print(f"difference                   : "
      f"{without_lst.paye.payable - slip.paye.payable}")
print(f"\nLST itself was              : {dict(slip.lines)['Local Service Tax']}")
print("At the 30% marginal rate, deducting LST first returns 30% of it through")
print("a lower PAYE charge. Getting the order wrong overstates the tax.")

# %%
grosses = [500_000, 1_000_000, 2_000_000, 5_000_000, 12_000_000]
ladder = []
for gross in grosses:
    slip_n = run.run(TaxContext(period=PERIOD, gross=Money(gross)))
    lines = dict(slip_n.lines)
    ladder.append({
        "gross": gross,
        "LST": int(lines["Local Service Tax"].amount),
        "PAYE": int(lines["PAYE"].amount),
        "NSSF 5%": int(lines["NSSF employee (5%)"].amount),
        "net pay": int(slip_n.net.amount),
        "take-home": float(slip_n.take_home_rate),
        "employer cost": int(slip_n.employer_cost.amount),
    })
ladder = pd.DataFrame(ladder).set_index("gross")
ladder["take-home"] = ladder["take-home"].map("{:.1%}".format)
ladder

# %% [markdown]
# ## 3. VAT module
#
# ### The distinction that matters: zero-rated is not exempt
#
# Both produce no output VAT, so they look identical on a sales report. They
# are not. A zero-rated supply is a **taxable** supply charged at 0%, so input
# VAT incurred in making it stays recoverable. An exempt supply is outside the
# VAT net, so input VAT attributable to it is **blocked** and becomes a real
# cost. Treating them the same way always overstates the refund.

# %%
pd.DataFrame([{"supply": str(s), "output rate": f"{s.output_rate:.0%}",
               "input VAT recoverable": s.input_recoverable}
              for s in SupplyType])

# %%
invoices = [
    Invoice("INV-001 export of coffee", Money("4000000"), SupplyType.ZERO_RATED),
    Invoice("INV-002 local sales", Money("10000000"), SupplyType.STANDARD),
    Invoice("INV-003 financial services", Money("3000000"), SupplyType.EXEMPT),
    Invoice("PUR-010 packaging", Money("5000000"), SupplyType.STANDARD,
            is_purchase=True),
    Invoice("PUR-011 export logistics", Money("2000000"), SupplyType.ZERO_RATED,
            is_purchase=True, vat_charged=Money("360000")),
    Invoice("PUR-012 branch costs", Money("1500000"), SupplyType.EXEMPT,
            is_purchase=True, vat_charged=Money("270000")),
]
for invoice in invoices:
    print(f"  {invoice}")

# %%
vat = VatTax()
completed = vat.prepare_return(invoices)
print(completed.explain())
print()
completed.by_supply

# %%
print("If exempt supplies were wrongly treated as zero-rated, the blocked")
print(f"input VAT of {completed.input_vat_blocked} would have been reclaimed,")
print(f"turning a payment of {completed.net_payable} into")
print(f"{completed.net_payable - completed.input_vat_blocked} — an understatement")
print("of the liability, and the kind of error an audit finds.")

# %% [markdown]
# ## 4. Corporate income tax
#
# Chargeable income is revenue less **allowable** deductions. The word is
# doing the work: a disallowed expense still reduces accounting profit while
# leaving taxable profit untouched, so the two figures diverge and the
# difference has to be added back.

# %%
expenses = [
    Expense("staff costs", Money("40000000")),
    Expense("rent", Money("12000000")),
    Expense("depreciation (accounting)", Money("8000000"), allowable=False),
    Expense("URA penalties", Money("2000000"), allowable=False),
]
for expense in expenses:
    print(f"  {expense}")

cit = CorporateIncomeTax()
result = cit.calculate(TaxContext(period=PERIOD, revenue=Money("100000000"),
                                  extras={"expenses": expenses}))
print()
print(result.explain())

# %%
accounting_profit = Money("100000000")
for expense in expenses:
    accounting_profit = accounting_profit - expense.amount
print(f"accounting profit : {accounting_profit}")
print(f"chargeable income : {result.chargeable}")
print(f"difference        : {result.chargeable - accounting_profit} "
      f"(the disallowed expenses, added back)")

# %%
print("a loss produces no charge, not a negative one:")
loss = cit.calculate(TaxContext(period=PERIOD, revenue=Money("10000000"),
                                deductions=Money("25000000")))
print(f"  chargeable {loss.chargeable}, payable {loss.payable}")
print(f"  note: {loss.notes[-1]}")

# %% [markdown]
# ## Extension A: withholding tax and rental income tax
#
# The brief offers two extensions; this build implements **option A**.

# %%
pd.DataFrame([{"payment type": str(kind), "resident rate": f"{rate:.0%}"}
              for kind, rate in WITHHOLDING_RATES.items()])

# %%
print("withholding on a professional fee:")
print(WithholdingTax(PaymentType.PROFESSIONAL_FEES).calculate(
    TaxContext(period=PERIOD, revenue=Money("5000000"))).explain())

print("\nand the goods-and-services threshold in action:")
for amount in ("500000", "2000000"):
    outcome = WithholdingTax(PaymentType.GOODS_AND_SERVICES).calculate(
        TaxContext(period=PERIOD, revenue=Money(amount)))
    print(f"  payment {Money(amount)} -> withheld {outcome.payable}")

# %%
rental = RentalIncomeTax()
rents = ["2000000", "3500000", "10000000", "20000000", "50000000"]
pd.DataFrame([{
    "annual rent": int(Money(r).amount),
    "chargeable": int(rental.calculate(
        TaxContext(period=PERIOD, revenue=Money(r))).chargeable.amount),
    "tax": int(rental.calculate(
        TaxContext(period=PERIOD, revenue=Money(r))).payable.amount),
} for r in rents]).set_index("annual rent")

# %%
print(rental.calculate(TaxContext(period=PERIOD,
                                  revenue=Money("20000000"))).explain())

# %% [markdown]
# ## One abstraction, five taxes
#
# Every tax above satisfies the same contract: `calculate(ctx) -> TaxResult`.
# That is what lets a report loop over them rather than special-casing each.

# %%
shared = TaxContext(period=PERIOD, gross=Money("3000000"),
                    revenue=Money("50000000"), deductions=Money("20000000"),
                    invoices=tuple(invoices))
summary = []
for tax in (PayeTax(), VatTax(), CorporateIncomeTax(),
            WithholdingTax(PaymentType.PROFESSIONAL_FEES), RentalIncomeTax()):
    outcome = tax.calculate(shared)
    summary.append({"tax": outcome.tax,
                    "chargeable": int(outcome.chargeable.amount),
                    "payable": int(outcome.payable.amount),
                    "effective rate": f"{outcome.effective_rate:.2%}"})
pd.DataFrame(summary).set_index("tax")

# %% [markdown]
# ## Validation: nothing fails silently

# %%
checks = [
    ("negative pay", lambda: TaxContext(period=PERIOD, gross=Money("-1"))),
    ("unknown residency", lambda: TaxContext(period=PERIOD, gross=Money("1"),
                                             residency="martian")),
    ("a float rate", lambda: VatTax(rate=0.18)),
    ("an unknown supply type", lambda: Invoice("X", Money("100"),
                                               supply="standard")),
    ("a negative expense", lambda: Expense("odd", Money("-1"))),
    ("an uncovered date", lambda: PayeTax().calculate(
        TaxContext(period=dt.date(1999, 1, 1), gross=Money("500000")))),
]
for label, attempt in checks:
    try:
        attempt()
        print(f"  {label:<24} -> NOT CAUGHT")
    except (TaxError, ValueError, TypeError) as problem:
        print(f"  {label:<24} -> {type(problem).__name__}: "
              f"{str(problem)[:60]}")

# %% [markdown]
# ## Discussion
#
# ### 1. Why is `float` unsafe for tax computation?
#
# Because binary floating point cannot represent most decimal fractions, and
# tax is defined in decimal. The demonstration above shows three distinct
# consequences, and the first is the one that would actually reach a payslip:
# `round(2.675, 2)` returns `2.67`, because the nearest double to 2.675 is
# fractionally *below* it. A round-half-up rule applied to a float therefore
# rounds half *down* whenever the stored value happens to fall on the low side,
# and which way it falls cannot be predicted from the decimal digits. The
# result is a figure that is wrong by one unit, inconsistently, with no
# diagnosable cause — the worst possible failure mode in a system whose output
# a person is legally obliged to pay.
#
# The second consequence is subtler and affects the payroll pipeline directly.
# Float addition is not associative: `(0.1 + 0.2) + 0.3` and
# `0.1 + (0.2 + 0.3)` are different doubles. A payslip sums LST, PAYE and NSSF;
# in float, the total would depend on the order the lines were added, so two
# faithful implementations of the same documented rules could disagree and
# neither would be at fault. The third is accumulation: applying a rate
# repeatedly drifts from the exact answer, and the drift grows with the number
# of operations, so the error is worst in exactly the high-volume payroll runs
# where it is least likely to be spotted.
#
# `Decimal` fixes all three because it stores decimal digits and lets the
# rounding rule be stated explicitly and not inherited from the hardware.
# `Money` therefore refuses `float` at construction and refuses `float` rates
# in multiplication, raising `TypeError` with an explanation. Refusing is
# better than coercing: `Money(335000.50)` could be silently accepted as
# `Money('335000')`, and a system that quietly discards half a shilling is
# worse than one that stops.
#
# ### 2. Where is inheritance justified here, and where was it deliberately avoided?
#
# **Justified: the `Tax` hierarchy.** `PayeTax`, `VatTax`,
# `CorporateIncomeTax`, `WithholdingTax` and `RentalIncomeTax` all subclass
# `Tax`. What they share is not behaviour but a *contract*:
# `calculate(ctx) -> TaxResult`. That is exactly the case inheritance exists
# for — the base class holds no arithmetic at all, because the five do
# genuinely different things (one walks progressive bands, one nets two running
# totals, three apply a flat rate to different bases). The payoff is concrete
# and visible in the cell above: a report can loop over all five, and the
# payroll pipeline can accept any `Tax` for its PAYE step, because the type
# guarantees the shape of the answer.
#
# **Justified: `Deduction`.** The payroll lines share a real contract —
# "given a context and a running base, how much do you take" — and the pipeline
# applies them in order without knowing what any of them is. Adding a new
# statutory deduction means adding a class, not editing the run method.
#
# **Deliberately avoided: `RentalIncomeTax` does not extend `PayeTax`.** The
# temptation is obvious, since both charge an individual on a progressive-ish
# basis with a threshold, and a quick `class RentalIncomeTax(PayeTax)` would
# have reused the band machinery. It would also have been wrong. Rental income
# is a separate chargeable head under the Act, not another employment band: its
# allowance, threshold and rate are set independently, and a future change to
# the PAYE bands must not silently move the rental calculation. Inheriting
# would have created exactly that coupling, for the sake of code that is four
# lines long.
#
# **Deliberately avoided: `ReplayClient` does not extend `ApiClient`.** In the
# weather project the same judgement arises. The two share an interface
# (`get_json`) but no implementation — one does HTTP with retries, the other
# reads files. Subclassing would have inherited a `requests.Session`, a retry
# policy and a timeout that the replay client has no use for, and would have
# invited someone to call an inherited method that makes a real request from
# what is supposed to be an offline object. Duck typing gives the substitution
# without the baggage.
#
# **Deliberately avoided: `Money` does not extend `Decimal`.** Subclassing
# `Decimal` would have made `Money` arithmetic work "for free", and would have
# meant `Money * Money` silently returning a `Decimal`, `Money + 0.5`
# accepting a float through the inherited operator, and the rounding policy
# being bypassable by any inherited method that returns a bare `Decimal`.
# Composition — holding a `Decimal` and exposing only the operations that make
# sense for money — is what allows the type to *refuse* things, and refusing is
# the entire point of the class.
#
# The pattern across all four decisions is the same: inheritance where the
# subtypes owe the same promise to a caller, composition or duck typing where
# they merely happen to need similar code.

# %% [markdown]
# ## Testing
#
# The suite in `tests/test_tax.py` covers the boundary cases the brief lists
# explicitly and the property-based checks it asks for with `hypothesis`:
# that tax is non-negative, monotonic in income, and never exceeds income.
# The cell below re-runs the three properties inline over a wide range.

# %%
import random

random.seed(2026)
probe = sorted(random.sample(range(0, 60_000_000), 400) + edges)

non_negative = all(not paye.calculate(
    TaxContext(period=PERIOD, gross=Money(g))).payable.is_negative() for g in probe)
never_exceeds = all(paye.calculate(
    TaxContext(period=PERIOD, gross=Money(g))).payable <= Money(g) for g in probe)
charges = [paye.calculate(TaxContext(period=PERIOD, gross=Money(g))).payable
           for g in probe]
monotonic = all(b >= a for a, b in zip(charges, charges[1:]))

print(f"over {len(probe)} incomes from 0 to 60,000,000:")
print(f"  tax is never negative   : {non_negative}")
print(f"  tax never exceeds income: {never_exceeds}")
print(f"  tax is monotonic        : {monotonic}")
assert non_negative and never_exceeds and monotonic
print("\n(hypothesis checks the same three properties over generated inputs in")
print(" tests/test_tax.py, which is stricter than this fixed sample)")
