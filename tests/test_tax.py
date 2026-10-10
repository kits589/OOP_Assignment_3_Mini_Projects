"""Tests for Mini-Project 2.

Covers the boundary cases the brief lists explicitly, the property-based
checks it asks for with hypothesis, and the behaviours the design
constraints imply -- in particular that Money refuses float and that
an uncovered date raises rather than guessing a schedule.
"""
import datetime as dt
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.tax.core import Residency, Tax, TaxContext, TaxResult, Workings
from src.tax.corporate import (CorporateIncomeTax, Expense, PaymentType,
                               RentalIncomeTax, WithholdingTax)
from src.tax.errors import (InvalidIncomeError, InvalidRateError,
                            NoScheduleForDateError, TaxError, UnknownSupplyError)
from src.tax.money import Money, float_is_unsafe
from src.tax.paye import (Band, PayeSchedule, PayeTax, SCHEDULES, ScheduleRegistry,
                          SURCHARGE_THRESHOLD)
from src.tax.payroll import (LocalServiceTax, NssfContribution, PayeDeduction,
                             PayrollRun, lst_for_income)
from src.tax.vat import Invoice, SupplyType, VatTax

PERIOD = dt.date(2026, 7, 1)


def resident(gross) -> TaxContext:
    return TaxContext(period=PERIOD, gross=Money(gross))


def non_resident(gross) -> TaxContext:
    return TaxContext(period=PERIOD, gross=Money(gross),
                      residency=Residency.NON_RESIDENT)


# ======================================================================
# Money
# ======================================================================
class TestMoney:
    """The value object the whole engine is built on."""

    def test_float_is_refused_loudly(self):
        """The single most important constraint in the brief."""
        with pytest.raises(TypeError, match="float"):
            Money(335000.50)
        with pytest.raises(TypeError):
            Money("1000") * 0.3
        with pytest.raises(TypeError):
            Money(True)

    def test_accepts_int_str_and_decimal(self):
        assert Money(335000) == Money("335000") == Money(Decimal("335000"))

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError):
            Money("not an amount")

    def test_operator_overloading_required_by_the_brief(self):
        a, b = Money("335000"), Money("75000")
        assert a + b == Money("410000")               # __add__
        assert a * Decimal("0.2") == Money("67000")   # __mul__
        assert b < a                                  # __lt__
        assert repr(a) == "Money('335000')"           # __repr__

    def test_subtraction_negation_and_absolute(self):
        assert Money("410000") - Money("335000") == Money("75000")
        assert -Money("100") == Money("-100")
        assert abs(Money("-100")) == Money("100")

    def test_multiplying_money_by_money_is_refused(self):
        with pytest.raises(TypeError, match="meaningless"):
            Money("1000") * Money("2")

    def test_rounding_is_half_up_to_whole_shillings(self):
        assert Money("0.5") == Money("1")
        assert Money("1.5") == Money("2")
        assert Money("0.4") == Money("0")
        assert Money("2.675") * Decimal("1") == Money("3")

    def test_comparison_and_hashing(self):
        assert Money("100") <= Money("100") and Money("100") >= Money("100")
        assert Money("100") > Money("99")
        assert len({Money("100"), Money(100), Money("100.0")}) == 1
        assert bool(Money("1")) and not bool(Money.zero())

    def test_sum_of_money_starts_from_zero(self):
        assert sum([Money("1"), Money("2")], Money.zero()) == Money("3")

    def test_the_float_demonstration_actually_fails(self):
        """Not a hypothetical: each case is checked to be a real failure."""
        evidence = float_is_unsafe()
        assert evidence["round_wrong"] is True          # round(2.675,2) -> 2.67
        assert evidence["round_float"] == 2.67
        assert evidence["round_exact"] == Decimal("2.68")
        assert evidence["assoc_equal"] is False         # not associative
        assert evidence["drift"] != 0                   # error accumulates
        assert "0.1 + 0.2 == 0.3 is False" in evidence["classic"]


# ======================================================================
# PAYE: the boundary cases the brief names
# ======================================================================
class TestPayeBoundaries:
    """Every band edge listed in the brief, checked against hand arithmetic."""

    @pytest.mark.parametrize("gross,expected", [
        (335_000, "0"),            # top of the nil band
        (335_001, "0"),            # 20% of 1 shilling, rounds to nothing
        (410_000, "15000"),        # the brief's own "15,000 + ..."
        (485_000, "33750"),        # the brief's own "33,750 + ..."
        (10_000_000, "2888250"),   # 33,750 + 30% of 9,515,000
        (10_000_001, "2888250"),   # the extra 40 cents rounds away
    ])
    def test_resident_band_edges(self, gross, expected):
        assert PayeTax().calculate(resident(gross)).payable == Money(expected)

    def test_nil_band_charges_nothing_throughout(self):
        for gross in (0, 1, 100_000, 334_999, 335_000):
            assert PayeTax().calculate(resident(gross)).payable == Money.zero()

    def test_the_charge_is_continuous_at_every_edge(self):
        """A one-shilling rise must never cause a jump in tax."""
        paye = PayeTax()
        for edge in (335_000, 410_000, 485_000, 10_000_000):
            below = paye.calculate(resident(edge)).payable
            above = paye.calculate(resident(edge + 1)).payable
            assert (above - below) <= Money("1")

    def test_marginal_rate_above_ten_million_is_forty_percent(self):
        """30% band plus the 10% additional charge, not 30% alone."""
        paye = PayeTax()
        at = paye.calculate(resident(10_000_000)).payable
        above = paye.calculate(resident(11_000_000)).payable
        assert (above - at) == Money("400000")

    def test_workings_name_every_band_applied(self):
        result = PayeTax().calculate(resident(12_000_000))
        labels = " ".join(w.label for w in result.workings)
        assert "additional charge" in labels
        # all four bands plus the surcharge; the nil band is shown too, so a
        # reader can see how much income fell in it rather than inferring it
        assert len(result.workings) == 5
        assert result.workings[0].amount.is_zero()
        assert "resident schedule" in " ".join(result.notes)

    @pytest.mark.parametrize("gross,expected", [
        (100_000, "10000"),        # non-residents pay from the first shilling
        (335_000, "33500"),
        (410_000, "48500"),
        (485_000, "67250"),
    ])
    def test_non_resident_band_edges(self, gross, expected):
        assert PayeTax().calculate(non_resident(gross)).payable == Money(expected)

    def test_non_residents_always_pay_at_least_as_much(self):
        paye = PayeTax()
        for gross in (50_000, 335_000, 500_000, 2_000_000, 15_000_000):
            assert (paye.calculate(non_resident(gross)).payable
                    >= paye.calculate(resident(gross)).payable)


# ======================================================================
# PAYE: property-based tests the brief asks for
# ======================================================================
money_amounts = st.integers(min_value=0, max_value=200_000_000)


class TestPayeProperties:
    """The three properties the brief names, checked with hypothesis."""

    @given(gross=money_amounts)
    @settings(max_examples=200, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_tax_is_never_negative(self, gross):
        assert not PayeTax().calculate(resident(gross)).payable.is_negative()

    @given(lower=money_amounts, extra=st.integers(min_value=0, max_value=5_000_000))
    @settings(max_examples=200, deadline=None)
    def test_tax_is_monotonic_in_income(self, lower, extra):
        """Earning more can never leave you paying less tax."""
        paye = PayeTax()
        assert (paye.calculate(resident(lower + extra)).payable
                >= paye.calculate(resident(lower)).payable)

    @given(gross=money_amounts)
    @settings(max_examples=200, deadline=None)
    def test_tax_never_exceeds_income(self, gross):
        """No band structure should ever take more than was earned."""
        assert PayeTax().calculate(resident(gross)).payable <= Money(gross)

    @given(gross=money_amounts)
    @settings(max_examples=100, deadline=None)
    def test_the_same_properties_hold_for_non_residents(self, gross):
        result = PayeTax().calculate(non_resident(gross))
        assert not result.payable.is_negative()
        assert result.payable <= Money(gross)

    @given(gross=st.integers(min_value=1, max_value=200_000_000))
    @settings(max_examples=100, deadline=None)
    def test_the_effective_rate_never_reaches_the_top_marginal_rate(self, gross):
        """A progressive schedule's average rate stays below its margin."""
        result = PayeTax().calculate(resident(gross))
        assert result.effective_rate < Decimal("0.40")


# ======================================================================
# Schedules and dates
# ======================================================================
class TestSchedules:
    def test_a_date_before_every_schedule_raises(self):
        with pytest.raises(NoScheduleForDateError) as caught:
            PayeTax().calculate(TaxContext(period=dt.date(2000, 1, 1),
                                           gross=Money("500000")))
        assert "2012-07-01" in str(caught.value)

    def test_an_unknown_residency_is_rejected(self):
        with pytest.raises(ValueError):
            TaxContext(period=PERIOD, gross=Money("1"), residency="martian")

    def test_the_registry_picks_the_latest_schedule_in_force(self):
        later = PayeSchedule(
            residency=Residency.RESIDENT, effective_from=dt.date(2030, 7, 1),
            bands=(Band(Money.zero(), Money("1000000"), Decimal("0.10")),
                   Band(Money("1000000"), None, Decimal("0.40"))),
            source="hypothetical future schedule")
        registry = ScheduleRegistry(list(SCHEDULES) + [later])
        assert registry.for_date(dt.date(2026, 7, 1),
                                 Residency.RESIDENT).effective_from == dt.date(2012, 7, 1)
        assert registry.for_date(dt.date(2031, 1, 1),
                                 Residency.RESIDENT).effective_from == dt.date(2030, 7, 1)

    def test_bands_must_be_contiguous_and_closed(self):
        with pytest.raises(ValueError, match="contiguous"):
            PayeSchedule(Residency.RESIDENT, PERIOD,
                         (Band(Money.zero(), Money("100"), Decimal("0")),
                          Band(Money("200"), None, Decimal("0.3"))))
        with pytest.raises(ValueError, match="open-ended"):
            PayeSchedule(Residency.RESIDENT, PERIOD,
                         (Band(Money.zero(), Money("100"), Decimal("0")),))
        with pytest.raises(ValueError, match="start at zero"):
            PayeSchedule(Residency.RESIDENT, PERIOD,
                         (Band(Money("100"), None, Decimal("0.3")),))

    @pytest.mark.parametrize("rate", [Decimal("-0.1"), Decimal("1.5"), 0.3, "0.3"])
    def test_band_rates_must_be_sane_decimals(self, rate):
        with pytest.raises(InvalidRateError):
            Band(Money.zero(), None, rate)

    def test_every_shipped_schedule_is_cited(self):
        assert all(schedule.source for schedule in SCHEDULES)


# ======================================================================
# Payroll
# ======================================================================
class TestPayroll:
    def test_the_documented_order_is_what_runs(self):
        order = PayrollRun().order()
        assert list(order["line"]) == ["Local Service Tax", "PAYE",
                                       "NSSF employee (5%)"]
        assert order.loc[0, "charged before PAYE"]        # LST only
        assert not order.loc[2, "charged before PAYE"]    # NSSF is not

    def test_lst_reduces_the_paye_base(self):
        """The whole point of applying it first."""
        ctx = TaxContext(period=PERIOD, gross=Money("2000000"))
        with_lst = PayrollRun().run(ctx)
        without = PayrollRun(deductions=[PayeDeduction(), NssfContribution()]).run(ctx)
        assert with_lst.paye.payable < without.paye.payable

    def test_nssf_is_charged_on_gross_not_on_gross_less_lst(self):
        ctx = TaxContext(period=PERIOD, gross=Money("2000000"))
        slip = PayrollRun().run(ctx)
        nssf = dict(slip.lines)["NSSF employee (5%)"]
        assert nssf == Money("2000000") * Decimal("0.05")

    def test_employer_contribution_never_reduces_net_pay(self):
        slip = PayrollRun().run(TaxContext(period=PERIOD, gross=Money("1000000")))
        assert slip.employer_cost > slip.gross
        assert slip.employer_cost - slip.gross == Money("100000")
        assert slip.net == slip.gross - slip.total_deductions

    def test_the_payslip_adds_up(self):
        slip = PayrollRun().run(TaxContext(period=PERIOD, gross=Money("2500000"),
                                           allowances=Money("300000")))
        taken = sum((amount for _, amount in slip.lines), Money.zero())
        assert slip.gross - taken == slip.net
        assert Decimal("0") < slip.take_home_rate < Decimal("1")

    def test_local_service_tax_can_be_switched_off(self):
        ctx = TaxContext(period=PERIOD, gross=Money("2000000"))
        off = PayrollRun(deductions=[LocalServiceTax(enabled=False),
                                     PayeDeduction(), NssfContribution()]).run(ctx)
        assert dict(off.lines)["Local Service Tax"] == Money.zero()

    def test_local_service_tax_can_be_overridden_with_a_fixed_amount(self):
        ctx = TaxContext(period=PERIOD, gross=Money("2000000"))
        slip = PayrollRun(deductions=[LocalServiceTax(monthly=Money("7500")),
                                      PayeDeduction(), NssfContribution()]).run(ctx)
        assert dict(slip.lines)["Local Service Tax"] == Money("7500")

    @pytest.mark.parametrize("gross,expected_instalment", [
        ("50000", "1250"), ("250000", "5000"), ("2000000", "25000")])
    def test_lst_bands(self, gross, expected_instalment):
        assert lst_for_income(Money(gross)) == Money(expected_instalment)

    def test_nssf_rate_must_be_a_decimal_in_range(self):
        with pytest.raises(InvalidRateError):
            NssfContribution(rate=0.05)
        with pytest.raises(InvalidRateError):
            NssfContribution(rate=Decimal("1.5"))

    def test_negative_pay_is_refused(self):
        with pytest.raises(InvalidIncomeError):
            TaxContext(period=PERIOD, gross=Money("-1"))


# ======================================================================
# VAT
# ======================================================================
class TestVat:
    @pytest.fixture
    def invoices(self):
        return [
            Invoice("S1", Money("10000000"), SupplyType.STANDARD),
            Invoice("S2", Money("4000000"), SupplyType.ZERO_RATED),
            Invoice("S3", Money("3000000"), SupplyType.EXEMPT),
            Invoice("P1", Money("5000000"), SupplyType.STANDARD, is_purchase=True),
            Invoice("P2", Money("2000000"), SupplyType.ZERO_RATED,
                    is_purchase=True, vat_charged=Money("360000")),
            Invoice("P3", Money("1500000"), SupplyType.EXEMPT,
                    is_purchase=True, vat_charged=Money("270000")),
        ]

    def test_output_vat_is_eighteen_percent_of_standard_sales_only(self, invoices):
        assert VatTax().output_vat(invoices) == Money("1800000")

    def test_zero_rated_and_exempt_both_charge_no_output_vat(self):
        assert SupplyType.ZERO_RATED.output_rate == Decimal("0")
        assert SupplyType.EXEMPT.output_rate == Decimal("0")

    def test_but_only_exempt_blocks_input_recovery(self, invoices):
        """The distinction the whole module exists for."""
        assert SupplyType.ZERO_RATED.input_recoverable is True
        assert SupplyType.EXEMPT.input_recoverable is False
        recoverable, blocked = VatTax().input_vat(invoices)
        assert recoverable == Money("1260000")      # 900,000 + 360,000
        assert blocked == Money("270000")

    def test_net_payable_is_output_less_recoverable_input(self, invoices):
        completed = VatTax().prepare_return(invoices)
        assert completed.net_payable == Money("540000")
        assert not completed.is_refund

    def test_a_refund_position_is_reported_as_such(self):
        invoices = [Invoice("S1", Money("1000000"), SupplyType.STANDARD),
                    Invoice("P1", Money("9000000"), SupplyType.STANDARD,
                            is_purchase=True)]
        completed = VatTax().prepare_return(invoices)
        assert completed.is_refund
        result = VatTax().calculate(TaxContext(period=PERIOD, invoices=tuple(invoices)))
        assert not result.payable.is_negative()
        assert any("refund" in note for note in result.notes)

    def test_an_unknown_supply_type_is_rejected(self):
        with pytest.raises(UnknownSupplyError):
            Invoice("X", Money("100"), supply="standard")

    def test_a_negative_invoice_is_rejected(self):
        with pytest.raises(ValueError):
            Invoice("X", Money("-100"))

    def test_the_rate_must_be_a_decimal(self):
        with pytest.raises(InvalidRateError):
            VatTax(rate=0.18)


# ======================================================================
# Corporate, and extension A
# ======================================================================
class TestCorporate:
    def test_flat_thirty_percent_on_chargeable_income(self):
        result = CorporateIncomeTax().calculate(
            TaxContext(period=PERIOD, revenue=Money("100000000"),
                       deductions=Money("40000000")))
        assert result.chargeable == Money("60000000")
        assert result.payable == Money("18000000")
        assert result.effective_rate == Decimal("0.300000")

    def test_disallowed_expenses_are_added_back(self):
        expenses = [Expense("salaries", Money("40000000")),
                    Expense("fines", Money("2000000"), allowable=False)]
        result = CorporateIncomeTax().calculate(
            TaxContext(period=PERIOD, revenue=Money("100000000"),
                       extras={"expenses": expenses}))
        assert result.chargeable == Money("60000000")    # the fine is not deducted
        assert "disallowed" in " ".join(result.notes)

    def test_a_loss_produces_no_charge_rather_than_a_negative_one(self):
        result = CorporateIncomeTax().calculate(
            TaxContext(period=PERIOD, revenue=Money("10000000"),
                       deductions=Money("25000000")))
        assert result.chargeable.is_zero() and result.payable.is_zero()
        assert any("loss" in note for note in result.notes)

    def test_an_expense_cannot_be_negative(self):
        with pytest.raises(InvalidIncomeError):
            Expense("odd", Money("-1"))


class TestExtensionA:
    """Withholding tax and rental income tax."""

    @pytest.mark.parametrize("kind,rate", [
        (PaymentType.PROFESSIONAL_FEES, "0.06"),
        (PaymentType.DIVIDENDS, "0.15"),
        (PaymentType.RENT_TO_RESIDENT, "0.10")])
    def test_withholding_rates(self, kind, rate):
        tax = WithholdingTax(kind)
        result = tax.calculate(TaxContext(period=PERIOD, revenue=Money("10000000")))
        assert result.payable == Money("10000000") * Decimal(rate)

    def test_goods_and_services_withholding_has_a_threshold(self):
        tax = WithholdingTax(PaymentType.GOODS_AND_SERVICES)
        below = tax.calculate(TaxContext(period=PERIOD, revenue=Money("500000")))
        above = tax.calculate(TaxContext(period=PERIOD, revenue=Money("2000000")))
        assert below.payable.is_zero()
        assert above.payable == Money("120000")
        assert any("threshold" in note for note in below.notes)

    def test_rental_tax_below_the_threshold_is_nil(self):
        result = RentalIncomeTax().calculate(
            TaxContext(period=PERIOD, revenue=Money("3000000")))
        assert result.payable.is_zero()

    def test_rental_tax_applies_the_allowance_then_the_threshold(self):
        # 20,000,000 less 20% = 16,000,000; less 2,820,000 = 13,180,000 at 12%
        result = RentalIncomeTax().calculate(
            TaxContext(period=PERIOD, revenue=Money("20000000")))
        assert result.chargeable == Money("13180000")
        assert result.payable == Money("1581600")

    @given(rent=st.integers(min_value=0, max_value=500_000_000))
    @settings(max_examples=100, deadline=None)
    def test_rental_tax_properties(self, rent):
        result = RentalIncomeTax().calculate(
            TaxContext(period=PERIOD, revenue=Money(rent)))
        assert not result.payable.is_negative()
        assert result.payable <= Money(rent)

    def test_an_unknown_payment_type_is_rejected(self):
        with pytest.raises(ValueError):
            WithholdingTax("professional fees")


# ======================================================================
# The abstraction itself
# ======================================================================
class TestTaxAbstraction:
    def test_tax_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Tax()

    def test_every_tax_satisfies_the_same_contract(self):
        taxes = [PayeTax(), VatTax(), CorporateIncomeTax(),
                 WithholdingTax(), RentalIncomeTax()]
        ctx = TaxContext(period=PERIOD, gross=Money("1000000"),
                         revenue=Money("5000000"), deductions=Money("1000000"))
        for tax in taxes:
            assert isinstance(tax, Tax)
            result = tax.calculate(ctx)
            assert isinstance(result, TaxResult)
            assert not result.payable.is_negative()

    def test_a_result_cannot_be_negative(self):
        with pytest.raises(InvalidIncomeError):
            TaxResult("made up", Money("100"), Money("-1"))

    def test_results_explain_themselves(self):
        explanation = PayeTax().calculate(resident(5_000_000)).explain()
        assert "PAYE" in explanation and "effective rate" in explanation

    def test_every_error_descends_from_tax_error(self):
        for error in (InvalidIncomeError, InvalidRateError,
                      NoScheduleForDateError, UnknownSupplyError):
            assert issubclass(error, TaxError)
