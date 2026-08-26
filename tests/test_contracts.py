"""Expiry and underlying normalisation.

The AM-settlement shift is the transformation with no obvious check against
TWS — the screen shows the last trading date and nothing contradicts it — so it
is the one that most needs a test.
"""

from datetime import date

import pytest

from ibkr_risk_mcp import contracts as C

from .conftest import FakeContract


def fop(expiry: str, trading_class: str, **kw) -> FakeContract:
    return FakeContract(
        symbol="ES",
        secType="FOP",
        right="P",
        strike=5800.0,
        lastTradeDateOrContractMonth=expiry,
        tradingClass=trading_class,
        multiplier="50",
        **kw,
    )


class FakeDetails:
    """ContractDetails as TWS server 178 actually returns it for ES options:
    ``realExpirationDate`` alongside the last trading day, and the AM/PM
    settlement visible in ``lastTradeTime`` — 08:30 for the quarterly's Special
    Opening Quotation, 15:00 for a weekly's afternoon close."""

    def __init__(self, contract, real="", last_trade_time="", under_conid=0):
        self.contract = contract
        self.realExpirationDate = real
        self.lastTradeTime = last_trade_time
        self.underConId = under_conid


@pytest.fixture(autouse=True)
def clean_registry():
    C.forget_details()
    yield
    C.forget_details()


class TestExpiryDates:
    def test_am_settled_quarterly_settles_the_next_day(self):
        # 17 December 2026 is a Thursday; the quarterly settles Friday the 18th.
        assert C.settlement_date(fop("20261217", "ES")) == date(2026, 12, 18)

    def test_pm_settled_weekly_settles_on_its_last_trading_day(self):
        assert C.settlement_date(fop("20261218", "EW4")) == date(2026, 12, 18)

    def test_the_two_meet_on_the_same_morning(self):
        """The trap in one assertion: two contracts whose last trading dates
        differ settle at the same time, and pairing them on lastTradeDate splits
        one expiry into two."""
        quarterly = fop("20261217", "ES")
        weekly = fop("20261218", "EW4")
        assert quarterly.lastTradeDateOrContractMonth != weekly.lastTradeDateOrContractMonth
        assert C.settlement_date(quarterly) == C.settlement_date(weekly)

    def test_a_weekly_in_an_am_settled_class_is_not_shifted(self):
        """The ES trading class carries weeklies too. Only the third-Friday
        expirations settle AM, so a Friday that is not the third is left
        alone."""
        # 4 December 2026 is the first Friday.
        assert C.settlement_date(fop("20261204", "ES")) == date(2026, 12, 4)

    def test_equity_options_are_never_shifted(self):
        equity = FakeContract(
            symbol="AAPL",
            secType="OPT",
            right="P",
            strike=200.0,
            lastTradeDateOrContractMonth="20261218",
            tradingClass="AAPL",
            multiplier="100",
        )
        assert C.settlement_date(equity) == date(2026, 12, 18)

    def test_expiry_info_reports_both_dates_and_flags_the_shift(self):
        info = C.expiry_info(fop("20261217", "ES"), asof=date(2026, 8, 14))
        assert info["lastTradeDate"] == "2026-12-17"
        assert info["settlementDate"] == "2026-12-18"
        assert info["amSettled"] is True
        assert info["daysToExpiry"] == (date(2026, 12, 18) - date(2026, 8, 14)).days

    def test_days_to_expiry_counts_to_settlement_not_last_trade(self):
        asof = date(2026, 8, 14)
        am = C.expiry_info(fop("20261217", "ES"), asof=asof)
        pm = C.expiry_info(fop("20261217", "EW3"), asof=asof)
        assert am["daysToExpiry"] == pm["daysToExpiry"] + 1

    def test_years_to_expiry_never_reaches_zero(self):
        """An expired contract prices at intrinsic, but a hard zero would also
        zero every greek, which reads as 'no risk' rather than 'no time'."""
        expired = fop("20200101", "EW1")
        assert C.years_to_expiry(expired, asof=date(2026, 8, 14)) > 0

    def test_date_offset_shortens_the_tenor(self):
        asof = date(2026, 8, 14)
        base = C.years_to_expiry(fop("20261218", "EW4"), asof=asof)
        rolled = C.years_to_expiry(fop("20261218", "EW4"), asof=asof, offset_days=30)
        assert base - rolled == pytest.approx(30 / 365, rel=1e-9)


class TestIBsOwnExpiration:
    """When the contract details have been seen, IB's answer wins over the
    heuristic. ``realExpirationDate`` exists for exactly this."""

    def test_real_expiration_overrides_the_reported_last_trade_date(self):
        contract = fop("20261217", "ES", conId=111)
        C.remember_details(FakeDetails(contract, real="20261218", last_trade_time="08:30:00"))
        assert C.settlement_date(contract) == date(2026, 12, 18)
        assert C.expiry_info(contract)["amSettled"] is True

    def test_am_settlement_is_recognised_even_when_the_date_is_already_right(self):
        """The case live TWS actually returns: the December quarterly reports
        20261218 with an 08:30 last trade time. The date needs no shift and the
        contract is still AM-settled, which is what tells it apart from the
        weekly expiring the same morning."""
        contract = fop("20261218", "ES", conId=222)
        C.remember_details(FakeDetails(contract, real="20261218", last_trade_time="08:30:00"))
        info = C.expiry_info(contract)
        assert info["settlementDate"] == "2026-12-18"
        assert info["amSettled"] is True

    def test_a_pm_weekly_on_the_same_day_is_not_am_settled(self):
        contract = fop("20261218", "EW3", conId=333)
        C.remember_details(FakeDetails(contract, real="20261218", last_trade_time="15:00:00"))
        info = C.expiry_info(contract)
        assert info["settlementDate"] == "2026-12-18"
        assert info["amSettled"] is False

    def test_details_without_a_real_expiration_fall_back_to_the_heuristic(self):
        contract = fop("20261217", "ES", conId=444)
        C.remember_details(FakeDetails(contract, real="", last_trade_time=""))
        assert C.settlement_date(contract) == date(2026, 12, 18)

    def test_an_unseen_contract_still_uses_the_heuristic(self):
        assert C.settlement_date(fop("20261217", "ES", conId=999)) == date(2026, 12, 18)

    def test_details_with_no_conid_are_ignored_rather_than_miskeyed(self):
        C.remember_details(FakeDetails(fop("20261217", "ES", conId=0), real="20270101"))
        assert C.settlement_date(fop("20261217", "ES", conId=0)) == date(2026, 12, 18)


class TestDateParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("20261218", date(2026, 12, 18)),
            ("202612", date(2026, 12, 1)),
            ("20261218 16:15:00", date(2026, 12, 18)),
        ],
    )
    def test_accepted_forms(self, raw, expected):
        assert C.parse_ib_date(raw) == expected

    def test_empty_is_an_error_not_a_default(self):
        with pytest.raises(ValueError):
            C.parse_ib_date("")

    def test_next_business_day_skips_the_weekend(self):
        assert C.next_business_day(date(2026, 8, 14)) == date(2026, 8, 17)  # Fri -> Mon


class TestMultiplier:
    def test_read_from_the_contract(self):
        assert C.contract_multiplier(fop("20261218", "EW4")) == 50.0

    def test_micro_is_not_assumed_to_be_the_full_size(self):
        mes = fop("20261218", "EW4")
        mes.symbol, mes.multiplier = "MES", "5"
        assert C.contract_multiplier(mes) == 5.0

    def test_missing_multiplier_falls_back_to_one(self):
        stock = FakeContract(symbol="AAPL", secType="STK", multiplier="")
        assert C.contract_multiplier(stock) == 1.0


class TestAssetClass:
    @pytest.mark.parametrize(
        "sec_type,expected",
        [
            ("FOP", "option"),
            ("OPT", "option"),
            ("STK", "equity"),
            ("FUT", "future"),
            ("BOND", "bond"),
            ("CASH", "cash"),
            ("WAR", "other"),
        ],
    )
    def test_buckets(self, sec_type, expected):
        assert C.asset_class(sec_type) == expected


class TestLeg:
    def test_sell_is_a_negative_quantity(self):
        assert C.Leg(conid=1, action="SELL", quantity=3).signed_quantity == -3
        assert C.Leg(conid=1, action="BUY", quantity=3).signed_quantity == 3


class TestRiskGroup:
    """Which factor a position responds to, as opposed to what instrument it
    is. `asset_class` cannot answer this: an ES option and a CAD option are both
    `option`, and shocking both by one equity percentage is how a currency
    strangle ends up dominating an equity curve."""

    @pytest.mark.parametrize(
        "sec_type,symbol,expected",
        [
            # IB names the currency, not the exchange ticker: the 6C is CAD.
            ("FUT", "CAD", "fx"),
            ("FOP", "CAD", "fx"),
            ("FOP", "EUR", "fx"),
            ("FUT", "JPY", "fx"),
            ("CASH", "EURUSD", "fx"),
            # Index futures and their options are the equity default.
            ("FUT", "ES", "equity"),
            ("FOP", "ES", "equity"),
            ("FOP", "NQ", "equity"),
            ("STK", "AAPL", "equity"),
            ("OPT", "GOOGL", "equity"),
            # Everything else the table knows about.
            ("FUT", "ZN", "rates"),
            ("FUT", "UB", "rates"),
            ("BOND", "US-T", "rates"),
            ("FUT", "GC", "metals"),
            ("FUT", "CL", "energy"),
        ],
    )
    def test_groups(self, sec_type, symbol, expected):
        contract = FakeContract(secType=sec_type, symbol=symbol)
        assert C.risk_group(contract) == expected

    def test_an_unlisted_futures_root_falls_through_to_equity(self):
        """Right for an index future, wrong for anything exotic — which is why
        the group is reported in every result and can be overridden."""
        assert C.risk_group(FakeContract(secType="FUT", symbol="ZZZ")) == "equity"

    def test_describe_carries_the_group(self):
        """A misclassification has to be visible. IB publishes no asset class
        for a bond or gold ETF quoted as a stock, so the only defence is that
        the guess is printed next to the position."""
        out = C.describe(FakeContract(
            secType="FOP", symbol="CAD", localSymbol="CAUZ6 P6900",
            lastTradeDateOrContractMonth="20261204", tradingClass="CAU",
        ))
        assert out["riskGroup"] == "fx"
