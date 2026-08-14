"""The stress engine, driven off the recorded portfolio.

Every assertion here is about a property of the curve rather than a
remembered number, so the tests survive a change of fixture but not a change
of behaviour.
"""

from datetime import date

import pytest

from ibkr_risk_mcp import stress as S
from ibkr_risk_mcp import pricing

from .conftest import FakeAccountValue

#: Pinned so the fixture's time to expiry — and therefore every price derived
#: from it — is the same today as it will be next year.
ASOF = date(2026, 8, 14)
RATE = 0.04


@pytest.fixture
def units(holdings):
    return [S.unit_from_holding(h, ASOF) for h in holdings]


@pytest.fixture
def shocks():
    return [round(-0.30 + i * 0.01, 4) for i in range(61)]


def curve(units, shocks, **kw):
    kw.setdefault("rate", RATE)
    cfg = S.StressConfig(shocks=shocks, asof=ASOF, **kw)
    cfg.validate()
    skews = {}
    if cfg.vol_mode == "sticky_moneyness":
        # build_skews is async only because it may fetch; the pure path is
        # rebuilt here so the engine can be exercised synchronously.
        grouped = {}
        for u in units:
            if u.asset_class == "option" and u.priceable:
                grouped.setdefault(u.skew_key, []).append(u)
        for key, group in grouped.items():
            if len({u.strike for u in group}) >= 3:
                skews[key] = pricing.VolSkew.from_strikes(
                    group[0].years, group[0].und_price, [u.strike for u in group],
                    [u.iv for u in group],
                )
    return S.run_curve(units, cfg, skews), cfg


class TestUnitConstruction:
    def test_options_carry_ibs_volatility_and_forward(self, units):
        opt = next(u for u in units if u.asset_class == "option")
        assert opt.iv == pytest.approx(0.1832)
        assert opt.und_price == pytest.approx(6412.5)
        assert opt.underlying_is_forward is True

    def test_a_futures_option_is_grouped_by_settlement_not_last_trade(self, units):
        """The quarterly and the weekly in the fixture have different last
        trading dates and settle the same morning — they must land in one
        skew bucket, not two."""
        quarterly = next(u for u in units if u.label == "ESZ6 P5800")
        weekly = next(u for u in units if u.label == "EW4Z6 P5500")
        assert quarterly.skew_key == weekly.skew_key

    def test_a_future_gets_a_notional_not_a_market_value_to_shock(self, units):
        fut = next(u for u in units if u.asset_class == "future")
        assert fut.notional == pytest.approx(1.0 * 50 * 6412.5)

    def test_multiplier_comes_from_the_contract(self, units):
        assert all(u.multiplier == 50.0 for u in units if u.symbol == "ES")
        assert next(u for u in units if u.symbol == "AAPL").multiplier == 1.0


class TestCurve:
    def test_pnl_is_exactly_zero_at_zero_shock(self, units, shocks):
        """By construction: P&L is model-now against model-shocked, so the two
        cancel. If this drifts, the curve has picked up a model-versus-market
        residual that does not belong in it."""
        result, _ = curve(units, shocks)
        zero = next(r for r in result["curve"] if r["shock"] == 0.0)
        assert zero["pnl_total"] == pytest.approx(0.0, abs=1e-6)

    def test_a_short_put_book_troughs_on_the_downside(self, units, shocks):
        result, _ = curve(units, shocks)
        assert result["trough"]["shock"] < 0
        assert result["trough"]["pnl"] < 0

    def test_a_minimum_on_the_edge_of_the_range_is_labelled_as_one(self, units, shocks):
        """This book is net short downside and long a future, so it just keeps
        losing: the worst point of a −30%…+30% run sits on the boundary. Calling
        that "the trough" would understate the risk by everything beyond it, so
        the engine flags it and says to widen the range."""
        result, _ = curve(units, shocks)
        assert result["trough"]["shock"] == -0.30
        assert result["trough"]["atRangeEdge"] is True
        assert "widen" in result["trough"]["note"].lower()
        assert "troughRefined" not in result

    def test_an_interior_trough_is_refined_and_not_flagged(self, units, shocks):
        """Buy far out-of-the-money puts and the curve turns inside the window:
        below their strike they outrun the short book and the P&L climbs again.

        A linear hedge cannot produce this. Short options give a concave curve
        whose minima are always at the edges, so it takes long gamma below the
        book to put a trough in the middle — which is the whole reason anyone
        asks this server where the trough is.
        """
        protection = S.RiskUnit(
            key="hedge",
            label="ES protective put",
            symbol="ES",
            asset_class="option",
            position=30.0,
            multiplier=50.0,
            market_value=None,
            sec_type="FOP",
            strike=4800.0,
            right="P",
            years=126 / 365,
            iv=0.28,
            und_price=6412.5,
            underlying_is_forward=True,
            skew_key=("ES", "2026-12-18"),
        )
        result, _ = curve(list(units) + [protection], shocks)
        assert "atRangeEdge" not in result["trough"]
        assert -0.30 < result["trough"]["shock"] < 0.0
        refined = result["troughRefined"]
        assert refined["interpolated"] is True
        assert abs(refined["shock"] - result["trough"]["shock"]) <= 0.01

    def test_totals_equal_the_sum_of_the_breakdowns(self, units, shocks):
        result, _ = curve(units, shocks)
        for row in result["curve"]:
            assert sum(row["pnl_by_asset_class"].values()) == pytest.approx(
                row["pnl_total"], abs=0.05
            )
            assert sum(row["pnl_by_symbol"].values()) == pytest.approx(
                row["pnl_total"], abs=0.05
            )

    def test_equity_moves_linearly_with_the_shock(self, units, shocks):
        result, _ = curve(units, shocks)
        row = next(r for r in result["curve"] if r["shock"] == -0.10)
        assert row["pnl_by_symbol"]["AAPL"] == pytest.approx(122150.0 * -0.10, rel=1e-9)

    def test_beta_scales_only_the_equity_leg(self, units, shocks):
        plain, _ = curve(units, shocks)
        levered, _ = curve(units, shocks, betas={"AAPL": 2.0})
        at = -0.10
        p = next(r for r in plain["curve"] if r["shock"] == at)
        l = next(r for r in levered["curve"] if r["shock"] == at)
        assert l["pnl_by_symbol"]["AAPL"] == pytest.approx(2 * p["pnl_by_symbol"]["AAPL"])
        assert l["pnl_by_asset_class"]["option"] == pytest.approx(
            p["pnl_by_asset_class"]["option"]
        )

    def test_futures_pnl_is_the_change_in_notional(self, units, shocks):
        result, _ = curve(units, shocks)
        row = next(r for r in result["curve"] if r["shock"] == 0.05)
        assert row["pnl_by_asset_class"]["future"] == pytest.approx(
            50 * 6412.5 * 0.05, rel=1e-9
        )

    def test_bonds_are_flat_unless_a_rate_shift_is_asked_for(self, units, shocks):
        flat, _ = curve(units, shocks)
        assert all(r["pnl_by_asset_class"].get("bond", 0.0) == 0.0 for r in flat["curve"])
        shifted, _ = curve(units, shocks, bond_rate_shift_bp=100.0, bond_duration_years=6.0)
        row = next(r for r in shifted["curve"] if r["shock"] == 0.0)
        assert row["pnl_by_asset_class"]["bond"] == pytest.approx(
            -6.0 * 0.01 * 48925.0, rel=1e-9
        )


class TestVolModes:
    def test_sticky_strike_keeps_each_strikes_volatility(self, units):
        cfg = S.StressConfig(shocks=[-0.1], vol_mode="sticky_strike")
        opt = next(u for u in units if u.asset_class == "option")
        assert S.shocked_vol(opt, -0.10, cfg, {}) == pytest.approx(opt.iv)

    def test_a_vol_bump_is_in_points(self, units):
        cfg = S.StressConfig(shocks=[0.0], vol_bump=0.05)
        opt = next(u for u in units if u.asset_class == "option")
        assert S.shocked_vol(opt, 0.0, cfg, {}) == pytest.approx(opt.iv + 0.05)

    def test_a_vol_bump_costs_a_net_short_vega_book_money(self, units, shocks):
        """At the money the fixture is net short vega — short 10 and 5 against
        long 10 — so a five-point bump is a loss with the underlying unchanged.

        Note that this does *not* hold at every shock: thirty percent lower the
        long 5500 puts are the least deep in the money and carry the most vega,
        so the same bump helps there. Asserting the trough would be asserting
        the wrong thing.
        """
        bumped, _ = curve(units, shocks, vol_bump=0.05)
        at_zero = next(r for r in bumped["curve"] if r["shock"] == 0.0)
        assert at_zero["pnl_total"] < 0
        assert at_zero["pnl_by_asset_class"]["option"] == pytest.approx(
            at_zero["pnl_total"], abs=0.05
        )

    def test_sticky_moneyness_uses_the_skew_where_there_is_one(self, units, shocks):
        """The fixture holds three ES strikes on one settlement date, which is
        exactly enough to define a smile, so the two modes must disagree."""
        strike_mode, _ = curve(units, shocks, vol_mode="sticky_strike")
        moneyness_mode, _ = curve(units, shocks, vol_mode="sticky_moneyness")
        assert strike_mode["trough"]["pnl"] != moneyness_mode["trough"]["pnl"]

    def test_time_decay_helps_a_net_short_option_book_at_the_money(self, units, shocks):
        plain, _ = curve(units, shocks)
        rolled, _ = curve(units, shocks, date_offset_days=30)
        at_zero_plain = next(r for r in plain["curve"] if r["shock"] == 0.0)["pnl_total"]
        at_zero_rolled = next(r for r in rolled["curve"] if r["shock"] == 0.0)["pnl_total"]
        assert at_zero_plain == pytest.approx(0.0, abs=1e-6)
        assert at_zero_rolled > 0


class TestValidation:
    def test_an_unknown_vol_mode_is_rejected(self):
        with pytest.raises(ValueError, match="vol_mode"):
            S.StressConfig(shocks=[0.0], vol_mode="sticky_delta").validate()

    def test_shocks_expressed_as_percents_are_caught(self):
        """A caller passing 10 for '10%' would otherwise get a curve for the
        underlying going up elevenfold, which is a plausible-looking answer to
        a question nobody asked."""
        with pytest.raises(ValueError, match="fractions"):
            S.StressConfig(shocks=[-10.0, 0.0, 10.0]).validate()

    def test_an_empty_shock_list_is_rejected(self):
        with pytest.raises(ValueError):
            S.StressConfig(shocks=[]).validate()


class TestReconciliation:
    def test_the_fixture_reconciles(self, holdings, fake_ib):
        result = S.reconcile(holdings, fake_ib, "DU1234567")
        assert result["reconciled"] is True
        assert abs(result["residualPct"]) < 0.01

    def test_futures_contribute_their_pnl_not_their_notional(self, holdings, fake_ib):
        """A future's notional is 320,625 against a mark-to-market of 5,625.
        Adding the wrong one moves the derived total by a quarter of the
        account, so this is the assertion that catches it."""
        result = S.reconcile(holdings, fake_ib, "DU1234567")
        assert result["futuresUnrealizedPnl"] == pytest.approx(5625.0)
        assert result["derivedNetLiquidation"] == pytest.approx(result["netLiquidation"])

    def test_a_bond_is_taken_at_ibs_value_not_quantity_times_price(self, holdings, fake_ib):
        """50,000 nominal at 97.85 is 48,925, not 4,892,500. Taking IB's
        marketValue sidesteps the percentage quote entirely, and the
        reconciliation is what proves it."""
        bond = next(h for h in holdings if h.asset_class == "bond")
        naive = bond.position * bond.market_price
        assert naive == pytest.approx(100 * bond.market_value)
        assert S.reconcile(holdings, fake_ib, "DU1234567")["reconciled"] is True

    def test_a_gap_over_one_percent_fails_with_the_residual_attached(
        self, holdings, fake_ib
    ):
        fake_ib.values = [
            FakeAccountValue(tag="NetLiquidation", value="1000000"),
            FakeAccountValue(tag="TotalCashValue", value="1119400"),
        ]
        result = S.reconcile(holdings, fake_ib, "DU1234567")
        assert result["reconciled"] is False
        assert result["residual"] != 0
        assert "1%" in result["reason"]

    def test_missing_account_values_fail_rather_than_reconcile_by_default(
        self, holdings, fake_ib
    ):
        fake_ib.values = []
        result = S.reconcile(holdings, fake_ib, "DU1234567")
        assert result["reconciled"] is False
        assert "NetLiquidation" in result["reason"]


class TestUnpriceableUnits:
    def test_an_option_without_greeks_is_held_flat_and_flagged(self, holdings, shocks):
        holdings[0].greeks = None
        holdings[0].greeks_error = "no market data subscription"
        units = [S.unit_from_holding(h, ASOF) for h in holdings]
        broken = next(u for u in units if u.key == "700000001")
        assert broken.priceable is False
        assert broken.note == "no market data subscription"
        result, cfg = curve(units, shocks)
        # It contributes nothing rather than crashing the run.
        assert S.unit_pnl(broken, -0.20, cfg, {}) == 0.0
        assert result["trough"]["pnl"] < 0

    def test_the_position_report_shows_model_against_market(self, units):
        cfg = S.StressConfig(shocks=[0.0], asof=ASOF, rate=RATE)
        rows = S._position_report(units, cfg)
        opt = next(r for r in rows if r["key"] == "700000001")
        assert "modelPrice" in opt and "modelVsMarket" in opt
        # The fixture's quotes are the model price rounded to the quarter-point
        # tick, so anything beyond that is the arithmetic having drifted.
        assert abs(opt["modelVsMarket"]) < 0.25


class TestParabolicTrough:
    def test_it_finds_the_vertex_of_a_parabola(self):
        points = [(x / 100, (x / 100 - 0.07) ** 2) for x in range(-30, 31)]
        refined = S._parabolic_trough(points)
        assert refined["shock"] == pytest.approx(0.07, abs=1e-6)

    def test_a_minimum_on_the_edge_is_not_refined(self):
        """Extrapolating past the last grid point would be inventing a trough
        outside the range the caller asked about."""
        points = [(x / 100, -x) for x in range(0, 31)]
        assert S._parabolic_trough(points) is None


class TestAssumptions:
    def test_they_are_reported_with_every_run(self):
        a = S.assumptions(S.StressConfig(shocks=[0.0], vol_mode="sticky_moneyness"))
        assert a["volMode"] == "sticky_moneyness"
        assert "ACT/365" in a["dayCount"]
        assert "same percentage" in a["underlyingCorrelation"]
