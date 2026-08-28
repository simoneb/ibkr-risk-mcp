"""The stress engine, driven off the recorded portfolio.

Every assertion here is about a property of the curve rather than a
remembered number, so the tests survive a change of fixture but not a change
of behaviour.
"""

import asyncio
from datetime import date

import pytest

from ibkr_risk_mcp import stress as S
from ibkr_risk_mcp import pricing

from .conftest import FakeAccountValue, FakeContract

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
    return S.run_curve(units, cfg, surfaces_for(units, cfg)), cfg


def surfaces_for(units, cfg):
    """build_surfaces is async only because it may fetch; the pure path is
    rebuilt here so the engine can be exercised synchronously."""
    if cfg.vol_mode != "sticky_moneyness":
        return {}
    grouped = {}
    for u in units:
        if u.asset_class == "option" and u.priceable:
            grouped.setdefault(u.skew_key, []).append(u)
    surfaces = {}
    for (symbol, _expiry), group in grouped.items():
        if len({u.strike for u in group}) < 3:
            continue
        surfaces.setdefault(symbol, pricing.VolSurface()).add(
            pricing.VolSkew.from_strikes(
                group[0].years,
                group[0].forward(0.0, group[0].years, cfg.rate),
                [u.strike for u in group],
                [u.iv for u in group],
            )
        )
    return surfaces


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

    def test_the_fixture_troughs_on_the_downside(self, units, shocks):
        result, _ = curve(units, shocks)
        assert result["trough"]["shock"] < 0
        assert result["trough"]["pnl"] < 0

    def test_a_minimum_on_the_edge_of_the_range_is_labelled_as_one(self, units, shocks):
        """This fixture keeps losing all the way down, so the worst point of a
        −30%…+30% run sits on the boundary. Calling
        that "the trough" would understate the risk by everything beyond it, so
        the engine flags it and says to widen the range."""
        result, _ = curve(units, shocks)
        assert result["trough"]["shock"] == -0.30
        assert result["trough"]["atRangeEdge"] is True
        assert "widen" in result["trough"]["note"].lower()
        assert "troughRefined" not in result

    def test_an_interior_trough_is_refined_and_not_flagged(self, units, shocks):
        """Add long far out-of-the-money puts and the curve turns inside the
        window: below their strike they outrun the rest and the P&L climbs again.

        A linear hedge cannot produce this. Written options make a curve
        concave, and a concave curve has its minima at the edges, so it takes
        long gamma below to put a trough in the middle — which is the whole
        reason anyone asks this server where the trough is.
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

    def test_a_beta_on_one_symbol_leaves_the_others_alone(self, units, shocks):
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


@pytest.fixture
def fx_unit():
    """A short CAD futures option: the position that made the case for the
    scope gate. On the live account it contributed -21,716 at a 20% fall and
    -7,183 at a 10% rise, against -29,027 and +2,408 for an entire ES campaign.
    """
    from ibkr_risk_mcp import contracts as C
    from ibkr_risk_mcp.marketdata import Holding

    contract = FakeContract(
        conId=900000001,
        symbol="CAD",
        localSymbol="CAUZ6 P6900",
        secType="FOP",
        right="P",
        strike=0.69,
        lastTradeDateOrContractMonth="20261204",
        tradingClass="CAU",
        multiplier="100000",
        exchange="CME",
        currency="USD",
    )
    holding = Holding(
        contract=contract,
        position=-2.0,
        market_price=0.00044,
        market_value=-88.0,
        average_cost=0.0,
        unrealized_pnl=0.0,
        asset_class=C.asset_class("FOP"),
        multiplier=C.contract_multiplier(contract),
        greeks={
            "impliedVol": 0.057,
            "delta": -0.05,
            "gamma": 1.0,
            "vega": 0.01,
            "theta": -0.001,
            "optPrice": 0.00044,
            "pvDividend": 0.0,
            "undPrice": 0.7235,
        },
    )
    return S.unit_from_holding(holding, ASOF)


class TestScope:
    """Only equity underlyings are on the axis by default. This is not a
    refinement of the model, it is what makes the number mean anything."""

    def test_the_unit_knows_its_own_risk_group(self, fx_unit, units):
        assert fx_unit.risk_group == "fx"
        assert all(u.risk_group in ("equity", "rates") for u in units)

    def test_an_fx_position_is_off_the_axis_by_default(self, fx_unit, shocks):
        result, _ = curve([fx_unit], shocks)
        assert all(r["pnl_total"] == 0.0 for r in result["curve"])

    def test_scope_all_puts_it_back(self, fx_unit, shocks):
        result, _ = curve([fx_unit], shocks, scope="all")
        assert any(r["pnl_total"] != 0.0 for r in result["curve"])
        at_crash = next(r for r in result["curve"] if r["shock"] == -0.30)
        assert at_crash["pnl_total"] < 0  # short the put, and it goes deep in

    def test_exclusion_is_total_where_a_zero_beta_would_leak(self, fx_unit, shocks):
        """The difference that matters. A beta of 0 scales the underlying's move
        and nothing else, so vega and theta still get through; an excluded
        position must contribute exactly nothing whatever else is switched on.
        """
        excluded, _ = curve([fx_unit], shocks, vol_bump=0.05, date_offset_days=30)
        assert all(r["pnl_total"] == 0.0 for r in excluded["curve"])

        leaked, _ = curve(
            [fx_unit], shocks, scope="all", betas={"CAD": 0.0},
            vol_bump=0.05, date_offset_days=30,
        )
        at_zero = next(r for r in leaked["curve"] if r["shock"] == 0.0)
        assert at_zero["pnl_total"] != 0.0

    def test_what_was_excluded_is_named_and_valued(self, fx_unit, units):
        cfg = S.StressConfig(shocks=[0.0])
        report = S.excluded_report(units + [fx_unit], cfg)
        by_group = {r["riskGroup"]: r for r in report}
        assert by_group["fx"]["symbols"] == ["CAD"]
        assert by_group["fx"]["marketValue"] == pytest.approx(-88.0)
        assert "rates" in by_group  # the fixture's bond

    def test_the_exclusion_is_warned_about(self, fx_unit, units):
        cfg = S.StressConfig(shocks=[0.0])
        text = " ".join(S.scope_warnings(units + [fx_unit], cfg))
        assert "CAD" in text and "scope='all'" in text

    def test_a_risk_group_override_moves_a_symbol_off_the_axis(self, units, shocks):
        """IB publishes no asset class for a bond or gold ETF quoted as a stock,
        so AAPL-as-equity is a guess like any other and has to be correctable."""
        plain, _ = curve(units, shocks)
        overridden, _ = curve(units, shocks, risk_groups={"AAPL": "rates"})
        at = -0.10
        assert next(r for r in plain["curve"] if r["shock"] == at)["pnl_by_symbol"]["AAPL"] != 0
        assert (
            next(r for r in overridden["curve"] if r["shock"] == at)["pnl_by_symbol"]["AAPL"]
            == 0.0
        )

    def test_a_bond_rate_shift_still_works_under_the_equity_scope(self, units, shocks):
        """A rate shift is a different axis, asked for by name. The equity scope
        takes bonds off the *shock* axis; silently ignoring the rate shift too
        would be a surprise, not a simplification."""
        shifted, _ = curve(units, shocks, bond_rate_shift_bp=100.0, bond_duration_years=6.0)
        row = next(r for r in shifted["curve"] if r["shock"] == 0.0)
        assert row["pnl_by_asset_class"]["bond"] == pytest.approx(
            -6.0 * 0.01 * 48925.0, rel=1e-9
        )


class TestBeta:
    """The beta scales the *shock*, not the P&L, and it reaches every class
    that responds to one."""

    def test_it_scales_the_shock_on_an_option_rather_than_its_pnl(self, units, shocks):
        """The precise statement: an option at beta 0.5 under a 20% shock must
        price identically to the same option at beta 1.0 under a 10% shock.

        Scaling the P&L instead would price it at the full move and then halve
        the answer, and for anything convex those are different numbers. This
        is the test that tells the two apart.
        """
        halved, _ = curve(units, shocks, betas={"ES": 0.5})
        full, _ = curve(units, shocks)
        at_double = next(r for r in halved["curve"] if r["shock"] == -0.20)
        at_single = next(r for r in full["curve"] if r["shock"] == -0.10)
        assert at_double["pnl_by_symbol"]["ES"] == pytest.approx(
            at_single["pnl_by_symbol"]["ES"]
        )

    def test_a_zero_beta_takes_a_symbol_off_the_curve(self, units, shocks):
        """Which is the point of widening the scope: an underlying that does not
        belong on this axis can be stood down, instead of being shocked by a
        percentage that means nothing for it."""
        standing_down, _ = curve(units, shocks, betas={"ES": 0.0})
        assert all(
            r["pnl_by_symbol"]["ES"] == pytest.approx(0.0, abs=1e-6)
            for r in standing_down["curve"]
        )

    def test_a_zero_beta_is_not_the_same_as_excluding_the_symbol(self, units, shocks):
        """A beta only scales the underlying's move. It does not touch vega or
        theta, so a stood-down position still contributes P&L the moment
        volatility or the valuation date moves — which is exactly when a reader
        would assume it had been removed. Subtracting `pnl_by_symbol` is the
        only clean exclusion.
        """
        bumped, _ = curve(units, shocks, betas={"ES": 0.0}, vol_bump=0.05)
        at_zero = next(r for r in bumped["curve"] if r["shock"] == 0.0)
        assert abs(at_zero["pnl_by_symbol"]["ES"]) > 1.0

    def test_it_reaches_the_futures_leg_too(self, units, shocks):
        halved, _ = curve(units, shocks, betas={"ES": 0.5})
        row = next(r for r in halved["curve"] if r["shock"] == 0.05)
        # The curve rounds to the cent, and half the notional lands on a
        # half-cent, so this is an absolute tolerance rather than a relative one.
        assert row["pnl_by_asset_class"]["future"] == pytest.approx(
            50 * 6412.5 * 0.05 * 0.5, abs=0.01
        )

    def test_the_most_specific_key_wins(self, units):
        """Root, local symbol and underlying are all valid keys, for a book
        holding two contracts on one root that have to be told apart."""
        cfg = S.StressConfig(
            shocks=[0.0], betas={"ES": 0.5, "ESZ6 P5800": 0.25, "ESZ6": 0.75}
        )
        quarterly = next(u for u in units if u.label == "ESZ6 P5800")
        weekly = next(u for u in units if u.label == "EW4Z6 P5500")
        weekly.und_symbol = "ESZ6"

        assert S._beta(quarterly, cfg) == 0.25  # label beats root
        assert S._beta(weekly, cfg) == 0.5  # root beats underlying
        weekly.symbol = "6E"
        assert S._beta(weekly, cfg) == 0.75  # underlying, when the root misses
        assert S._beta(next(u for u in units if u.symbol == "AAPL"), cfg) == 1.0


class TestVolResponse:
    """The volatility *level* moving with the shock — the term neither vol mode
    carries and `vol_bump` cannot express, being flat along the axis."""

    def test_the_slope_is_volatility_points_per_one_percent_move(self, units):
        cfg = S.StressConfig(shocks=[0.0], vol_slope_down=1.0)
        opt = next(u for u in units if u.asset_class == "option")
        assert S.shocked_vol(opt, -0.20, cfg, {}) == pytest.approx(opt.iv + 0.20)

    def test_it_is_asymmetric_by_construction(self, units):
        """One slope per direction, because an index gives up far more
        volatility on the way down than it recovers on the way up. A downside
        slope must do nothing at all to a rally."""
        opt = next(u for u in units if u.asset_class == "option")
        down_only = S.StressConfig(shocks=[0.0], vol_slope_down=1.0)
        up_only = S.StressConfig(shocks=[0.0], vol_slope_up=0.5)
        assert S.shocked_vol(opt, 0.10, down_only, {}) == pytest.approx(opt.iv)
        assert S.shocked_vol(opt, 0.10, up_only, {}) == pytest.approx(opt.iv - 0.05)
        assert S.shocked_vol(opt, -0.10, up_only, {}) == pytest.approx(opt.iv)

    def test_it_stacks_on_the_vol_bump_rather_than_replacing_it(self, units):
        cfg = S.StressConfig(shocks=[0.0], vol_bump=0.05, vol_slope_down=1.0)
        opt = next(u for u in units if u.asset_class == "option")
        assert S.shocked_vol(opt, -0.10, cfg, {}) == pytest.approx(opt.iv + 0.05 + 0.10)

    def test_the_curve_still_starts_at_exactly_zero(self, units, shocks):
        """The response is proportional to the shock, so it vanishes at zero and
        cannot put a step at the origin — unlike vol_bump, which is a real
        immediate cost and does."""
        sloped, _ = curve(units, shocks, vol_slope_down=1.5, vol_slope_up=0.5)
        at_zero = next(r for r in sloped["curve"] if r["shock"] == 0.0)
        assert at_zero["pnl_total"] == pytest.approx(0.0, abs=1e-6)

    def test_a_downside_slope_touches_only_the_downside(self, units, shocks):
        plain, _ = curve(units, shocks)
        sloped, _ = curve(units, shocks, vol_slope_down=1.0)
        for a, b in zip(plain["curve"], sloped["curve"]):
            if a["shock"] > 0:
                assert a["pnl_total"] == pytest.approx(b["pnl_total"])
            elif a["shock"] < 0:
                assert abs(a["pnl_total"] - b["pnl_total"]) > 1.0

    def test_a_short_option_pays_for_the_volatility_it_is_short(self, units):
        """Taken on one position rather than the portfolio, because the sign is
        only unambiguous where the vega is: the fixture's net vega changes sign
        along the curve, so asserting the total would assert the wrong thing.
        """
        short_put = next(u for u in units if u.label == "ESZ6 P5800")
        assert short_put.position < 0
        plain = S.StressConfig(shocks=[-0.10], rate=RATE, asof=ASOF)
        sloped = S.StressConfig(shocks=[-0.10], rate=RATE, asof=ASOF, vol_slope_down=1.0)
        assert S.unit_pnl(short_put, -0.10, sloped, {}) < S.unit_pnl(
            short_put, -0.10, plain, {}
        )

    def test_zero_slopes_reprice_exactly_as_before(self, units, shocks):
        """The default has to be inert: every curve produced before this
        parameter existed must still come out to the cent."""
        default, _ = curve(units, shocks)
        explicit, _ = curve(units, shocks, vol_slope_down=0.0, vol_slope_up=0.0)
        assert default["curve"] == explicit["curve"]


class TestScopeWarnings:
    """Both knobs make a result less self-evident, so neither may apply
    silently."""

    def test_an_attenuated_position_is_named(self, units):
        cfg = S.StressConfig(shocks=[0.0], betas={"ES": 0.2})
        text = " ".join(S.scope_warnings(units, cfg))
        assert "ES" in text and "gap risk" in text

    def test_a_flat_volatility_level_is_flagged_on_an_option_book(self, units):
        cfg = S.StressConfig(shocks=[0.0])
        assert any("vol_slope_down" in w for w in S.scope_warnings(units, cfg))

    def test_a_slope_that_is_set_states_what_bounds_it(self, units):
        cfg = S.StressConfig(shocks=[0.0], vol_slope_down=1.0)
        text = " ".join(S.scope_warnings(units, cfg))
        assert "parallel" in text and "tenor" in text


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
        """At the money the fixture has negative vega on balance, so a five-point
        bump is a loss with the underlying unchanged.

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

    def test_vol_slopes_in_the_wrong_units_are_caught(self):
        """1.0 is one volatility point per 1%. Someone reading it as percent or
        basis points would pass 100 and get hundreds of points on a moderate
        shock, which prices as a plausible-looking catastrophe."""
        with pytest.raises(ValueError, match="volatility points"):
            S.StressConfig(shocks=[0.0], vol_slope_down=100.0).validate()

    def test_an_unknown_scope_is_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            S.StressConfig(shocks=[0.0], scope="equities").validate()

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


def option_unit(strike, iv, years, forward, expiry, position=-1.0, symbol="ES"):
    """A bare option unit, for the surface cases the recorded portfolio cannot
    reach — it holds every strike it has on a single settlement date, so it
    cannot exercise the tenor axis at all."""
    return S.RiskUnit(
        key=f"{symbol}-{expiry}-{strike:.0f}",
        label=f"{symbol} {expiry} P{strike:.0f}",
        symbol=symbol,
        asset_class="option",
        position=position,
        multiplier=50.0,
        market_value=None,
        sec_type="FOP",
        risk_group="equity",
        und_symbol=symbol,
        strike=strike,
        right="P",
        years=years,
        iv=iv,
        und_price=forward,
        underlying_is_forward=True,
        skew_key=(symbol, expiry),
    )


@pytest.fixture
def two_tenor_units():
    """A full ladder on the front expiry and one lonely strike on the back —
    the shape a real book usually has, and the one a per-expiry smile cannot
    handle."""
    front = [
        option_unit(5800.0, 0.24, 0.10, 6400.0, "2026-09-18"),
        option_unit(6100.0, 0.20, 0.10, 6400.0, "2026-09-18"),
        option_unit(6400.0, 0.17, 0.10, 6400.0, "2026-09-18"),
    ]
    back = [option_unit(6000.0, 0.21, 0.60, 6450.0, "2027-03-19")]
    return front + back


def surfaces_of(units, **kw):
    cfg = S.StressConfig(shocks=[0.0], vol_mode="sticky_moneyness", rate=RATE, **kw)
    surfaces, warnings = asyncio.run(S.build_surfaces(units, cfg))
    return surfaces, warnings, cfg


class TestVolSurface:
    """The surface replaced a bag of per-expiry smiles. What it has to buy is
    the tenor axis; what it must not cost is the curve's zero."""

    def test_the_smile_shift_is_zero_at_zero_shock(self, units):
        surfaces, _, cfg = surfaces_of(units)
        for unit in units:
            if unit.asset_class == "option" and unit.priceable:
                assert S.smile_shift(unit, 0.0, cfg, surfaces) == pytest.approx(0.0, abs=1e-12)

    def test_the_level_at_zero_shock_is_still_ibs_own_volatility(self, two_tenor_units):
        """The back-month strike is not a node of any fitted smile, so an
        absolute lookup would hand it the front month's interpolated value
        instead of the volatility IB published for it."""
        surfaces, _, cfg = surfaces_of(two_tenor_units)
        back = two_tenor_units[-1]
        assert S.shocked_vol(back, 0.0, cfg, surfaces) == pytest.approx(back.iv)

    def test_sticky_moneyness_still_starts_the_curve_at_exactly_zero(self, units, shocks):
        rows, _ = curve(units, shocks, vol_mode="sticky_moneyness")
        at_zero = next(r for r in rows["curve"] if r["shock"] == 0.0)
        assert at_zero["pnl_total"] == pytest.approx(0.0, abs=1e-6)

    def test_a_thin_expiry_borrows_its_shape_from_another_tenor(self, two_tenor_units):
        """One strike defines no slope of its own, so on its own expiry it
        cannot move at all. Read off the surface it does: the front month's
        smile carries across in total variance.

        The sign is downwards, and that is the model rather than a surprise.
        A 6000 put sits below a 6450 forward today and above a 5160 one after
        the shock, so under sticky moneyness it stops being a wing and picks up
        the low volatility that belongs at the money. This is exactly what
        sticky_strike refuses to do, and why it is the conservative mode.
        """
        surfaces, _, cfg = surfaces_of(two_tenor_units)
        back = two_tenor_units[-1]
        assert S.smile_shift(back, -0.20, cfg, surfaces) < 0.0
        assert S.smile_shift(back, -0.20, cfg, {}) == 0.0

    def test_the_borrowed_shape_is_named_in_the_warnings(self, two_tenor_units):
        _, warnings, _ = surfaces_of(two_tenor_units)
        assert any("interpolated from the other ES tenors" in w for w in warnings)

    def test_an_underlying_with_no_smile_anywhere_falls_back_and_says_so(self):
        lonely = [option_unit(6000.0, 0.21, 0.60, 6450.0, "2027-03-19")]
        surfaces, warnings, cfg = surfaces_of(lonely)
        assert surfaces == {}
        assert any("repriced sticky_strike" in w for w in warnings)
        assert S.shocked_vol(lonely[0], -0.20, cfg, surfaces) == pytest.approx(lonely[0].iv)

    def test_an_empty_surface_prices_exactly_like_sticky_strike(self, units, shocks):
        strike_mode, _ = curve(units, shocks, vol_mode="sticky_strike")
        cfg = S.StressConfig(
            shocks=shocks, vol_mode="sticky_moneyness", rate=RATE, asof=ASOF
        )
        with_nothing = S.run_curve(units, cfg, {})
        assert [r["pnl_total"] for r in with_nothing["curve"]] == [
            r["pnl_total"] for r in strike_mode["curve"]
        ]

    def test_the_report_lists_the_quotes_that_were_read(self, units):
        surfaces, _, _ = surfaces_of(units)
        rows = S.surface_report(surfaces)
        assert rows and all(r["underlying"] == "ES" for r in rows)
        quoted = {round(p["strike"]) for r in rows for p in r["points"]}
        assert quoted == {5500, 5800, 6100}

    def test_the_report_is_empty_when_no_surface_was_built(self, units):
        assert S.surface_report({}) == []

    def test_sticky_strike_builds_no_surface_at_all(self, units):
        cfg = S.StressConfig(shocks=[0.0], vol_mode="sticky_strike", rate=RATE)
        surfaces, warnings = asyncio.run(S.build_surfaces(units, cfg))
        assert surfaces == {} and warnings == []


class TestVolScenarios:
    def test_the_default_set_opens_with_a_constant_volatility_curve(self):
        first = S.DEFAULT_VOL_SCENARIOS[0]
        assert (first.vol_slope_down, first.vol_slope_up, first.vol_coord) == (0.0, 0.0, False)

    def test_the_default_set_is_the_two_curves_risk_navigator_draws(self):
        """Nothing invented on top: the constant-volatility case and IB's own
        Vol.Coord. model. The additive slopes that used to ship as defaults were
        guesses, and on a ratio book they had the wrong shape as well as the
        wrong size."""
        assert [s.name for s in S.DEFAULT_VOL_SCENARIOS] == ["const", "vol_coord"]
        assert S.DEFAULT_VOL_SCENARIOS[1].vol_coord is True

    def test_the_default_shocks_reach_forty_percent_down(self):
        assert min(S.DEFAULT_SHOCKS) == pytest.approx(-0.40)
        assert max(S.DEFAULT_SHOCKS) == pytest.approx(0.10)
        assert S.DEFAULT_SHOCKS == tuple(sorted(S.DEFAULT_SHOCKS))

    def test_a_scenario_only_touches_the_volatility_terms(self):
        overrides = S.VolScenario("stress", vol_slope_down=1.4, vol_slope_up=0.7).overrides()
        assert set(overrides) == {"vol_slope_down", "vol_slope_up", "vol_bump", "vol_coord"}

    def test_two_scenarios_differ_only_by_their_slope(self, units, shocks):
        const, _ = curve(units, shocks, vol_slope_down=0.0)
        stressed, _ = curve(units, shocks, vol_slope_down=1.4)
        at_zero = [c["curve"][len(shocks) // 2] for c in (const, stressed)]
        assert at_zero[0]["shock"] == at_zero[1]["shock"]
        worst = [min(r["pnl_total"] for r in c["curve"]) for c in (const, stressed)]
        assert worst[1] < worst[0]



class TestVolCoord:
    """IB's volatility-coordinated model. The asymmetry is documented by IB; the
    term damping is this server's fit, and both are stated as such in the
    output."""

    @pytest.fixture
    def cfg(self):
        return S.StressConfig(shocks=[0.0], vol_coord=True, rate=RATE)

    def test_nothing_moves_at_zero_shock(self, cfg, units):
        assert S.vol_coord_factor(0.0, 0.3, cfg) == pytest.approx(1.0)
        for u in units:
            if u.asset_class == "option" and u.priceable:
                assert S.shocked_vol(u, 0.0, cfg, {}) == pytest.approx(u.iv)

    def test_the_curve_still_starts_at_exactly_zero(self, units, shocks):
        rows, _ = curve(units, shocks, vol_coord=True)
        at_zero = next(r for r in rows["curve"] if r["shock"] == 0.0)
        assert at_zero["pnl_total"] == pytest.approx(0.0, abs=1e-6)

    def test_a_fall_moves_volatility_ten_times_as_hard_as_a_rise(self, cfg):
        up = S.vol_coord_factor(+0.10, 0.0, cfg) - 1.0
        down = S.vol_coord_factor(-0.10, 0.0, cfg) - 1.0
        assert down == pytest.approx(-10.0 * up)

    def test_it_is_relative_so_the_surface_steepens_on_its_own(self, cfg):
        """The whole reason this model exists. One multiplier, applied to a wing
        already quoted at 41% and to a 31% at the money, puts half again as many
        *points* on the wing. No additive slope reproduces that at any value,
        which is what made the parallel shift the wrong shape rather than merely
        the wrong size."""
        factor = S.vol_coord_factor(-0.20, 0.30, cfg)
        atm_points = 0.31 * (factor - 1.0)
        wing_points = 0.41 * (factor - 1.0)
        assert wing_points > atm_points
        assert wing_points / atm_points == pytest.approx(0.41 / 0.31)

    def test_a_longer_tenor_takes_less_of_the_shock(self, cfg):
        front = S.vol_coord_factor(-0.20, 0.05, cfg)
        back = S.vol_coord_factor(-0.20, 0.60, cfg)
        assert 1.0 < back < front
        # Undamped, a 20% fall would triple every volatility on the board.
        assert S.vol_coord_factor(-0.20, 0.0, cfg) == pytest.approx(3.0)

    def test_the_damping_is_what_keeps_a_back_month_plausible(self, cfg):
        """A six-month contract at 31% must not come back at 93%."""
        assert 0.31 * S.vol_coord_factor(-0.20, 0.5, cfg) < 0.55

    def test_a_rally_cannot_drive_volatility_negative(self, cfg):
        assert S.vol_coord_factor(+3.0, 0.0, cfg) >= 0.0

    def test_it_replaces_the_additive_slopes_rather_than_stacking(self, units):
        opt = next(u for u in units if u.asset_class == "option" and u.priceable)
        both = S.StressConfig(shocks=[0.0], vol_coord=True, vol_slope_down=1.4, rate=RATE)
        coord_only = S.StressConfig(shocks=[0.0], vol_coord=True, rate=RATE)
        assert S.shocked_vol(opt, -0.20, both, {}) == pytest.approx(
            S.shocked_vol(opt, -0.20, coord_only, {})
        )

    def test_a_vol_bump_still_lands_on_top(self, units):
        opt = next(u for u in units if u.asset_class == "option" and u.priceable)
        plain = S.StressConfig(shocks=[0.0], vol_coord=True, rate=RATE)
        bumped = S.StressConfig(shocks=[0.0], vol_coord=True, vol_bump=0.05, rate=RATE)
        assert S.shocked_vol(opt, -0.20, bumped, {}) == pytest.approx(
            S.shocked_vol(opt, -0.20, plain, {}) + 0.05
        )

    def test_the_model_is_spelled_out_in_the_assumptions(self):
        on = S.assumptions(S.StressConfig(shocks=[0.0], vol_coord=True))
        assert on["volCoord"] is True
        assert "not published by IB" in on["volCoordModel"]
        off = S.assumptions(S.StressConfig(shocks=[0.0]))
        assert off["volCoord"] is False
        assert off["volCoordModel"] is None

class TestCurvePoints:
    def test_a_point_carries_the_shocked_underlying_and_the_fraction_of_nlv(self):
        rows = [
            {
                "shock": -0.20,
                "pnl_total": -43000.0,
                "pnl_by_asset_class": {},
                "pnl_by_symbol": {},
            }
        ]
        point = S.curve_points(rows, spot=7030.0, net_liquidation=500_000.0)[0]
        assert point["shock_pct"] == pytest.approx(-20.0)
        assert point["underlying"] == pytest.approx(7030.0 * 0.80)
        assert point["pnl_pct_of_nlv"] == pytest.approx(-0.086)
        assert point["portfolio_value"] == pytest.approx(457_000.0)

    def test_the_fraction_is_omitted_rather_than_assumed(self):
        rows = [
            {"shock": 0.0, "pnl_total": 0.0, "pnl_by_asset_class": {}, "pnl_by_symbol": {}}
        ]
        point = S.curve_points(rows, spot=None, net_liquidation=None)[0]
        assert "pnl_pct_of_nlv" not in point
        assert "portfolio_value" not in point
        assert "underlying" not in point


class TestReferenceUnderlying:
    def test_it_names_the_heaviest_in_scope_exposure(self, units):
        cfg = S.StressConfig(shocks=[0.0], rate=RATE)
        ref = S.reference_underlying(units, cfg)
        assert ref["symbol"] == "ES"
        assert ref["spot"] == pytest.approx(6412.5)

    def test_an_off_axis_underlying_cannot_become_the_reference(self, fx_unit):
        """A lone CAD option is the only position there is, and it is still not
        the label for an equity axis — there is nothing to label."""
        cfg = S.StressConfig(shocks=[0.0], rate=RATE)
        assert S.reference_underlying([fx_unit], cfg) is None
        assert S.reference_underlying([fx_unit], S.StressConfig(shocks=[0.0], scope="all"))[
            "symbol"
        ] == "CAD"

    def test_nothing_priceable_gives_no_reference(self):
        assert S.reference_underlying([], S.StressConfig(shocks=[0.0])) is None


class TestScopeWarningsSwitch:
    def test_the_volatility_sentence_can_be_left_to_the_caller(self, units):
        cfg = S.StressConfig(shocks=[0.0], vol_slope_down=1.0)
        with_vol = S.scope_warnings(units, cfg)
        without = S.scope_warnings(units, cfg, include_vol=False)
        assert any("Implied volatility responds" in w for w in with_vol)
        assert not any("Implied volatility responds" in w for w in without)
        assert len(without) == len(with_vol) - 1


@pytest.fixture
def offline(monkeypatch, holdings, fake_ib):
    """The whole tool without TWS: the recorded portfolio stands in for the
    account, and everything else runs for real."""

    async def fake_get():
        return fake_ib

    async def fake_load(with_greeks=True, symbol=None):
        return holdings

    monkeypatch.setattr(S.connection, "get", fake_get)
    monkeypatch.setattr(S.connection, "require_account", lambda: "DU1234567")
    monkeypatch.setattr(S.MD, "load_holdings", fake_load)
    return fake_ib


def run_stress_curve(scenarios=None, date_offsets=None, **kw):
    kw.setdefault("rate", RATE)
    kw.setdefault("asof", ASOF)
    kw.setdefault("shocks", list(S.DEFAULT_SHOCKS))
    kw.setdefault("vol_mode", "sticky_moneyness")
    cfg = S.StressConfig(**kw)
    if scenarios is None:
        scenarios = list(S.DEFAULT_VOL_SCENARIOS)
    return asyncio.run(S.stress_curve(cfg, scenarios, date_offsets))


class TestStressCurve:
    def test_one_curve_per_scenario_over_one_shock_axis(self, offline):
        out = run_stress_curve()
        assert [c["name"] for c in out["curves"]] == ["const", "vol_coord"]
        assert all(len(c["points"]) == len(S.DEFAULT_SHOCKS) for c in out["curves"])
        axes = {tuple(p["shock"] for p in c["points"]) for c in out["curves"]}
        assert len(axes) == 1

    def test_the_constant_volatility_curve_starts_at_exactly_zero(self, offline):
        """The one curve with an external check. If this does not sit on Risk
        Navigator's blue line the volatility lookup is wrong, and nothing else
        in the result is worth reading."""
        const = run_stress_curve()["curves"][0]
        at_zero = next(p for p in const["points"] if p["shock"] == 0.0)
        assert const["volSlopeDown"] == 0.0
        assert at_zero["pnl"] == pytest.approx(0.0, abs=1e-6)

    def test_a_steeper_slope_costs_more_where_the_book_is_short_vega(self, offline):
        out = run_stress_curve(
            [S.VolScenario("a"), S.VolScenario("b", vol_slope_down=0.7),
             S.VolScenario("c", vol_slope_down=1.4)]
        )
        at_ten_down = [
            next(p["pnl"] for p in c["points"] if p["shock"] == pytest.approx(-0.10))
            for c in out["curves"]
        ]
        assert at_ten_down == sorted(at_ten_down, reverse=True)

    def test_the_ordering_reverses_in_the_far_tail_and_that_is_the_point(self, offline):
        """Deep enough down, the long 5500 puts are the least far in the money
        and carry the most vega of anything in the book, so a rising volatility
        starts *helping*. `moderate` ends up above `const` at -30%.

        This is why the regimes are returned as separate curves rather than
        collapsed into a band: which one is worst depends on where you are on
        the axis, and a reader who assumes the steepest slope is the worst case
        everywhere will read the tail backwards.
        """
        out = run_stress_curve(
            [S.VolScenario("const"), S.VolScenario("moderate", vol_slope_down=0.7)]
        )
        by_name = {c["name"]: c for c in out["curves"]}

        def pnl(name, shock):
            return next(
                p["pnl"] for p in by_name[name]["points"] if p["shock"] == pytest.approx(shock)
            )

        assert pnl("moderate", -0.10) < pnl("const", -0.10)
        assert pnl("moderate", -0.30) > pnl("const", -0.30)

    def test_every_point_carries_its_fraction_of_the_account(self, offline):
        out = run_stress_curve()
        assert out["netLiquidation"] > 0
        point = out["curves"][0]["points"][0]
        assert point["pnl_pct_of_nlv"] == pytest.approx(
            point["pnl"] / out["netLiquidation"], rel=1e-6
        )
        assert point["underlying"] == pytest.approx(6412.5 * (1.0 + point["shock"]))

    def test_the_surface_it_read_is_reported(self, offline):
        """The acceptance criterion: an empty surface under sticky_moneyness
        means every option quietly fell back to sticky_strike."""
        out = run_stress_curve()
        assert out["volSurfaceUsed"]
        assert {p["strike"] for r in out["volSurfaceUsed"] for p in r["points"]} == {
            5500.0,
            5800.0,
            6100.0,
        }

    def test_an_empty_surface_is_called_out_rather_than_left_to_be_noticed(self, offline):
        out = run_stress_curve(vol_mode="sticky_moneyness", shocks=[-0.1, 0.0])
        assert out["volSurfaceUsed"]
        empty = run_stress_curve(vol_mode="sticky_strike")
        assert empty["volSurfaceUsed"] == []

    def test_the_off_axis_position_is_named_and_valued(self, offline):
        out = run_stress_curve()
        assert any(row["riskGroup"] == "rates" for row in out["excluded"])
        assert any("kept only equity underlyings" in w for w in out["warnings"])

    def test_the_slope_is_flagged_as_an_input_once_not_per_curve(self, offline):
        out = run_stress_curve([S.VolScenario("a"), S.VolScenario("b", vol_slope_down=0.7)])
        slope_notes = [w for w in out["warnings"] if "additive volatility slope" in w]
        assert len(slope_notes) == 1
        assert not any("Implied volatility responds" in w for w in out["warnings"])

    def test_the_parallel_shift_caveat_stays_off_a_run_that_has_no_slope(self, offline):
        """It described a parallel shift flat across tenors on a run whose only
        model was vol_coord — relative, and damped by tenor. A warning that
        contradicts the model in use is worse than none."""
        out = run_stress_curve()
        assert not any("parallel shift" in w for w in out["warnings"])
        assert not any("flat across tenors" in w for w in out["warnings"])
        assert any("which is worst depends on where you are on the axis" in w
                   for w in out["warnings"])

    def test_the_assumptions_do_not_claim_a_single_volatility_slope(self, offline):
        """`assumptions` describes one config and there are three here. Leaving
        the base config's zeros in place would read as "volatility is held
        constant" on the one result whose whole point is that it is not."""
        out = run_stress_curve()
        assumed = out["assumptions"]
        assert "volSlopeDown" not in assumed
        assert "varies by curve" in assumed["volatilityLevel"]
        assert [s["name"] for s in assumed["volScenarios"]] == ["const", "vol_coord"]
        assert assumed["volMode"] == "sticky_moneyness"

    def test_the_reference_underlying_is_named(self, offline):
        out = run_stress_curve()
        assert out["underlying"] == {"symbol": "ES", "spot": 6412.5}

    def test_it_reconciles_against_the_recorded_account(self, offline):
        assert run_stress_curve()["reconciled"] is True

    def test_duplicate_scenario_names_are_rejected(self, offline):
        with pytest.raises(ValueError, match="unique"):
            run_stress_curve([S.VolScenario("a"), S.VolScenario("a", vol_slope_down=1.0)])

    def test_no_scenarios_is_rejected(self, offline):
        with pytest.raises(ValueError, match="at least one"):
            run_stress_curve([])

    def test_a_scenario_slope_in_the_wrong_units_is_caught(self, offline):
        with pytest.raises(ValueError, match="volatility points"):
            run_stress_curve([S.VolScenario("percent", vol_slope_down=100.0)])


class TestVolCoordProvenance:
    """The decay is a fit, not a published number, and it was fitted on a book
    that held nothing past four months. Both facts have to reach the reader at
    runtime — `assumptions` is where you look once you already suspect
    something."""

    def test_the_factory_decay_announces_itself_as_a_fit(self, units):
        cfg = S.StressConfig(shocks=[0.0], vol_coord=True)
        assert cfg.vol_coord_decay == S.DEFAULT_VOL_COORD_DECAY
        out = S.vol_coord_warnings(units, cfg)
        assert any("factory decay" in w and "read off a chart by eye" in w for w in out)

    def test_a_recalibrated_decay_does_not(self, units):
        cfg = S.StressConfig(shocks=[0.0], vol_coord=True, vol_coord_decay=3.1)
        assert not any("factory decay" in w for w in S.vol_coord_warnings(units, cfg))

    def test_a_position_past_the_calibration_is_named(self, units):
        """The fixture's ES options sit inside the calibrated range, so nothing
        fires until the range is pulled in under them — which is the same thing
        that happens to a real book holding a LEAPS."""
        cfg = S.StressConfig(shocks=[0.0], vol_coord=True, vol_coord_calibrated_to_years=0.1)
        out = S.vol_coord_warnings(units, cfg)
        assert any("beyond 0.10 years" in w for w in out)
        assert any("understates the risk of anything long-dated" in w for w in out)

    def test_nothing_fires_when_every_tenor_is_covered(self, units):
        cfg = S.StressConfig(
            shocks=[0.0], vol_coord=True, vol_coord_decay=3.1,
            vol_coord_calibrated_to_years=10.0,
        )
        assert S.vol_coord_warnings(units, cfg) == []

    def test_the_extrapolation_it_warns_about_is_real(self):
        """An exponential does not merely lose accuracy past its fit, it decays
        to nothing: a one-year option would be repriced as though a 20% crash
        barely moved its volatility. That is the reason the warning exists and
        the reason a floor on VR would be the wrong fix — it would hide it."""
        cfg = S.StressConfig(shocks=[0.0], vol_coord=True)
        assert S.vol_coord_factor(-0.20, 0.25, cfg) > 1.5
        assert S.vol_coord_factor(-0.20, 1.0, cfg) < 1.02

    def test_the_curve_carries_the_warning(self, offline):
        out = run_stress_curve()
        assert any("factory decay" in w for w in out["warnings"])

    def test_a_curve_without_vol_coord_is_not_lectured_about_it(self, offline):
        out = run_stress_curve([S.VolScenario("const")])
        assert not any("factory decay" in w for w in out["warnings"])


class TestCalibrateVolCoord:
    """The point of the calibration entry point is that the decay stops being a
    constant somebody once fitted."""

    @pytest.fixture
    def cfg(self):
        return S.StressConfig(shocks=[0.0], scope="equity", asof=ASOF, rate=RATE)

    def targets_from(self, units, cfg, decay, shocks):
        """A curve this engine produced at a known decay, used as the thing to
        recover. If the fit cannot find a decay it generated itself, it will
        certainly not find one Risk Navigator generated."""
        known = S.replace(cfg, vol_coord=True, vol_coord_decay=decay, shocks=shocks)
        rows = S.run_curve(units, known, {})["curve"]
        return {r["shock"]: r["pnl_total"] for r in rows}

    def test_it_recovers_a_decay_it_was_given(self, units, cfg):
        shocks = [-0.25, -0.20, -0.15, -0.10, -0.05]
        out = S.calibrate_vol_coord(units, cfg, self.targets_from(units, cfg, 2.5, shocks))
        assert out["decay"] == pytest.approx(2.5, abs=0.05)
        assert out["rms"] < 1.0

    def test_every_point_comes_back_with_its_residual(self, units, cfg):
        shocks = [-0.20, -0.15, -0.10, -0.05]
        out = S.calibrate_vol_coord(units, cfg, self.targets_from(units, cfg, 3.0, shocks))
        assert [p["shock"] for p in out["points"]] == sorted(shocks)
        assert all("target" in p and "model" in p and "residual" in p for p in out["points"])

    def test_it_reports_the_tenors_the_targets_actually_constrain(self, units, cfg):
        shocks = [-0.20, -0.15, -0.10, -0.05]
        out = S.calibrate_vol_coord(units, cfg, self.targets_from(units, cfg, 3.0, shocks))
        assert out["calibratedToYears"] == pytest.approx(
            max(u.years for u in units if u.asset_class == "option" and u.priceable), abs=1e-3
        )
        assert any("extrapolation" in w for w in out["warnings"])

    def test_it_shows_the_most_extreme_volatility_the_fit_produces(self, units, cfg):
        shocks = [-0.25, -0.20, -0.15, -0.10]
        out = S.calibrate_vol_coord(units, cfg, self.targets_from(units, cfg, 3.0, shocks))
        extreme = out["mostExtremeVol"]
        assert extreme["atShock"] == pytest.approx(-0.25)
        assert extreme["impliedVolAfter"] > extreme["impliedVolBefore"]

    def test_too_few_points_is_refused_rather_than_fitted(self, units, cfg):
        with pytest.raises(ValueError, match="three points"):
            S.calibrate_vol_coord(units, cfg, {-0.20: -28000.0, -0.10: -22000.0})


def expiry_unit(label, expiry, strike, position, *, iv=0.22, years=None, right="P"):
    """A synthetic ES option on a stated expiry.

    The recorded portfolio settles all three of its options on one morning,
    which is the right fixture for the skew buckets and the wrong one for a
    breakdown whose entire purpose is telling expiries apart.
    """
    settles = date.fromisoformat(expiry)
    return S.RiskUnit(
        key=label,
        label=label,
        symbol="ES",
        asset_class="option",
        position=position,
        multiplier=50.0,
        market_value=None,
        sec_type="FOP",
        expiry=settles,
        strike=strike,
        right=right,
        years=years if years is not None else max((settles - ASOF).days, 1) / 365,
        iv=iv,
        und_price=6412.5,
        underlying_is_forward=True,
        skew_key=("ES", expiry),
    )


@pytest.fixture
def two_expiries(units):
    """The recorded book plus a nearer expiry, one short and one long."""
    return list(units) + [
        expiry_unit("ESU6 P5800", "2026-09-30", 5800.0, -6.0),
        expiry_unit("ESU6 P5500", "2026-09-30", 5500.0, 4.0),
    ]


class TestExpiryBreakdown:
    """`pnl_by_symbol` puts nine ES expiries under one key. On a campaign that
    runs one root across many expiries that is the wrong unit of account: the
    operative question is which expiry is holding the trough down, and the only
    way to answer it was to cross the position list against the curve by hand.
    """

    def test_the_default_response_is_the_one_callers_already_parse(self, units, shocks):
        """Retrocompatibility, asserted rather than assumed. Nothing about the
        shape of a call that passes no new parameter may move."""
        result, _ = curve(units, shocks)
        row = result["curve"][0]
        assert set(row) == {"shock", "pnl_total", "pnl_by_asset_class", "pnl_by_symbol"}
        assert "troughByExpiry" not in result

    def test_expiry_replaces_symbol_rather_than_joining_it(self, two_expiries, shocks):
        result, _ = curve(two_expiries, shocks, breakdown="expiry")
        row = result["curve"][0]
        assert "pnl_by_expiry" in row
        assert "pnl_by_symbol" not in row

    def test_both_returns_both_and_none_returns_neither(self, two_expiries, shocks):
        both, _ = curve(two_expiries, shocks, breakdown="both")
        assert {"pnl_by_symbol", "pnl_by_expiry"} <= set(both["curve"][0])
        none, _ = curve(two_expiries, shocks, breakdown="none")
        assert not {"pnl_by_symbol", "pnl_by_expiry"} & set(none["curve"][0])
        assert "troughByExpiry" not in none

    def test_every_point_sums_to_its_own_total(self, two_expiries, shocks):
        """The acceptance criterion, and the reason positions with no expiry
        get a key of their own: a breakdown that quietly dropped the future and
        the bond would not add up, and a reader could not tell that from a
        breakdown that was simply wrong."""
        result, _ = curve(two_expiries, shocks, breakdown="expiry")
        for row in result["curve"]:
            assert sum(row["pnl_by_expiry"].values()) == pytest.approx(
                row["pnl_total"], abs=0.05
            )

    def test_the_key_is_the_settlement_date_so_a_weekly_and_a_quarterly_meet(
        self, units, shocks
    ):
        """The fixture's ESZ6 and EW4Z6 last trade on different days and settle
        the same morning. Two rows there would be two names for one expiry."""
        result, _ = curve(units, shocks, breakdown="expiry")
        keys = set(result["curve"][0]["pnl_by_expiry"])
        assert "ES 2026-12-18" in keys
        assert "ES 2026-12-17" not in keys

    def test_positions_with_no_expiry_are_named_by_their_class(self, units, shocks):
        result, _ = curve(units, shocks, breakdown="expiry")
        keys = set(result["curve"][0]["pnl_by_expiry"])
        assert "ES (future)" in keys
        assert "AAPL (equity)" in keys
        assert not any(k.startswith("AAPL 20") for k in keys)

    def test_an_expiry_row_is_the_pnl_of_that_expiry_alone(self, two_expiries, shocks):
        """Checked against the same units run on their own, which is the only
        statement of what the key is supposed to mean."""
        result, _ = curve(two_expiries, shocks, breakdown="expiry")
        sept = [u for u in two_expiries if u.expiry == date(2026, 9, 30)]
        alone, _ = curve(sept, shocks, breakdown="expiry")
        at = -0.20
        combined_row = next(r for r in result["curve"] if r["shock"] == at)
        alone_row = next(r for r in alone["curve"] if r["shock"] == at)
        assert combined_row["pnl_by_expiry"]["ES 2026-09-30"] == pytest.approx(
            alone_row["pnl_total"], abs=0.05
        )


class TestTroughByExpiry:
    def test_it_reports_each_expiry_worst_first(self, two_expiries, shocks):
        result, _ = curve(two_expiries, shocks, breakdown="expiry")
        rows = result["troughByExpiry"]
        assert [r["pnl"] for r in rows] == sorted(r["pnl"] for r in rows)
        assert {r["key"] for r in rows} == set(result["curve"][0]["pnl_by_expiry"])

    def test_each_row_is_that_expiry_own_minimum(self, two_expiries, shocks):
        result, _ = curve(two_expiries, shocks, breakdown="expiry")
        for row in result["troughByExpiry"]:
            along = [r["pnl_by_expiry"][row["key"]] for r in result["curve"]]
            assert row["pnl"] == pytest.approx(min(along), abs=0.01)

    def test_it_also_says_what_each_expiry_does_at_the_portfolio_trough(
        self, two_expiries, shocks
    ):
        """The two columns answer different questions. What an expiry can cost
        is its own minimum; whether it is the one to close is what it
        contributes where the account's floor actually sits, and an expiry
        whose worst point is nowhere near the book's is not the one to buy
        back."""
        result, _ = curve(two_expiries, shocks, breakdown="expiry")
        at = result["trough"]["shock"]
        row = next(r for r in result["curve"] if r["shock"] == at)
        contributions = {
            e["key"]: e["pnlAtPortfolioTrough"] for e in result["troughByExpiry"]
        }
        assert contributions == pytest.approx(row["pnl_by_expiry"], abs=0.01)
        assert sum(contributions.values()) == pytest.approx(result["trough"]["pnl"], abs=0.05)


class TestValuationDate:
    """An absolute date instead of counting days out by hand — the arithmetic
    that goes wrong over a weekend."""

    def test_a_date_becomes_the_offset_it_is(self):
        assert S.offset_to(date(2026, 9, 30), ASOF) == 47
        assert S.resolve_offsets(valuation_date="2026-09-30", asof=ASOF) == [47]

    def test_a_weekend_is_not_adjusted_away(self):
        """No calendar magic. Answering for Monday because Saturday is not a
        trading day would misdescribe the axis the curve was drawn on."""
        saturday = date(2026, 8, 15)
        assert saturday.weekday() == 5
        assert S.resolve_offsets(valuation_date="2026-08-15", asof=ASOF) == [1]

    def test_a_date_in_the_past_is_refused(self):
        with pytest.raises(ValueError, match="cannot value a book in the past"):
            S.resolve_offsets(valuation_date="2026-08-01", asof=ASOF)

    def test_a_date_that_is_not_a_date_says_so(self):
        with pytest.raises(ValueError, match="ISO calendar dates"):
            S.parse_valuation_date("30 September")

    def test_the_two_ways_of_saying_when_cannot_both_be_used(self):
        with pytest.raises(ValueError, match="only one of them can"):
            S.resolve_offsets(date_offset_days=3, valuation_date="2026-09-30", asof=ASOF)

    def test_a_family_of_dates_resolves_in_the_order_given(self):
        assert S.resolve_offsets(
            valuation_dates=["2026-08-14", "2026-08-17"], asof=ASOF
        ) == [0, 3]
        assert S.resolve_offsets(date_offsets=[0, 3], asof=ASOF) == [0, 3]

    def test_the_same_date_twice_is_not_a_comparison(self):
        with pytest.raises(ValueError, match="asked for twice"):
            S.resolve_offsets(date_offsets=[3, 3], asof=ASOF)

    def test_an_absolute_date_prices_the_same_as_the_offset_it_stands_for(
        self, units, shocks
    ):
        by_offset, _ = curve(units, shocks, date_offset_days=47)
        by_date, _ = curve(
            units, shocks, date_offset_days=S.offset_to(date(2026, 9, 30), ASOF)
        )
        assert by_date["curve"] == by_offset["curve"]

    def test_a_negative_offset_is_refused_at_the_config(self):
        cfg = S.StressConfig(shocks=[0.0], date_offset_days=-5)
        with pytest.raises(ValueError, match="value the book in the past"):
            cfg.validate()


class TestMultipleValuationDates:
    """Two dates in one call, off one snapshot. Two calls could not promise
    that: the book and the market move between them, and part of the difference
    then is not the days at all."""

    def test_the_scenarios_are_crossed_with_the_dates(self, offline):
        out = run_stress_curve(date_offsets=[0, 3])
        assert [c["label"] for c in out["curves"]] == [
            "const @ 2026-08-14",
            "const @ 2026-08-17",
            "vol_coord @ 2026-08-14",
            "vol_coord @ 2026-08-17",
        ]
        assert [c["name"] for c in out["curves"]] == [
            "const",
            "const",
            "vol_coord",
            "vol_coord",
        ]

    def test_one_date_leaves_the_label_as_the_scenario_name(self, offline):
        out = run_stress_curve()
        assert [c["label"] for c in out["curves"]] == ["const", "vol_coord"]
        assert all(c["dateOffsetDays"] == 0 for c in out["curves"])
        assert all(c["valuationDate"] == "2026-08-14" for c in out["curves"])

    def test_the_dates_are_named_at_the_top(self, offline):
        out = run_stress_curve(date_offsets=[0, 3])
        assert out["valuationDates"] == [
            {"offsetDays": 0, "valuationDate": "2026-08-14"},
            {"offsetDays": 3, "valuationDate": "2026-08-17"},
        ]

    def test_time_is_the_only_thing_that_moved(self, offline):
        """The curve is model-now against model-then, so a date offset is pure
        decay: at zero shock the fixture's net short options make money by
        waiting, and the two curves must differ there."""
        out = run_stress_curve(date_offsets=[0, 3])
        by_label = {c["label"]: c for c in out["curves"]}

        def at_zero(label):
            return next(p["pnl"] for p in by_label[label]["points"] if p["shock"] == 0.0)

        assert at_zero("const @ 2026-08-14") == pytest.approx(0.0, abs=1e-6)
        assert at_zero("const @ 2026-08-17") != pytest.approx(0.0, abs=1e-6)

    def test_rolling_time_forward_is_flagged_as_decay_not_forecast(self, offline):
        out = run_stress_curve(date_offsets=[0, 3])
        assert any("not a forecast" in w for w in out["warnings"])
        assert not any("not a forecast" in w for w in run_stress_curve()["warnings"])

    def test_the_assumptions_stop_claiming_one_date(self, offline):
        """Same reasoning as the volatility terms: with a family of dates the
        base config's single offset describes none of the curves."""
        one = run_stress_curve()
        assert one["assumptions"]["dateOffsetDays"] == 0
        many = run_stress_curve(date_offsets=[0, 3])
        assert "dateOffsetDays" not in many["assumptions"]
        assert len(many["assumptions"]["valuationDates"]) == 2

    def test_the_curves_carry_their_expiry_breakdown_too(self, offline):
        out = run_stress_curve(date_offsets=[0, 3], breakdown="expiry")
        for entry in out["curves"]:
            assert entry["troughByExpiry"]
            assert "pnl_by_expiry" in entry["points"][0]
            assert "pnl_by_symbol" not in entry["points"][0]
