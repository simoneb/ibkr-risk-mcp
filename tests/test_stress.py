"""The stress engine, driven off the recorded portfolio.

Every assertion here is about a property of the curve rather than a
remembered number, so the tests survive a change of fixture but not a change
of behaviour.
"""

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
