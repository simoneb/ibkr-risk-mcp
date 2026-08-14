"""The repricing layer, checked against identities rather than against
remembered numbers.

Put-call parity, monotonicity in volatility, and an implied-volatility
round-trip pin the pricer down without anyone having to trust a hardcoded
price to four decimals.
"""

import math

import pytest

from ibkr_risk_mcp import pricing

from .conftest import load_json

R = 0.04


class TestBlack76:
    def test_put_call_parity(self):
        F, K, T, vol = 6412.5, 6000.0, 0.35, 0.18
        call = pricing.black76_price(F, K, T, vol, R, "C")
        put = pricing.black76_price(F, K, T, vol, R, "P")
        assert call - put == pytest.approx(math.exp(-R * T) * (F - K), rel=1e-10)

    def test_price_rises_with_volatility(self):
        prices = [
            pricing.black76_price(6412.5, 5800.0, 0.35, v, R, "P")
            for v in (0.10, 0.15, 0.20, 0.30)
        ]
        assert prices == sorted(prices)

    def test_put_price_rises_as_the_underlying_falls(self):
        prices = [
            pricing.black76_price(f, 5800.0, 0.35, 0.18, R, "P")
            for f in (7000, 6500, 6000, 5500, 5000)
        ]
        assert prices == sorted(prices)

    def test_at_expiry_the_price_is_intrinsic(self):
        assert pricing.black76_price(5500.0, 5800.0, 0.0, 0.18, R, "P") == pytest.approx(300.0)
        assert pricing.black76_price(6000.0, 5800.0, 0.0, 0.18, R, "P") == pytest.approx(0.0)

    def test_zero_volatility_is_discounted_intrinsic(self):
        T = 0.5
        price = pricing.black76_price(5500.0, 5800.0, T, 0.0, R, "P")
        assert price == pytest.approx(math.exp(-R * T) * 300.0)

    def test_an_unrecognised_right_is_an_error(self):
        with pytest.raises(ValueError):
            pricing.black76_price(100.0, 100.0, 1.0, 0.2, R, "X")


class TestGreeks:
    def test_delta_matches_a_finite_difference(self):
        F, K, T, vol = 6412.5, 6000.0, 0.35, 0.18
        h = 0.01
        up = pricing.black76_price(F + h, K, T, vol, R, "P")
        down = pricing.black76_price(F - h, K, T, vol, R, "P")
        assert pricing.black76_greeks(F, K, T, vol, R, "P")["delta"] == pytest.approx(
            (up - down) / (2 * h), rel=1e-5
        )

    def test_vega_is_per_volatility_point(self):
        """IB quotes vega per point (0.01 of volatility), and the local greeks
        have to be in the same units or the two cannot be compared."""
        F, K, T, vol = 6412.5, 6000.0, 0.35, 0.18
        h = 0.0001
        analytic = pricing.black76_greeks(F, K, T, vol, R, "P")["vega"]
        numeric = (
            pricing.black76_price(F, K, T, vol + h, R, "P")
            - pricing.black76_price(F, K, T, vol - h, R, "P")
        ) / (2 * h)
        assert analytic == pytest.approx(numeric * 0.01, rel=1e-4)

    def test_theta_is_per_calendar_day(self):
        F, K, T, vol = 6412.5, 6000.0, 0.35, 0.18
        analytic = pricing.black76_greeks(F, K, T, vol, R, "P")["theta"]
        numeric = pricing.black76_price(F, K, T - 1 / 365, vol, R, "P") - pricing.black76_price(
            F, K, T, vol, R, "P"
        )
        assert analytic == pytest.approx(numeric, rel=5e-3)

    def test_put_delta_is_negative_and_call_delta_positive(self):
        args = (6412.5, 6000.0, 0.35, 0.18, R)
        assert pricing.black76_greeks(*args, "P")["delta"] < 0
        assert pricing.black76_greeks(*args, "C")["delta"] > 0


class TestEquityOptions:
    def test_black_scholes_equals_black76_on_the_implied_forward(self):
        S, K, T, vol, div = 244.3, 240.0, 0.5, 0.28, 1.2
        via_spot = pricing.black_scholes_price(S, K, T, vol, R, "P", div)
        forward = pricing.forward_from_spot(S, T, R, div)
        assert via_spot == pytest.approx(pricing.black76_price(forward, K, T, vol, R, "P"))

    def test_dividends_lower_the_forward(self):
        assert pricing.forward_from_spot(100.0, 1.0, R, 2.0) < pricing.forward_from_spot(
            100.0, 1.0, R, 0.0
        )

    def test_spot_delta_includes_the_carry_factor(self):
        S, K, T, vol = 244.3, 240.0, 0.5, 0.28
        h = 0.001
        up = pricing.black_scholes_price(S + h, K, T, vol, R, "P")
        down = pricing.black_scholes_price(S - h, K, T, vol, R, "P")
        assert pricing.black_scholes_greeks(S, K, T, vol, R, "P")["delta"] == pytest.approx(
            (up - down) / (2 * h), rel=1e-4
        )


class TestImpliedVol:
    def test_round_trip(self):
        F, K, T, vol = 6412.5, 5800.0, 0.35, 0.1832
        price = pricing.black76_price(F, K, T, vol, R, "P")
        assert pricing.implied_vol_black76(price, F, K, T, R, "P") == pytest.approx(vol, rel=1e-6)

    def test_a_price_below_intrinsic_has_no_implied_volatility(self):
        """Returning the nearest bound would invent a number where there is
        none; None is the answer that cannot be misread."""
        F, K, T = 5000.0, 5800.0, 0.35
        below_intrinsic = math.exp(-R * T) * (K - F) * 0.5
        assert pricing.implied_vol_black76(below_intrinsic, F, K, T, R, "P") is None

    def test_a_price_above_the_upper_bound_has_none_either(self):
        assert pricing.implied_vol_black76(1e9, 6412.5, 5800.0, 0.35, R, "P") is None


class TestVolSkew:
    @pytest.fixture
    def skew(self):
        # A downward-sloping put skew, as index options actually trade.
        return pricing.VolSkew.from_strikes(
            years=0.35,
            forward=6412.5,
            strikes=[5400, 5800, 6100, 6400, 6700],
            vols=[0.245, 0.203, 0.181, 0.162, 0.149],
        )

    def test_interpolates_between_quoted_strikes(self, skew):
        mid = skew.at_strike(5950.0)
        assert 0.181 < mid < 0.203

    def test_extrapolation_is_flat_in_both_wings(self, skew):
        """A fitted curve will happily produce a negative volatility two
        hundred points out. A visibly-too-low flat wing is the better
        failure."""
        assert skew.at_strike(3000.0) == pytest.approx(0.245)
        assert skew.at_strike(9000.0) == pytest.approx(0.149)

    def test_reading_at_a_shocked_forward_slides_the_smile(self, skew):
        """Sticky moneyness in one assertion: with the forward 10% lower, the
        5800 strike is much closer to the money than it was, so it picks up a
        lower volatility than the one it holds today."""
        today = skew.at_strike(5800.0)
        shocked = skew.at_strike(5800.0, forward=6412.5 * 0.90)
        assert shocked < today

    def test_a_single_strike_still_produces_a_flat_curve(self):
        skew = pricing.VolSkew.from_strikes(0.35, 6412.5, [5800], [0.18])
        assert skew.at_strike(4000.0) == pytest.approx(0.18)
        assert skew.points == 1

    def test_duplicate_strikes_are_averaged_not_rejected(self):
        skew = pricing.VolSkew.from_strikes(
            0.35, 6412.5, [5800, 5800, 6100], [0.20, 0.22, 0.18]
        )
        assert skew.at_strike(5800.0) == pytest.approx(0.21)

    def test_a_forward_of_zero_is_an_error(self):
        with pytest.raises(ValueError):
            pricing.VolSkew.from_strikes(0.35, 0.0, [5800], [0.18])


class TestVolSurface:
    @pytest.fixture
    def surface(self):
        return pricing.build_surface(load_json("vol_surface.json")["rows"])

    def test_built_from_recorded_rows(self, surface):
        assert len(surface.tenors) == 3

    def test_term_structure_is_not_flat(self, surface):
        """The fixture is the ES shape from the brief: 12% at the front, near
        15% at 139 days. Reading the front month for the back one understates
        a long-dated position badly, and this is the assertion that says the
        surface does not do that."""
        atm_near = surface.iv(surface.tenors[0], 6400.0, 6412.5)
        atm_far = surface.iv(surface.tenors[-1], 6400.0, 6412.5)
        assert atm_near < atm_far
        assert atm_near == pytest.approx(0.121, abs=0.01)
        assert atm_far == pytest.approx(0.150, abs=0.01)

    def test_interpolation_across_tenors_lands_between_them(self, surface):
        t0, t1 = surface.tenors[0], surface.tenors[1]
        mid = surface.iv((t0 + t1) / 2, 6400.0, 6412.5)
        assert (
            min(surface.iv(t0, 6400.0, 6412.5), surface.iv(t1, 6400.0, 6412.5))
            <= mid
            <= max(surface.iv(t0, 6400.0, 6412.5), surface.iv(t1, 6400.0, 6412.5))
        )

    def test_total_variance_grows_with_tenor(self):
        """Interpolating volatility directly can make total variance fall as
        time passes, which is a calendar arbitrage. Interpolating variance
        cannot."""
        surface = pricing.build_surface(load_json("vol_surface.json")["rows"])
        variances = [
            surface.iv(t, 6400.0, 6412.5) ** 2 * t
            for t in [0.05, 0.1, 0.2, 0.3, 0.38]
        ]
        assert variances == sorted(variances)

    def test_beyond_the_quoted_tenors_the_nearest_skew_is_held(self, surface):
        assert surface.iv(5.0, 6400.0, 6412.5) == pytest.approx(
            surface.iv(surface.tenors[-1], 6400.0, 6412.5)
        )

    def test_rows_without_a_volatility_are_skipped_not_defaulted(self):
        surface = pricing.build_surface(
            [
                {"yearsToExpiry": 0.1, "strike": 6000, "impliedVol": None, "undPrice": 6412.5},
                {"yearsToExpiry": 0.1, "strike": 6100, "impliedVol": 0.19, "undPrice": 6412.5},
            ]
        )
        assert surface.skews[list(surface.skews)[0]].points == 1

    def test_an_empty_surface_returns_nothing_rather_than_a_guess(self):
        assert pricing.VolSurface().iv(0.3, 6000.0, 6412.5) is None
