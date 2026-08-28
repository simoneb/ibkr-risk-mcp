"""The tool layer.

Thin, and mostly docstrings — but it is where the ways of saying *when* a curve
is valued are reconciled into one offset, and where the calibration targets are
checked before anything is fitted to them. A mistake there is invisible from
the engine's own tests, because the engine never sees the parameter that was
dropped.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from ibkr_risk_mcp import calibration
from ibkr_risk_mcp import server
from ibkr_risk_mcp import stress as S

from .test_stress import offline  # noqa: F401  (fixture)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("IBKR_CALIBRATION_FILE", str(tmp_path / "vol_coord.json"))
    return tmp_path / "vol_coord.json"


def call(name, **kw):
    """Through FastMCP rather than at the Python function.

    Calling the decorated coroutine directly would skip pydantic entirely — the
    `Field(...)` defaults arrive as FieldInfo objects and a dict never becomes
    the model it declares — so it would exercise none of the parameter handling
    these tests exist to cover. `call_tool` is the path a client takes.
    """
    _content, structured = asyncio.run(server.mcp.call_tool(name, kw))
    return structured


SHOCKS = [-0.20, -0.10, 0.0, 0.10]


class TestStressCurveWiring:
    def test_a_family_of_dates_reaches_the_engine(self, offline):
        out = call("stress_curve", shocks=SHOCKS, date_offsets=[0, 2])
        assert out["success"] is True
        assert len(out["curves"]) == 4
        assert {c["dateOffsetDays"] for c in out["curves"]} == {0, 2}

    def test_absolute_dates_land_on_the_offsets_they_stand_for(self, offline):
        today = date.today()
        out = call(
            "stress_curve",
            shocks=SHOCKS,
            valuation_dates=[today.isoformat(), (today + timedelta(days=5)).isoformat()],
        )
        assert {c["dateOffsetDays"] for c in out["curves"]} == {0, 5}

    def test_two_ways_of_saying_when_is_an_error_not_a_guess(self, offline):
        out = call(
            "stress_curve", shocks=SHOCKS, date_offsets=[0, 2], valuation_dates=["2030-01-01"]
        )
        assert out["success"] is False
        assert "only one of them can" in out["error"]

    def test_the_expiry_breakdown_reaches_the_points(self, offline):
        out = call("stress_curve", shocks=SHOCKS, breakdown="expiry")
        point = out["curves"][0]["points"][0]
        assert "pnl_by_expiry" in point
        assert "pnl_by_symbol" not in point

    def test_an_unset_decay_takes_the_stored_calibration(self, store, offline):
        calibration.save({"decay": 2.75, "calibratedToYears": 0.9})
        out = call("stress_curve", shocks=SHOCKS)
        assert out["assumptions"]["volScenarios"]  # the run happened
        assert any("YOUR calibration" in w for w in out["warnings"])

    def test_an_explicit_decay_overrides_it(self, store, offline):
        calibration.save({"decay": 2.75})
        out = call("stress_curve", shocks=SHOCKS, vol_coord_decay=6.0)
        assert not any("YOUR calibration" in w for w in out["warnings"])


class TestStressPortfolioWiring:
    def test_an_absolute_valuation_date_becomes_the_offset(self, offline):
        target = date.today() + timedelta(days=7)
        out = call("stress_portfolio", shocks=SHOCKS, valuation_date=target.isoformat())
        assert out["assumptions"]["dateOffsetDays"] == 7
        assert out["assumptions"]["valuationDate"] == target.isoformat()

    def test_a_date_and_an_offset_together_are_refused(self, offline):
        out = call(
            "stress_portfolio",
            shocks=SHOCKS,
            date_offset_days=3,
            valuation_date="2030-01-01",
        )
        assert out["success"] is False
        assert "only one of them can" in out["error"]

    def test_the_default_call_still_returns_the_old_shape(self, offline):
        out = call("stress_portfolio", shocks=SHOCKS)
        row = out["curve"][0]
        assert "pnl_by_symbol" in row
        assert "pnl_by_expiry" not in row
        assert "troughByExpiry" not in out


class TestStressWhatifWiring:
    def test_the_difference_curve_is_broken_out_by_expiry_too(self, offline):
        """The legs cannot be resolved without TWS, so this checks the part that
        does not need them: with no legs the difference is zero at every expiry,
        and the keys are still there to be read."""
        out = call("stress_whatif", legs=[], shocks=SHOCKS, breakdown="expiry")
        assert out["success"] is True
        point = out["difference"]["curve"][0]
        assert set(point["pnl_by_expiry"]) == set(
            out["base"]["curve"][0]["pnl_by_expiry"]
        )
        assert all(v == 0.0 for v in point["pnl_by_expiry"].values())
        assert out["difference"]["troughByExpiry"]


class TestCalibrateVolCoordWiring:
    def test_upside_only_targets_are_refused_before_anything_is_fitted(
        self, store, offline
    ):
        out = call(
            "calibrate_vol_coord",
            targets=[{"shock": 0.05, "pnl": -100.0}, {"shock": 0.10, "pnl": -200.0},
                     {"shock": 0.15, "pnl": -300.0}],
        )
        assert out["success"] is False
        assert "none of these shocks is negative" in out["error"]

    def test_too_few_points_are_refused(self, store, offline):
        out = call(
            "calibrate_vol_coord",
            targets=[{"shock": -0.10, "pnl": -100.0}, {"shock": -0.20, "pnl": -200.0}],
        )
        assert out["success"] is False
        assert "three distinct shocks" in out["error"]

    def test_a_fit_is_stored_and_picked_up_by_the_next_curve(self, store, offline):
        units = [S.unit_from_holding(h) for h in asyncio.run(S.MD.load_holdings())]
        cfg = S.StressConfig(
            shocks=[-0.25, -0.20, -0.15, -0.10], vol_coord=True, vol_coord_decay=2.5
        )
        rows = S.run_curve(units, cfg, {})["curve"]
        out = call(
            "calibrate_vol_coord",
            targets=[{"shock": r["shock"], "pnl": r["pnl_total"]} for r in rows],
        )
        assert out["success"] is True
        assert out["decay"] == pytest.approx(2.5, abs=0.05)
        assert out["stored"] is True
        assert store.exists()
        assert S.StressConfig(shocks=[0.0]).vol_coord_decay == pytest.approx(2.5, abs=0.05)
