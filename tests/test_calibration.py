"""The stored vol_coord calibration.

The decay is the one number in this server that is neither IB's nor measured
from the account, and until it could be stored, every session went back to a
fit somebody else made against somebody else's book. These tests are about the
two things that makes true: that a stored fit is picked up without being asked
for, and that a fit is never stored without knowing what it was fitted against.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date

import pytest

from ibkr_risk_mcp import calibration
from ibkr_risk_mcp import stress as S

from .test_stress import ASOF, RATE, offline  # noqa: F401  (fixture)


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "vol_coord.json"
    monkeypatch.setenv("IBKR_CALIBRATION_FILE", str(path))
    return path


@pytest.fixture
def units(holdings):
    return [S.unit_from_holding(h, ASOF) for h in holdings]


class TestTheStore:
    def test_nothing_stored_means_the_factory_number(self, store):
        assert calibration.load() is None
        assert calibration.decay(S.DEFAULT_VOL_COORD_DECAY) == S.DEFAULT_VOL_COORD_DECAY
        assert calibration.provenance() is None

    def test_a_stored_fit_comes_back(self, store):
        calibration.save({"decay": 3.21, "calibratedToYears": 0.6})
        assert calibration.decay(S.DEFAULT_VOL_COORD_DECAY) == 3.21
        assert calibration.calibrated_to_years(0.345) == 0.6

    def test_the_environment_decides_where_it_lives(self, store):
        calibration.save({"decay": 3.21})
        assert store.exists()
        assert json.loads(store.read_text())["decay"] == 3.21

    def test_a_corrupt_file_falls_back_rather_than_taking_a_run_down(self, store):
        """A stale file has nothing to do with the question being asked, and
        the fallback is the factory decay — which is what the caller would have
        had anyway."""
        store.write_text("{not json", encoding="utf-8")
        assert calibration.load() is None
        assert calibration.decay(4.736) == 4.736

    def test_a_file_with_no_usable_decay_is_ignored(self, store):
        store.write_text(json.dumps({"decay": 0}), encoding="utf-8")
        assert calibration.load() is None

    def test_a_fit_written_before_the_reach_was_recorded_keeps_the_old_limit(self, store):
        """Falling back to the shipped figure rather than to infinity. A
        calibration that says nothing about long tenors still knows nothing
        about them, and claiming otherwise would silence the warning that
        matters most."""
        calibration.save({"decay": 3.21})
        assert calibration.calibrated_to_years(0.345) == 0.345


class TestTheConfigReadsIt:
    def test_a_stored_decay_becomes_the_default(self, store):
        calibration.save({"decay": 2.5, "calibratedToYears": 0.9})
        cfg = S.StressConfig(shocks=[0.0])
        assert cfg.vol_coord_decay == 2.5
        assert cfg.vol_coord_calibrated_to_years == 0.9

    def test_an_explicit_decay_still_wins(self, store):
        calibration.save({"decay": 2.5})
        assert S.StressConfig(shocks=[0.0], vol_coord_decay=9.9).vol_coord_decay == 9.9


class TestWhatTheWarningSays:
    """Three provenances and they are not interchangeable: your fit, the
    factory fit, and a number the caller typed."""

    def _warnings(self, units, cfg):
        return " ".join(S.vol_coord_warnings(units, cfg))

    def test_the_factory_decay_still_announces_itself_as_somebody_else_fit(
        self, store, units
    ):
        cfg = S.StressConfig(shocks=[0.0], vol_coord=True, asof=ASOF)
        text = self._warnings(units, cfg)
        assert "factory decay" in text
        assert "calibrate_vol_coord tool" in text

    def test_a_calibrated_run_says_what_it_was_fitted_against(self, store, units):
        calibration.save(
            {
                "decay": 3.0,
                "calibratedToYears": 0.4,
                "fittedAt": "2026-08-28T10:00:00",
                "account": "DU1234567",
                "points": [{"shock": -0.2}, {"shock": -0.1}],
            }
        )
        cfg = S.StressConfig(shocks=[0.0], vol_coord=True, asof=ASOF)
        text = self._warnings(units, cfg)
        assert "factory decay" not in text
        assert "YOUR calibration" in text
        assert "2026-08-28T10:00:00" in text
        assert "DU1234567" in text

    def test_a_hand_typed_decay_borrows_nobody_provenance(self, store, units):
        calibration.save({"decay": 3.0, "fittedAt": "2026-08-28T10:00:00"})
        cfg = S.StressConfig(shocks=[0.0], vol_coord=True, vol_coord_decay=7.0, asof=ASOF)
        assert "did not fit it" in S.decay_source(cfg)
        assert "2026-08-28" not in S.decay_source(cfg)

    def test_the_source_reaches_the_assumptions(self, store, units):
        cfg = S.StressConfig(shocks=[0.0], vol_coord=True, asof=ASOF)
        assumed = S.assumptions(cfg)
        assert assumed["volCoordDecay"] == S.DEFAULT_VOL_COORD_DECAY
        assert "factory fit" in assumed["volCoordDecaySource"]

    def test_a_run_without_vol_coord_claims_no_decay_at_all(self, store):
        assumed = S.assumptions(S.StressConfig(shocks=[0.0]))
        assert assumed["volCoordDecay"] is None
        assert assumed["volCoordDecaySource"] is None


def run_calibration(offline_ib, targets, **kw):
    persist = kw.pop("persist", True)
    cfg = S.StressConfig(shocks=sorted(targets), asof=ASOF, rate=RATE, **kw)
    return asyncio.run(S.run_calibration(cfg, targets, persist=persist))


@pytest.fixture
def targets(holdings):
    """A curve this engine produced at a known decay. If the fit cannot recover
    a decay it generated itself it will certainly not recover Risk Navigator's.
    """
    units = [S.unit_from_holding(h, ASOF) for h in holdings]
    shocks = [-0.25, -0.20, -0.15, -0.10, -0.05]
    cfg = S.StressConfig(
        shocks=shocks, asof=ASOF, rate=RATE, vol_coord=True, vol_coord_decay=2.5
    )
    rows = S.run_curve(units, cfg, {})["curve"]
    return {r["shock"]: r["pnl_total"] for r in rows}


class TestRunCalibration:
    def test_it_fits_and_keeps_the_answer(self, store, offline, targets):
        out = run_calibration(offline, targets)
        assert out["decay"] == pytest.approx(2.5, abs=0.05)
        assert out["stored"] is True
        assert out["storedAt"] == str(store)
        assert calibration.decay(4.736) == pytest.approx(2.5, abs=0.05)

    def test_the_next_run_picks_it_up_without_being_told(self, store, offline, targets):
        run_calibration(offline, targets)
        assert S.StressConfig(shocks=[0.0]).vol_coord_decay == pytest.approx(2.5, abs=0.05)

    def test_what_it_was_fitted_against_is_stored_beside_it(self, store, offline, targets):
        run_calibration(offline, targets)
        record = json.loads(store.read_text())
        assert record["account"] == "DU1234567"
        assert record["scope"] == "equity"
        assert len(record["points"]) == len(targets)
        assert date.fromisoformat(record["fittedAt"][:10])

    def test_persist_false_fits_without_adopting(self, store, offline, targets):
        out = run_calibration(offline, targets, persist=False)
        assert out["decay"] == pytest.approx(2.5, abs=0.05)
        assert out["stored"] is False
        assert not store.exists()
        assert any("persist=false" in w for w in out["warnings"])

    def test_a_book_that_does_not_reconcile_is_fitted_but_never_stored(
        self, store, offline, targets, monkeypatch
    ):
        """The asymmetry that decides this: a curve missing a position says so
        through `reconciled`, while a decay that absorbed the same gap would go
        on deforming every later run with nothing to give it away."""
        monkeypatch.setattr(
            S, "reconcile", lambda *a, **k: {"reconciled": False, "residual": 1e6}
        )
        out = run_calibration(offline, targets)
        assert out["decay"] > 0
        assert out["reconciled"] is False
        assert out["stored"] is False
        assert not store.exists()
        assert any("NOT STORED" in w for w in out["warnings"])
