"""Local repricing: Black-76, Black-Scholes, and volatility interpolation.

This module is deliberately free of any IB dependency. Everything the stress
engine needs to reprice a position is passed in as numbers, so the pricing
layer is testable against recorded fixtures without TWS running — which is the
only way to tell a pricing bug apart from a market-data problem.

The convention throughout: **the forward is the state variable.** Futures
options are quoted on a forward already; equity options are converted to one
by subtracting the present value of dividends that IB reports in
``modelGreeks.pvDividend`` and carrying the rest at the risk-free rate. One
shocked quantity, one pricer, and the equity-option path is a thin wrapper
rather than a second model that can disagree with the first.

**These are European prices.** Equity options and CME futures options are both
American, and Black-76 has no early exercise in it. Out of the money the
difference is negligible, which is where a short-premium book lives; in the
money it is real. Measured against live IB data on a 75-strike put with spot at
71.94: 4.51 here against IB's 4.65, a 2.9% shortfall that is the early-exercise
premium. The stress engine reports that gap per position as ``modelVsMarket``
rather than absorbing it, so it stays visible instead of quietly biasing a
curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import norm

#: Below this, a volatility is treated as zero and the option is worth its
#: discounted intrinsic value. Above it, Black-76 is well behaved.
MIN_VOL = 1e-6
#: Same idea for time: at less than about fifteen minutes the formula is all
#: intrinsic anyway and the d1/d2 division starts to lose precision.
MIN_YEARS = 1e-6


def _norm_right(right: str) -> str:
    r = (right or "").strip().upper()
    if r in ("C", "CALL"):
        return "C"
    if r in ("P", "PUT"):
        return "P"
    raise ValueError(f"right must be a call or a put, got {right!r}")


def black76_price(F: float, K: float, T: float, sigma: float, r: float, right: str) -> float:
    """Undiscounted-forward Black-76 price of one option (per unit, not per
    contract — the multiplier is applied by the caller).

    ``F`` forward, ``K`` strike, ``T`` years to settlement, ``sigma``
    volatility, ``r`` continuously compounded risk-free rate.
    """
    right = _norm_right(right)
    df = math.exp(-r * max(T, 0.0))
    if T <= MIN_YEARS or sigma <= MIN_VOL:
        intrinsic = max(F - K, 0.0) if right == "C" else max(K - F, 0.0)
        return df * intrinsic
    if F <= 0 or K <= 0:
        intrinsic = max(F - K, 0.0) if right == "C" else max(K - F, 0.0)
        return df * intrinsic
    vt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vt * vt) / vt
    d2 = d1 - vt
    if right == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_greeks(
    F: float, K: float, T: float, sigma: float, r: float, right: str
) -> dict[str, float]:
    """Greeks consistent with :func:`black76_price`, in the units IB uses:
    delta per unit of forward, vega per volatility *point* (0.01), theta per
    calendar day.

    These are the server's own numbers. Where IB's model greeks are available
    they are reported instead — this exists for hypothetical legs that have no
    position and therefore no IB greeks, and for the sanity check that the
    local model and IB's agree before any curve built on it is believed.
    """
    right = _norm_right(right)
    df = math.exp(-r * max(T, 0.0))
    if T <= MIN_YEARS or sigma <= MIN_VOL or F <= 0 or K <= 0:
        itm = (F > K) if right == "C" else (F < K)
        return {
            "delta": (1.0 if right == "C" else -1.0) * df * (1.0 if itm else 0.0),
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }
    vt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vt * vt) / vt
    d2 = d1 - vt
    pdf = norm.pdf(d1)
    delta = df * (norm.cdf(d1) if right == "C" else norm.cdf(d1) - 1.0)
    gamma = df * pdf / (F * vt)
    vega = df * F * pdf * math.sqrt(T)
    price = black76_price(F, K, T, sigma, r, right)
    theta_annual = -df * F * pdf * sigma / (2 * math.sqrt(T)) + r * price
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega * 0.01,
        "theta": theta_annual / 365.0,
    }


def forward_from_spot(S: float, T: float, r: float, pv_dividend: float = 0.0) -> float:
    """The forward implied by a spot price, the rate, and the present value of
    the dividends paid before expiry — the quantity IB reports as
    ``pvDividend``. This is what puts equity options and futures options on the
    same footing."""
    return (S - (pv_dividend or 0.0)) * math.exp(r * max(T, 0.0))


def black_scholes_price(
    S: float, K: float, T: float, sigma: float, r: float, right: str, pv_dividend: float = 0.0
) -> float:
    """Black-Scholes on a spot underlying with discrete dividends handled the
    way IB reports them: as a present value subtracted from the spot. Equal by
    construction to Black-76 on the resulting forward."""
    return black76_price(forward_from_spot(S, T, r, pv_dividend), K, T, sigma, r, right)


def black_scholes_greeks(
    S: float, K: float, T: float, sigma: float, r: float, right: str, pv_dividend: float = 0.0
) -> dict[str, float]:
    """Greeks with respect to **spot**. Delta and gamma are rescaled by
    dF/dS = e^{rT}, which the forward-space greeks do not include."""
    F = forward_from_spot(S, T, r, pv_dividend)
    g = black76_greeks(F, K, T, sigma, r, right)
    carry = math.exp(r * max(T, 0.0))
    return {
        "delta": g["delta"] * carry,
        "gamma": g["gamma"] * carry * carry,
        "vega": g["vega"],
        "theta": g["theta"],
    }


def implied_vol_black76(
    price: float, F: float, K: float, T: float, r: float, right: str
) -> float | None:
    """Invert Black-76. Returns None rather than a number when the price is
    outside the no-arbitrage bounds — a quote below intrinsic has no implied
    volatility, and returning the nearest bound would invent one."""
    right = _norm_right(right)
    df = math.exp(-r * max(T, 0.0))
    intrinsic = df * (max(F - K, 0.0) if right == "C" else max(K - F, 0.0))
    upper = df * (F if right == "C" else K)
    if not (intrinsic - 1e-12 <= price <= upper + 1e-12) or T <= MIN_YEARS:
        return None
    if price <= intrinsic + 1e-12:
        return 0.0

    def f(sigma: float) -> float:
        return black76_price(F, K, T, sigma, r, right) - price

    try:
        return float(brentq(f, 1e-6, 10.0, xtol=1e-8, maxiter=200))
    except ValueError:
        return None


@dataclass
class VolSkew:
    """One expiry's volatility smile, held as a function of log-moneyness.

    Log-moneyness rather than strike is the whole point: under
    ``sticky_moneyness`` the smile travels with the forward, so a shocked
    strike is looked up at ``ln(K/F')`` on the *unshocked* curve. Holding the
    curve in strike space would make that a re-fit instead of a lookup.

    Extrapolation is flat in both wings. A quadratic fit through three quoted
    strikes will happily produce a negative volatility two hundred points out,
    and a flat wing that is visibly too low is a better failure than a
    plausible-looking negative variance.
    """

    years: float
    forward: float
    log_moneyness: np.ndarray
    vols: np.ndarray
    _curve: PchipInterpolator | None = field(default=None, repr=False)

    @classmethod
    def from_strikes(
        cls, years: float, forward: float, strikes: Sequence[float], vols: Sequence[float]
    ) -> "VolSkew":
        pairs = sorted(
            (float(k), float(v))
            for k, v in zip(strikes, vols)
            if k and k > 0 and v is not None and v > 0 and math.isfinite(v)
        )
        # Two quotes at the same strike (a call and a put, say) would break a
        # monotone interpolator; average them instead of dropping either.
        merged: dict[float, list[float]] = {}
        for k, v in pairs:
            merged.setdefault(k, []).append(v)
        ks = np.array(sorted(merged), dtype=float)
        vs = np.array([float(np.mean(merged[k])) for k in ks], dtype=float)
        if forward <= 0 or len(ks) == 0:
            raise ValueError("a skew needs a positive forward and at least one quoted strike")
        skew = cls(years=years, forward=float(forward), log_moneyness=np.log(ks / forward), vols=vs)
        if len(ks) >= 3:
            skew._curve = PchipInterpolator(skew.log_moneyness, vs, extrapolate=False)
        return skew

    @property
    def points(self) -> int:
        return int(len(self.vols))

    def at_log_moneyness(self, k: float) -> float:
        lo, hi = float(self.log_moneyness[0]), float(self.log_moneyness[-1])
        if k <= lo:
            return float(self.vols[0])
        if k >= hi:
            return float(self.vols[-1])
        if self._curve is not None:
            return float(self._curve(k))
        return float(np.interp(k, self.log_moneyness, self.vols))

    def at_strike(self, strike: float, forward: float | None = None) -> float:
        """Volatility for a strike, read at its moneyness against ``forward``
        (the original forward when omitted)."""
        fwd = forward if forward and forward > 0 else self.forward
        return self.at_log_moneyness(math.log(strike / fwd))


@dataclass
class VolSurface:
    """A set of skews keyed by years to expiry, with interpolation across
    tenors done in **total variance**, not in volatility.

    Interpolating volatility directly is the mistake that puts ES at a flat 12%
    for every tenor because that is what the front month prints. Total variance
    sigma^2 * T is what is additive in time, and linear interpolation of it is
    the standard no-calendar-arbitrage choice. Outside the quoted tenors the
    nearest skew's volatility is held flat rather than extrapolated.
    """

    skews: dict[float, VolSkew] = field(default_factory=dict)

    def add(self, skew: VolSkew) -> None:
        self.skews[round(skew.years, 6)] = skew

    @property
    def tenors(self) -> list[float]:
        return sorted(self.skews)

    def skew_at(self, years: float) -> VolSkew | None:
        """The quoted skew nearest in time, used when a caller wants the raw
        smile rather than an interpolated volatility."""
        if not self.skews:
            return None
        return self.skews[min(self.tenors, key=lambda t: abs(t - years))]

    def iv(self, years: float, strike: float, forward: float) -> float | None:
        """Volatility at an arbitrary (tenor, strike), the strike read at its
        moneyness against the supplied forward."""
        tenors = self.tenors
        if not tenors:
            return None
        k = math.log(strike / forward) if forward > 0 and strike > 0 else 0.0
        if len(tenors) == 1 or years <= tenors[0]:
            return self.skews[tenors[0]].at_log_moneyness(k)
        if years >= tenors[-1]:
            return self.skews[tenors[-1]].at_log_moneyness(k)
        hi_i = next(i for i, t in enumerate(tenors) if t >= years)
        t0, t1 = tenors[hi_i - 1], tenors[hi_i]
        v0 = self.skews[t0].at_log_moneyness(k)
        v1 = self.skews[t1].at_log_moneyness(k)
        w0, w1 = v0 * v0 * t0, v1 * v1 * t1
        w = w0 + (w1 - w0) * (years - t0) / (t1 - t0)
        return math.sqrt(max(w, 0.0) / years) if years > 0 else v0


def build_surface(quotes: Iterable[dict]) -> VolSurface:
    """Assemble a :class:`VolSurface` from the rows ``get_vol_surface`` returns.

    Each row needs ``yearsToExpiry``, ``strike``, ``impliedVol`` and
    ``undPrice``; rows without a usable volatility are skipped rather than
    defaulted, so a half-empty surface is visibly half-empty.
    """
    by_tenor: dict[float, dict[str, list]] = {}
    for q in quotes:
        iv = q.get("impliedVol")
        strike = q.get("strike")
        years = q.get("yearsToExpiry")
        fwd = q.get("undPrice")
        if not iv or not strike or not years or not fwd:
            continue
        bucket = by_tenor.setdefault(round(float(years), 6), {"k": [], "v": [], "f": []})
        bucket["k"].append(float(strike))
        bucket["v"].append(float(iv))
        bucket["f"].append(float(fwd))
    surface = VolSurface()
    for years, bucket in by_tenor.items():
        forward = float(np.mean(bucket["f"]))
        try:
            surface.add(VolSkew.from_strikes(years, forward, bucket["k"], bucket["v"]))
        except ValueError:
            continue
    return surface
