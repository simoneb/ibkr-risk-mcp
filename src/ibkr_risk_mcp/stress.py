"""The stress engine: reprice the whole portfolio across underlying shocks.

What this reproduces is Risk Navigator's P&L-versus-underlying curve, and in
particular the **trough** — the worst point of the curve and the shock at which
it sits. That number is this server's primary output.

The model, stated plainly because the numbers are only as good as it is:

* every underlying moves by the same percentage at once, which is Risk
  Navigator's own default assumption ("equal percentage price changes for all
  underlyings"). Equity positions can be scaled by a beta; options and futures
  move with their own underlying one-for-one.
* options are repriced with Black-76 on a shocked forward, using **IB's**
  implied volatility rather than one implied locally, so the only local
  contribution is arithmetic.
* P&L is measured as *model price now* against *model price after the shock*,
  never model against market. The two differ by a residual on every contract —
  IB's model and this one do not agree to the cent — and differencing two
  market-vs-model errors would put that residual into a curve that should start
  at exactly zero.

The residual itself is not swept away: it is reported per position as
``modelVsMarket``, and the portfolio total is reconciled against
``NetLiquidation`` at zero shock. A curve that does not reconcile is returned
with ``reconciled: false`` and the residual attached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

from ib_async import Contract

from . import contracts as C
from . import marketdata as MD
from . import pricing
from .config import settings
from .connection import connection

log = logging.getLogger(__name__)

VOL_MODES = ("sticky_strike", "sticky_moneyness")

SCOPES = ("equity", "all")

#: How far the reconciliation may miss NetLiquidation before the result is
#: flagged. One percent is the prompt's threshold and a fair one: below it the
#: gap is quote staleness, above it something structural is wrong.
RECONCILE_TOLERANCE = 0.01


@dataclass
class StressConfig:
    shocks: Sequence[float]
    vol_mode: str = "sticky_strike"
    vol_bump: float = 0.0
    #: How the volatility *level* responds to the shock, in volatility points
    #: per 1% move of the underlying.
    #:
    #: This is the term both vol modes leave out. ``vol_bump`` is flat along the
    #: shock axis, and ``sticky_moneyness`` slides a strike along the *existing*
    #: smile without ever raising it. A real index does neither: a 20% fall
    #: takes at-the-money volatility with it, and a book that is net short
    #: options pays for that move on top of everything the curve already counts.
    #: With both slopes at zero — the default, and what every earlier result
    #: assumed — that P&L is silently set to nothing.
    #:
    #: ``vol_slope_down=1.0`` adds one volatility point per 1% fall, so a −20%
    #: shock reprices at +20 points. The two directions are separate parameters
    #: rather than one signed slope because the effect is not symmetric:
    #: volatility rises far harder on the way down than it gives back on the way
    #: up.
    #:
    #: Like ``bond_duration_years``, this is an input rather than a measurement,
    #: and the curve is only as good as the number handed to it.
    vol_slope_down: float = 0.0
    #: Volatility points *removed* per 1% rise. Positive means volatility falls
    #: as the market rallies, which is the usual direction.
    vol_slope_up: float = 0.0
    date_offset_days: int = 0
    betas: dict[str, float] = field(default_factory=dict)
    default_beta: float = 1.0
    #: Which underlyings are on the shock axis at all.
    #:
    #: ``equity`` — the default — puts only equity-underlying positions on the
    #: curve and excludes everything else outright. This is not a refinement,
    #: it is what makes the number mean anything: the engine's one assumption is
    #: that every underlying moves by the same percentage, and off the equity
    #: axis that is nonsense. A −20% shock on a CAD futures option prices
    #: USD/CAD going from 0.72 to 0.58. TWS Risk Navigator draws the same line
    #: in its Equity tab, and the two curves agree to within a fraction of a
    #: percent once it is drawn here too.
    #:
    #: ``all`` restores the old behaviour and shocks every underlying alike.
    #: Excluded positions are always listed, never dropped in silence.
    #:
    #: A bond rate shift is a *different* axis that the caller asked for
    #: explicitly, so ``bond_rate_shift_bp`` keeps working under either scope.
    scope: str = "equity"
    #: Per-symbol override of :func:`contracts.risk_group`, e.g.
    #: ``{"TLT": "rates", "GLD": "metals"}``. IB publishes no asset class for a
    #: bond or gold ETF quoted as a stock, so those land in ``equity`` unless
    #: they are named here.
    risk_groups: dict[str, str] = field(default_factory=dict)
    bond_rate_shift_bp: float = 0.0
    bond_duration_years: float = 5.0
    rate: float = field(default_factory=lambda: settings.risk_free_rate)
    #: Valuation date. Today unless pinned, which the tests do so that a
    #: recorded portfolio prices the same way next year as it does now.
    asof: date | None = None
    #: Whether sticky_moneyness may pull extra strikes from IB to build a smile
    #: when the portfolio itself has too few. Off by default: it is the only
    #: part of a stress run that issues market data requests beyond the
    #: positions, and on a large portfolio that is a lot of them.
    fetch_skew: bool = False

    def validate(self) -> None:
        if self.vol_mode not in VOL_MODES:
            raise ValueError(f"vol_mode must be one of {VOL_MODES}, got {self.vol_mode!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {self.scope!r}")
        if not self.shocks:
            raise ValueError("shocks must contain at least one value")
        if max(abs(s) for s in self.shocks) > 1.0:
            raise ValueError(
                "shocks are fractions, not percents: 0.05 is +5%. A value above 1.0 would "
                "mean the underlying more than doubling, which is almost certainly a "
                "misread of the units."
            )
        if max(abs(self.vol_slope_down), abs(self.vol_slope_up)) > 10.0:
            raise ValueError(
                "vol slopes are volatility points per 1% move: 1.0 means a 20% fall "
                "reprices at +20 points. A value above 10 would put hundreds of points on "
                "a moderate shock, which is almost certainly percent or basis points in "
                "the wrong units."
            )


@dataclass
class RiskUnit:
    """One repricable thing. Built from a real position or a hypothetical leg;
    the engine cannot tell the difference, which is what makes the what-if
    curve comparable to the base curve."""

    key: str
    label: str
    symbol: str
    asset_class: str
    position: float
    multiplier: float
    market_value: float | None
    #: Signed notional, set only for futures. IB's ``marketValue`` for a future
    #: is its mark-to-market, not something a percentage shock applies to, so
    #: the exposure is rebuilt here from price times multiplier times position.
    notional: float | None = None
    sec_type: str = ""
    #: Which risk factor this position responds to, from the underlying rather
    #: than the secType. ``asset_class`` says "option"; this says "fx".
    risk_group: str = "equity"
    #: The underlying instrument's own symbol — the future an option is written
    #: on, rather than the option's root. Carried so a beta can be keyed on it:
    #: ``{"ESZ6": 1.0}`` picks out one quarterly where ``{"ES": 1.0}`` takes
    #: every contract on the root.
    und_symbol: str | None = None
    strike: float | None = None
    right: str | None = None
    years: float | None = None
    iv: float | None = None
    und_price: float | None = None
    pv_dividend: float = 0.0
    #: True when ``und_price`` is already a forward (a futures option) and false
    #: when it is a spot that has to be carried to expiry (an equity option).
    underlying_is_forward: bool = False
    skew_key: tuple[str, str] | None = None
    hypothetical: bool = False
    note: str | None = None

    @property
    def priceable(self) -> bool:
        if self.asset_class != "option":
            return True
        return None not in (self.strike, self.right, self.years, self.iv, self.und_price)

    def forward(self, shock: float, years: float, rate: float) -> float:
        spot = float(self.und_price or 0.0) * (1.0 + shock)
        if self.underlying_is_forward:
            return spot
        return pricing.forward_from_spot(spot, years, rate, self.pv_dividend)

    def model_price(self, shock: float, iv: float, years: float, rate: float) -> float:
        return pricing.black76_price(
            self.forward(shock, years, rate),
            float(self.strike),
            years,
            iv,
            rate,
            str(self.right),
        )


# --------------------------------------------------------------------------
# building units
# --------------------------------------------------------------------------


def _skew_key(holding_symbol: str, contract: Contract) -> tuple[str, str]:
    return (holding_symbol, C.settlement_date(contract).isoformat())


def unit_from_holding(h: MD.Holding, asof: date | None = None) -> RiskUnit:
    contract = h.contract
    base = RiskUnit(
        key=str(contract.conId),
        label=contract.localSymbol or contract.symbol,
        symbol=contract.symbol,
        asset_class=h.asset_class,
        position=h.position,
        multiplier=h.multiplier,
        market_value=h.market_value,
        sec_type=contract.secType,
        und_symbol=h.und_symbol,
        risk_group=C.risk_group(contract),
    )
    if h.asset_class == "future":
        if h.market_price is not None:
            base.notional = h.position * h.multiplier * h.market_price
        else:
            base.note = "no price for this future; it is held flat across the curve"
        return base
    if h.asset_class != "option":
        return base
    greeks = h.greeks or {}
    base.strike = float(contract.strike) if contract.strike else None
    base.right = (contract.right or "")[:1].upper() or None
    base.years = C.years_to_expiry(contract, asof=asof)
    base.iv = greeks.get("impliedVol")
    base.und_price = greeks.get("undPrice")
    base.pv_dividend = greeks.get("pvDividend") or 0.0
    base.underlying_is_forward = contract.secType == "FOP"
    base.skew_key = _skew_key(contract.symbol, contract)
    if base.iv is None or base.und_price is None:
        base.note = h.greeks_error or "IB published no model greeks for this contract"
    elif greeks.get("source"):
        # A locally implied volatility carries its provenance all the way to the
        # position report. It is a worse number than IB's and the reader should
        # be able to see which rows it applies to.
        base.note = greeks["source"]
    return base


async def units_from_legs(
    legs: Sequence[C.Leg], asof: date | None = None
) -> tuple[list[RiskUnit], list[dict[str, Any]]]:
    """Resolve hypothetical legs and give them the same shape as positions.

    A hypothetical option is priced off IB's current model greeks for that
    contract, so "portfolio plus these legs" starts from the same volatilities
    as "portfolio", and the difference between the two curves is the structure
    and nothing else.
    """
    problems: list[dict[str, Any]] = []
    resolved: list[tuple[C.Leg, Contract]] = []
    for leg in legs:
        try:
            resolved.append((leg, await MD.resolve_leg(leg)))
        except Exception as exc:
            problems.append({"leg": leg.model_dump(exclude_none=True), "error": str(exc)})

    option_pairs = [(leg, c) for leg, c in resolved if C.is_option(c.secType)]
    greeks_rows = (
        await MD.model_greeks_batch([c for _, c in option_pairs]) if option_pairs else []
    )
    greeks_by_conid = {
        c.conId: row for (_, c), row in zip(option_pairs, greeks_rows)
    }

    units: list[RiskUnit] = []
    for leg, contract in resolved:
        klass = C.asset_class(contract.secType)
        multiplier = C.contract_multiplier(contract)
        qty = float(leg.signed_quantity)
        unit = RiskUnit(
            key=f"hypo:{contract.conId}",
            label=contract.localSymbol or contract.symbol,
            symbol=contract.symbol,
            asset_class=klass,
            position=qty,
            multiplier=multiplier,
            market_value=None,
            sec_type=contract.secType,
            risk_group=C.risk_group(contract),
            hypothetical=True,
        )
        if klass == "option":
            row = greeks_by_conid.get(contract.conId)
            greeks = row.greeks if row else None
            if not greeks:
                problems.append(
                    {
                        "leg": leg.model_dump(exclude_none=True),
                        "error": (row.error if row else "no greeks")
                        or "IB published no model greeks",
                    }
                )
                continue
            unit.strike = float(contract.strike)
            unit.right = (contract.right or "")[:1].upper()
            unit.years = C.years_to_expiry(contract, asof=asof)
            unit.iv = greeks["impliedVol"]
            unit.und_price = greeks["undPrice"]
            unit.pv_dividend = greeks.get("pvDividend") or 0.0
            unit.underlying_is_forward = contract.secType == "FOP"
            unit.skew_key = _skew_key(contract.symbol, contract)
            price = greeks.get("optPrice")
            unit.market_value = (price * qty * multiplier) if price is not None else None
        else:
            price = await MD.spot_price(contract)
            unit.market_value = (price * qty * multiplier) if price is not None else None
            if klass == "future":
                unit.notional = unit.market_value
            if price is None:
                unit.note = "no price for this leg; it contributes nothing to the curve"
        units.append(unit)
    return units, problems


# --------------------------------------------------------------------------
# volatility handling
# --------------------------------------------------------------------------


async def build_skews(
    units: Sequence[RiskUnit], cfg: StressConfig
) -> tuple[dict[tuple[str, str], pricing.VolSkew], list[str]]:
    """A smile per (underlying, settlement date), from the portfolio's own
    implied volatilities where it has enough of them.

    Sticky moneyness needs a slope. Three strikes on the same expiry is the
    minimum that has one; with fewer, the honest answer is that the portfolio
    does not pin down the skew, and the engine says so and falls back to sticky
    strike for that expiry rather than inventing a flat smile and presenting it
    as the alternative model.
    """
    warnings: list[str] = []
    if cfg.vol_mode != "sticky_moneyness":
        return {}, warnings

    grouped: dict[tuple[str, str], list[RiskUnit]] = {}
    for u in units:
        if u.asset_class == "option" and u.priceable and u.skew_key:
            grouped.setdefault(u.skew_key, []).append(u)

    skews: dict[tuple[str, str], pricing.VolSkew] = {}
    for key, group in grouped.items():
        strikes = [u.strike for u in group]
        vols = [u.iv for u in group]
        forward = group[0].forward(0.0, group[0].years or 0.0, cfg.rate)
        if len({round(float(s), 6) for s in strikes}) >= 3:
            skews[key] = pricing.VolSkew.from_strikes(
                group[0].years or 0.0, forward, strikes, vols
            )
            continue
        if cfg.fetch_skew:
            fetched = await _fetch_skew(group[0], key)
            if fetched is not None:
                skews[key] = fetched
                continue
        warnings.append(
            f"{key[0]} {key[1]}: only {len(set(strikes))} strike(s) held, which does not "
            "define a smile — this expiry was repriced sticky_strike. Pass "
            "fetch_skew=true to pull neighbouring strikes from IB."
        )
    return skews, warnings


async def _fetch_skew(unit: RiskUnit, key: tuple[str, str]) -> pricing.VolSkew | None:
    rows = await MD.skew_for(
        unit.symbol,
        Contract(
            symbol=unit.symbol,
            lastTradeDateOrContractMonth=key[1].replace("-", ""),
            tradingClass="",
        ),
        forward=float(unit.und_price or 0.0),
        # A futures option hangs off a future. Without this, a symbol that is
        # also a stock — ES is Eversource Energy as well as the E-mini —
        # resolves to the wrong instrument and returns a plausible skew for it.
        sec_type="FUT" if unit.sec_type == "FOP" else "STK",
    )
    quotes = [(r["strike"], r["impliedVol"]) for r in rows if r.get("impliedVol")]
    if len(quotes) < 3:
        return None
    return pricing.VolSkew.from_strikes(
        unit.years or 0.0,
        float(unit.und_price or 0.0),
        [q[0] for q in quotes],
        [q[1] for q in quotes],
    )


def shocked_vol(
    unit: RiskUnit,
    shock: float,
    cfg: StressConfig,
    skews: dict[tuple[str, str], pricing.VolSkew],
) -> float:
    """The volatility to reprice this contract at, under this shock.

    ``sticky_strike`` — the strike keeps its volatility. This is the default and
    it is what Risk Navigator's default curve does. It is also the conservative
    choice, since it does not let a falling market hand a position a lower
    volatility than the one it holds today.

    ``sticky_moneyness`` — the smile travels with the forward, so the strike
    picks up the volatility that currently belongs to its new moneyness. The
    lookup is done on the *unshocked* smile at ``ln(K/F')``, which is what makes
    it a lookup rather than a refit.

    Neither mode moves the *level* of the surface: one pins volatility to the
    strike, the other reads it off the smile the portfolio has today. The
    level's own response to the shock is :func:`vol_response`, and it is added
    on top of whichever mode is in use — the two answer different questions and
    do not overlap.

    ``shock`` here is the move of *this* position's underlying, already scaled
    by its beta, so a position attenuated on the shock axis gets an attenuated
    volatility response to match.
    """
    base = float(unit.iv or 0.0)
    if cfg.vol_mode == "sticky_moneyness" and unit.skew_key in skews:
        skew = skews[unit.skew_key]
        new_forward = unit.forward(shock, unit.years or 0.0, cfg.rate)
        base = skew.at_strike(float(unit.strike), forward=new_forward)
    return max(base + cfg.vol_bump + vol_response(shock, cfg), pricing.MIN_VOL)


def vol_response(shock: float, cfg: StressConfig) -> float:
    """Volatility points the shock itself adds, as a fraction of 1.

    The slopes are quoted per 1% move and the shock is a fraction, so the two
    conversions cancel: a slope of 1.0 against a −0.20 shock is +0.20, twenty
    volatility points. Zero on both sides leaves the level where the vol mode
    put it, which is what every result before this parameter existed assumed.

    **This is a parallel shift, and on a wing-heavy book that is the wrong
    shape.** A real surface steepens in a sell-off: the far out-of-the-money
    puts pick up much more volatility than the money does. Measured on a live
    1-2-1 ratio spread book, net short vega at the money, the difference is not
    subtle — a parallel rise costs money at every shock and the curve only ever
    gets worse, whereas letting the wings take more eventually turns it back up
    below the trough. A steepening term was tried and removed: the values that
    reproduced the shape priced the long wings above 100% implied volatility,
    which is curve fitting rather than modelling. So this stays parallel, stays
    an input rather than a measurement, and says so in `warnings`.
    """
    if shock < 0:
        level = cfg.vol_slope_down * (-shock)
    else:
        level = -cfg.vol_slope_up * shock
    return level


# --------------------------------------------------------------------------
# the curve
# --------------------------------------------------------------------------


def unit_risk_group(unit: RiskUnit, cfg: StressConfig) -> str:
    """The unit's risk group, after any per-symbol override."""
    for key in (unit.label, unit.symbol, unit.und_symbol):
        if key and key in cfg.risk_groups:
            return str(cfg.risk_groups[key])
    return unit.risk_group


def in_scope(unit: RiskUnit, cfg: StressConfig) -> bool:
    return cfg.scope == "all" or unit_risk_group(unit, cfg) == "equity"


def _beta(unit: RiskUnit, cfg: StressConfig) -> float:
    """How much of the shock this position sees. Most specific key wins.

    ``label`` is the local symbol (``ESZ6 P5800``), ``symbol`` the root
    (``ES``), ``und_symbol`` the instrument an option is written on (``ESZ6``).
    The root is what most callers want; the other two are there for a book that
    holds two contracts on one root and has to tell them apart.
    """
    for key in (unit.label, unit.symbol, unit.und_symbol):
        if key and key in cfg.betas:
            return float(cfg.betas[key])
    return float(cfg.default_beta)


def unit_pnl(
    unit: RiskUnit,
    shock: float,
    cfg: StressConfig,
    skews: dict[tuple[str, str], pricing.VolSkew],
) -> float:
    """P&L for one position under one shock.

    The beta scales **the shock**, not the P&L. An option is therefore repriced
    at the move its own underlying actually makes, with strike, smile and
    convexity all measured at the forward it would reach; scaling the P&L
    instead would price it at the index move and then shrink the answer, which
    for anything convex is a different number and a wrong one.

    This widens the scope of ``betas``, which used to reach equity alone on the
    grounds that an option should move with its own underlying. It still does —
    but only when that underlying is the one being shocked. A short EUR strangle
    in an S&P scenario is not a 20%-down position, and with the beta confined to
    equity there was no way to say so. ``default_beta`` is 1.0, so a run that
    passes no betas prices exactly as it did before.
    """
    # A rate shift is a separate axis the caller asked for by name, so it is
    # answered before the equity scope has any say — a bond is off the equity
    # curve either way, but silently ignoring bond_rate_shift_bp because of it
    # would be a different and surprising thing.
    if unit.asset_class == "bond":
        if not cfg.bond_rate_shift_bp:
            return 0.0
        dy = cfg.bond_rate_shift_bp / 10_000.0
        return -cfg.bond_duration_years * dy * float(unit.market_value or 0.0)

    # Off the axis entirely, not attenuated to zero: a beta of 0 would still let
    # vega and theta through, and an excluded position must contribute nothing
    # at all. What it does contribute in its own scenario is not modelled here,
    # which is what the warning says.
    if not in_scope(unit, cfg):
        return 0.0

    eff = shock * _beta(unit, cfg)

    if unit.asset_class == "option":
        if not unit.priceable:
            return 0.0
        years_now = float(unit.years)
        years_then = max(years_now - cfg.date_offset_days / C.DAYS_PER_YEAR, pricing.MIN_YEARS)
        base = unit.model_price(0.0, float(unit.iv), years_now, cfg.rate)
        shocked = unit.model_price(eff, shocked_vol(unit, eff, cfg, skews), years_then, cfg.rate)
        return (shocked - base) * unit.position * unit.multiplier

    if unit.asset_class == "equity":
        return float(unit.market_value or 0.0) * eff

    if unit.asset_class == "future":
        return float(unit.notional or 0.0) * eff

    return 0.0


def _parabolic_trough(points: list[tuple[float, float]]) -> dict[str, Any] | None:
    """Refine the grid minimum by fitting a parabola through it and its two
    neighbours.

    The true trough almost never lands on a grid point. This says where it
    actually is, and is labelled ``interpolated`` so it is never mistaken for a
    repriced value.
    """
    if len(points) < 3:
        return None
    i = min(range(len(points)), key=lambda j: points[j][1])
    if i == 0 or i == len(points) - 1:
        return None
    (x0, y0), (x1, y1), (x2, y2) = points[i - 1], points[i], points[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-15:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a <= 0:
        return None
    x = -b / (2 * a)
    if not (x0 <= x <= x2):
        return None
    c = (
        x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2
    ) / denom
    return {
        "shock": round(float(x), 6),
        "pnl": round(float(a * x * x + b * x + c), 2),
        "interpolated": True,
    }


def trough_of(points: list[tuple[float, float]]) -> dict[str, Any]:
    """The minimum of a curve, labelled when it sits on the edge of the range.

    A minimum on the first or last shock is not a trough, it is the edge of the
    window — the curve was still falling when the range ran out. Reporting it
    as "the trough" understates the risk by however much lies beyond it, which
    is exactly the number the caller asked for.
    """
    worst = min(points, key=lambda p: p[1])
    out: dict[str, Any] = {"shock": float(worst[0]), "pnl": round(float(worst[1]), 2)}
    if len(points) > 1 and worst[0] in (points[0][0], points[-1][0]):
        out["atRangeEdge"] = True
        out["note"] = (
            f"The curve is still falling at {worst[0]:+.0%}, the end of the range asked "
            "for, so this is the worst point *in the window*, not the trough. Widen "
            "`shocks` to find where it turns, if it turns at all within a plausible "
            "range."
        )
    return out


def run_curve(
    units: Sequence[RiskUnit],
    cfg: StressConfig,
    skews: dict[tuple[str, str], pricing.VolSkew],
) -> dict[str, Any]:
    """The P&L curve and its trough for one set of units."""
    rows: list[dict[str, Any]] = []
    for shock in cfg.shocks:
        by_class: dict[str, float] = {}
        by_symbol: dict[str, float] = {}
        total = 0.0
        for unit in units:
            pnl = unit_pnl(unit, shock, cfg, skews)
            total += pnl
            by_class[unit.asset_class] = by_class.get(unit.asset_class, 0.0) + pnl
            by_symbol[unit.symbol] = by_symbol.get(unit.symbol, 0.0) + pnl
        rows.append(
            {
                "shock": round(float(shock), 6),
                "pnl_total": round(float(total), 2),
                "pnl_by_asset_class": {k: round(float(v), 2) for k, v in sorted(by_class.items())},
                "pnl_by_symbol": {k: round(float(v), 2) for k, v in sorted(by_symbol.items())},
            }
        )
    points = [(r["shock"], r["pnl_total"]) for r in rows]
    best = max(points, key=lambda p: p[1])
    out: dict[str, Any] = {
        "curve": rows,
        "trough": trough_of(points),
        "peak": {"shock": float(best[0]), "pnl": round(float(best[1]), 2)},
    }
    refined = _parabolic_trough(points)
    if refined:
        out["troughRefined"] = refined
    return out


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------


def reconcile(holdings: Sequence[MD.Holding], ib, account: str) -> dict[str, Any]:
    """Rebuild NetLiquidation from the positions and say whether it matches.

    The identity is cash, plus the market value of the securities, plus the
    *unrealised P&L* of the futures — a future has no market value to add,
    since its variation margin settles daily, and adding its notional instead
    is a mistake large enough to be obvious once it is checked. Bonds are taken
    at IB's market value rather than quantity times price, which for a
    percentage-of-nominal quote would be a hundred times too large.

    A residual over one percent returns ``reconciled: false``. Nothing derived
    from a portfolio that does not reconcile should be presented as fact.
    """
    net_liq = MD.account_number(ib, account, "NetLiquidation")
    cash = MD.account_number(ib, account, "TotalCashValue")

    securities = sum(
        float(h.market_value or 0.0) for h in holdings if h.asset_class in ("option", "equity", "bond")
    )
    futures_pnl = sum(
        float(h.unrealized_pnl or 0.0) for h in holdings if h.asset_class == "future"
    )
    missing_values = [
        h.contract.localSymbol or h.contract.symbol
        for h in holdings
        if h.market_value is None and h.asset_class != "cash"
    ]

    out: dict[str, Any] = {
        "netLiquidation": net_liq,
        "totalCashValue": cash,
        "securitiesMarketValue": round(securities, 2),
        "futuresUnrealizedPnl": round(futures_pnl, 2),
        "positionsMissingValue": missing_values,
    }
    if net_liq is None or cash is None:
        out["reconciled"] = False
        out["reason"] = (
            "IB did not report NetLiquidation or TotalCashValue for this account, so the "
            "portfolio total could not be checked against anything."
        )
        return out

    derived = cash + securities + futures_pnl
    residual = derived - net_liq
    out["derivedNetLiquidation"] = round(derived, 2)
    out["residual"] = round(residual, 2)
    out["residualPct"] = round(residual / net_liq, 6) if net_liq else None
    out["reconciled"] = bool(net_liq) and abs(residual) <= RECONCILE_TOLERANCE * abs(net_liq)
    if not out["reconciled"]:
        out["reason"] = (
            "The positions do not add up to NetLiquidation within 1%. Treat the P&L curve "
            "as indicative only: something is missing from the snapshot (an unpriced "
            "position, a second account, or a currency this server did not convert)."
        )
    return out


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def _position_report(units: Sequence[RiskUnit], cfg: StressConfig) -> list[dict[str, Any]]:
    """Per-position diagnostics, including how far the local model sits from
    IB's own price. This is the number that says whether the curve can be
    believed at all."""
    out = []
    for u in units:
        beta = _beta(u, cfg)
        row: dict[str, Any] = {
            "key": u.key,
            "label": u.label,
            "symbol": u.symbol,
            "undSymbol": u.und_symbol,
            "assetClass": u.asset_class,
            "riskGroup": unit_risk_group(u, cfg),
            # Only worth a line when it is False; a column of True is noise.
            "offAxis": True if not in_scope(u, cfg) else None,
            # Only when it is doing something. A column of 1.0s buries the rows
            # where a beta actually moved the answer.
            "beta": beta if beta != 1.0 else None,
            "position": u.position,
            "multiplier": u.multiplier,
            "marketValue": u.market_value,
            "hypothetical": u.hypothetical or None,
        }
        if u.asset_class == "option":
            row.update(
                {
                    "strike": u.strike,
                    "right": u.right,
                    "yearsToExpiry": round(u.years, 6) if u.years else None,
                    "impliedVol": u.iv,
                    "undPrice": u.und_price,
                }
            )
            if u.priceable:
                model = u.model_price(0.0, float(u.iv), float(u.years), cfg.rate)
                row["modelPrice"] = round(float(model), 4)
                if u.market_value is not None and u.position:
                    market_unit = u.market_value / (u.position * u.multiplier)
                    row["marketPrice"] = round(float(market_unit), 4)
                    row["modelVsMarket"] = round(float(model - market_unit), 4)
        if u.note:
            row["note"] = u.note
        out.append({k: v for k, v in row.items() if v is not None})
    return out


def excluded_report(units: Sequence[RiskUnit], cfg: StressConfig) -> list[dict[str, Any]]:
    """Every position the scope took off the axis, with what it is worth.

    Excluding non-equity underlyings is what makes the curve mean something, but
    it also means the curve is no longer the whole account — and the reader has
    to be able to see the difference rather than infer it. Grouped by risk
    factor, because that is the unit in which the missing exposure would have to
    be modelled.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for unit in units:
        if in_scope(unit, cfg):
            continue
        group = unit_risk_group(unit, cfg)
        row = grouped.setdefault(
            group, {"riskGroup": group, "symbols": set(), "marketValue": 0.0, "positions": 0}
        )
        row["symbols"].add(unit.symbol)
        row["marketValue"] += float(unit.market_value or 0.0)
        row["positions"] += 1
    return [
        {
            "riskGroup": row["riskGroup"],
            "symbols": sorted(row["symbols"]),
            "positions": row["positions"],
            "marketValue": round(row["marketValue"], 2),
        }
        for _, row in sorted(grouped.items())
    ]


def scope_warnings(units: Sequence[RiskUnit], cfg: StressConfig) -> list[str]:
    """What the beta and the volatility slope did to the curve's meaning.

    Both are improvements that make a result *less* self-evident. A beta below
    one takes a position most of the way off the axis, and a reader who does not
    know that will read the trough as covering it. A volatility slope adds a
    real P&L term from a number nobody measured. Neither should have to be
    inferred from `assumptions`.
    """
    out: list[str] = []

    excluded = excluded_report(units, cfg)
    if excluded:
        detail = "; ".join(
            f"{r['riskGroup']} ({', '.join(r['symbols'])}, {r['marketValue']:,.0f})"
            for r in excluded
        )
        out.append(
            f"scope={cfg.scope!r} kept only equity underlyings on the axis. Left off "
            f"entirely: {detail}. This is deliberate — one percentage shock applied to "
            "every underlying at once is meaningless off the equity axis, and a currency "
            "future shocked 20% prices an exchange rate move that has never happened — but "
            "it does mean the curve is no longer the whole account. Those positions carry "
            "risk that is not anywhere on it. Pass scope='all' to put them back."
        )

    attenuated = sorted(
        {
            u.symbol
            for u in units
            if u.asset_class in ("option", "future") and _beta(u, cfg) != 1.0
        }
    )
    if attenuated:
        out.append(
            f"Positions on {', '.join(attenuated)} were repriced at a beta-scaled move of "
            "their own underlying rather than at the full shock. That is what makes this "
            "curve readable when one underlying does not belong on the axis — but it does "
            "not *measure* those positions, it stands them down. A short strangle "
            "attenuated to a fifth of the equity move still carries its whole gap risk, "
            "and none of that risk is on this curve. Their contribution here is in "
            "`pnl_by_symbol`; their own scenario is not modelled by this server."
        )

    if cfg.vol_slope_down or cfg.vol_slope_up:
        out.append(
            f"Implied volatility responds to the shock at {cfg.vol_slope_down:g} point(s) "
            f"per 1% down and {cfg.vol_slope_up:g} per 1% up. That is your input, not a "
            "measurement, and two things about how it is applied bound it: it is a "
            "*parallel* shift, while a real surface also steepens in a sell-off — which "
            "understates a long out-of-the-money put — and it is flat across tenors, while "
            "a 120-day volatility moves less than the front month, so a slope calibrated "
            "on the front overstates the move on a long-dated position."
        )
    elif any(u.asset_class == "option" and u.priceable for u in units):
        out.append(
            "The volatility level is held constant along the shock axis: this curve prices "
            "the move in the underlying and not the move in volatility that comes with it. "
            "For a net short option book that is the optimistic half of the answer. Set "
            "vol_slope_down (1.0 is one volatility point per 1% fall) to put it in."
        )
    return out


def assumptions(cfg: StressConfig) -> dict[str, Any]:
    return {
        "volMode": cfg.vol_mode,
        "volBump": cfg.vol_bump,
        "volSlopeDown": cfg.vol_slope_down,
        "volSlopeUp": cfg.vol_slope_up,
        "volatilityLevel": (
            f"responds to the shock at {cfg.vol_slope_down:g} point(s) per 1% down and "
            f"{cfg.vol_slope_up:g} per 1% up, as a parallel shift of the whole surface"
            if (cfg.vol_slope_down or cfg.vol_slope_up)
            else "held constant along the shock axis — the volatility level does not "
            "respond to the move at all. Set vol_slope_down to price the volatility a "
            "sell-off brings with it; a net short option book loses money on that term "
            "and none of it is in this curve."
        ),
        "dateOffsetDays": cfg.date_offset_days,
        "riskFreeRate": cfg.rate,
        "scope": cfg.scope,
        "scopeMeaning": (
            "only equity underlyings are on the shock axis; FX, rates and anything else is "
            "excluded outright and listed under `excluded`. This matches what TWS Risk "
            "Navigator's Equity tab does."
            if cfg.scope == "equity"
            else "every underlying is on the shock axis, moved by the same percentage — "
            "including FX and rates, where that assumption does not hold"
        ),
        "riskGroupOverrides": cfg.risk_groups or None,
        "defaultBeta": cfg.default_beta,
        "betas": cfg.betas or None,
        "betaScope": "the beta scales the shock on every position that responds to one — "
        "options and futures as well as equities. An option is repriced at its own "
        "beta-scaled move of its underlying, not at the index move.",
        "bondRateShiftBp": cfg.bond_rate_shift_bp,
        "bondDurationYears": cfg.bond_duration_years if cfg.bond_rate_shift_bp else None,
        "dayCount": "ACT/365 to the settlement date",
        "model": "Black-76 on a shocked forward; equity options carried from spot "
        "using IB's pvDividend",
        "underlyingCorrelation": "all underlyings shocked by the same percentage "
        "simultaneously (Risk Navigator's default assumption), each scaled by its beta",
    }


async def stress_portfolio(cfg: StressConfig) -> dict[str, Any]:
    cfg.validate()
    ib = await connection.get()
    holdings = await MD.load_holdings(with_greeks=True)
    units = [unit_from_holding(h, cfg.asof) for h in holdings]
    skews, warnings = await build_skews(units, cfg)

    unpriceable = [u for u in units if u.asset_class == "option" and not u.priceable]
    if unpriceable:
        warnings.append(
            f"{len(unpriceable)} option position(s) had no usable model greeks and were "
            "held flat across every shock — the curve understates the risk by whatever "
            "they carry. They are listed in `positions` with a note."
        )
    implied_locally = [
        u for u in units if u.asset_class == "option" and u.priceable and u.note
    ]
    if implied_locally:
        warnings.append(
            f"{len(implied_locally)} option position(s) are repriced from a volatility "
            "implied locally off the mark price, because IB published no model greeks "
            "for them — commonly a missing market data entitlement. They are in the "
            "curve, but they are this server's numbers rather than IB's. See `positions`."
        )
    other = {u.symbol for u in units if u.asset_class == "other"}
    if other:
        warnings.append(
            f"Held flat because this server does not model them: {', '.join(sorted(other))}."
        )
    warnings.extend(scope_warnings(units, cfg))

    result = run_curve(units, cfg, skews)
    reconciliation = reconcile(holdings, ib, connection.require_account())
    return {
        **result,
        "reconciliation": reconciliation,
        "reconciled": reconciliation.get("reconciled", False),
        "assumptions": assumptions(cfg),
        "excluded": excluded_report(units, cfg),
        "positions": _position_report(units, cfg),
        "warnings": warnings,
    }


async def stress_whatif(legs: Sequence[C.Leg], cfg: StressConfig) -> dict[str, Any]:
    """The base curve, the curve with the legs added, and the difference.

    Three troughs come back, and the third — the trough of the *difference* —
    is not the difference of the first two: adding a put can move where the
    worst point sits as well as how deep it is, and reading only the depths
    hides that.
    """
    cfg.validate()
    ib = await connection.get()
    holdings = await MD.load_holdings(with_greeks=True)
    base_units = [unit_from_holding(h, cfg.asof) for h in holdings]
    leg_units, leg_problems = await units_from_legs(legs, cfg.asof)
    combined = base_units + leg_units

    skews, warnings = await build_skews(combined, cfg)
    if leg_problems:
        warnings.append(
            f"{len(leg_problems)} leg(s) could not be priced and are not in the "
            "'with legs' curve — see `legProblems`. The comparison below is of the "
            "portfolio against the legs that did resolve."
        )
    warnings.extend(scope_warnings(combined, cfg))

    base = run_curve(base_units, cfg, skews)
    withlegs = run_curve(combined, cfg, skews)
    diff_points = [
        {
            "shock": b["shock"],
            "pnl_total": round(float(w["pnl_total"] - b["pnl_total"]), 2),
        }
        for b, w in zip(base["curve"], withlegs["curve"])
    ]
    diff_pairs = [(r["shock"], r["pnl_total"]) for r in diff_points]
    difference: dict[str, Any] = {
        "curve": diff_points,
        "trough": trough_of(diff_pairs),
    }
    refined = _parabolic_trough(diff_pairs)
    if refined:
        difference["troughRefined"] = refined

    reconciliation = reconcile(holdings, ib, connection.require_account())
    return {
        "base": base,
        "withLegs": withlegs,
        "difference": difference,
        "troughs": {
            "base": base["trough"],
            "withLegs": withlegs["trough"],
            "difference": difference["trough"],
        },
        "legs": _position_report(leg_units, cfg),
        "legProblems": leg_problems,
        "excluded": excluded_report(combined, cfg),
        "reconciliation": reconciliation,
        "reconciled": reconciliation.get("reconciled", False),
        "assumptions": assumptions(cfg),
        "warnings": warnings,
    }
