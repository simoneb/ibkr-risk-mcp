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

#: How far the reconciliation may miss NetLiquidation before the result is
#: flagged. One percent is the prompt's threshold and a fair one: below it the
#: gap is quote staleness, above it something structural is wrong.
RECONCILE_TOLERANCE = 0.01


@dataclass
class StressConfig:
    shocks: Sequence[float]
    vol_mode: str = "sticky_strike"
    vol_bump: float = 0.0
    date_offset_days: int = 0
    betas: dict[str, float] = field(default_factory=dict)
    default_beta: float = 1.0
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
        if not self.shocks:
            raise ValueError("shocks must contain at least one value")
        if max(abs(s) for s in self.shocks) > 1.0:
            raise ValueError(
                "shocks are fractions, not percents: 0.05 is +5%. A value above 1.0 would "
                "mean the underlying more than doubling, which is almost certainly a "
                "misread of the units."
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
    it is what Risk Navigator's default curve does; it is also the conservative
    choice for a short-put book, since it does not let a falling market hand the
    position a lower volatility.

    ``sticky_moneyness`` — the smile travels with the forward, so the strike
    picks up the volatility that currently belongs to its new moneyness. The
    lookup is done on the *unshocked* smile at ``ln(K/F')``, which is what makes
    it a lookup rather than a refit.
    """
    base = float(unit.iv or 0.0)
    if cfg.vol_mode == "sticky_moneyness" and unit.skew_key in skews:
        skew = skews[unit.skew_key]
        new_forward = unit.forward(shock, unit.years or 0.0, cfg.rate)
        base = skew.at_strike(float(unit.strike), forward=new_forward)
    return max(base + cfg.vol_bump, pricing.MIN_VOL)


# --------------------------------------------------------------------------
# the curve
# --------------------------------------------------------------------------


def _beta(unit: RiskUnit, cfg: StressConfig) -> float:
    return float(cfg.betas.get(unit.symbol, cfg.betas.get(unit.label, cfg.default_beta)))


def unit_pnl(
    unit: RiskUnit,
    shock: float,
    cfg: StressConfig,
    skews: dict[tuple[str, str], pricing.VolSkew],
) -> float:
    if unit.asset_class == "option":
        if not unit.priceable:
            return 0.0
        years_now = float(unit.years)
        years_then = max(years_now - cfg.date_offset_days / C.DAYS_PER_YEAR, pricing.MIN_YEARS)
        base = unit.model_price(0.0, float(unit.iv), years_now, cfg.rate)
        shocked = unit.model_price(shock, shocked_vol(unit, shock, cfg, skews), years_then, cfg.rate)
        return (shocked - base) * unit.position * unit.multiplier

    if unit.asset_class == "equity":
        return float(unit.market_value or 0.0) * shock * _beta(unit, cfg)

    if unit.asset_class == "future":
        return float(unit.notional or 0.0) * shock

    if unit.asset_class == "bond":
        if not cfg.bond_rate_shift_bp:
            return 0.0
        dy = cfg.bond_rate_shift_bp / 10_000.0
        return -cfg.bond_duration_years * dy * float(unit.market_value or 0.0)

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
            "`shocks` to find where it turns — a book that is net short downside may "
            "simply keep losing."
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
        row: dict[str, Any] = {
            "key": u.key,
            "label": u.label,
            "symbol": u.symbol,
            "assetClass": u.asset_class,
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


def assumptions(cfg: StressConfig) -> dict[str, Any]:
    return {
        "volMode": cfg.vol_mode,
        "volBump": cfg.vol_bump,
        "dateOffsetDays": cfg.date_offset_days,
        "riskFreeRate": cfg.rate,
        "defaultBeta": cfg.default_beta,
        "betas": cfg.betas or None,
        "bondRateShiftBp": cfg.bond_rate_shift_bp,
        "bondDurationYears": cfg.bond_duration_years if cfg.bond_rate_shift_bp else None,
        "dayCount": "ACT/365 to the settlement date",
        "model": "Black-76 on a shocked forward; equity options carried from spot "
        "using IB's pvDividend",
        "underlyingCorrelation": "all underlyings shocked by the same percentage "
        "simultaneously (Risk Navigator's default assumption)",
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

    result = run_curve(units, cfg, skews)
    reconciliation = reconcile(holdings, ib, connection.require_account())
    return {
        **result,
        "reconciliation": reconciliation,
        "reconciled": reconciliation.get("reconciled", False),
        "assumptions": assumptions(cfg),
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
        "reconciliation": reconciliation,
        "reconciled": reconciliation.get("reconciled", False),
        "assumptions": assumptions(cfg),
        "warnings": warnings,
    }
