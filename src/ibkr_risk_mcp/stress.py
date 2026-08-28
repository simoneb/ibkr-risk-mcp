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
import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

from ib_async import Contract

from . import calibration
from . import contracts as C
from . import marketdata as MD
from . import pricing
from .config import settings
from .connection import connection

log = logging.getLogger(__name__)

VOL_MODES = ("sticky_strike", "sticky_moneyness")

SCOPES = ("equity", "all")

#: Which per-point P&L breakdowns a curve carries.
#:
#: ``symbol`` is what every result has always returned and remains the default,
#: because the responses are already large and a second dictionary per point is
#: not free: a book on nine expiries and thirty-three symbols pays for it
#: twenty-six times over. ``expiry`` answers the question ``symbol`` cannot on
#: an options book — with every ES expiry collapsed under one "ES" key, "which
#: expiry is holding the trough down" had to be reconstructed by hand from the
#: position list. ``none`` is there for a caller that only wants the curve.
BREAKDOWNS = ("symbol", "expiry", "both", "none")

#: How far the reconciliation may miss NetLiquidation before the result is
#: flagged. One percent is the prompt's threshold and a fair one: below it the
#: gap is quote staleness, above it something structural is wrong.
RECONCILE_TOLERANCE = 0.01

#: Decay of ``VR(t)`` in the vol_coord model. **Fitted, not published.** IB
#: documents that the term-structure response exists and is decreasing; this
#: value came from one Risk Navigator screenshot, on one account, from nine
#: points read off a chart by eye. Every run that uses it says so.
DEFAULT_VOL_COORD_DECAY = 4.736

#: How far out in tenor :data:`DEFAULT_VOL_COORD_DECAY` was ever constrained.
#: The book it was fitted on held nothing past four months, so beyond this the
#: exponential is extrapolation — and it extrapolates to zero, which would
#: price a long-dated option as carrying no volatility risk at all. Positions
#: past it are named in `warnings` rather than silently repriced.
DEFAULT_VOL_COORD_CALIBRATED_YEARS = 0.345


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
    #: Use IB's own volatility-coordinated model instead of the additive slopes.
    #:
    #: Risk Navigator's second curve — the one it labels ``Vol.Coord.`` — moves
    #: volatility as a *deterministic function of the price shock*, and IB
    #: documents its shape: the nominal shock is ``-X`` for a rise and ``-10X``
    #: for a fall, applied **relatively** rather than in points, then damped
    #: across tenors by a response function ``VR(t)`` that is 1 at zero and
    #: decreasing.
    #:
    #: Relative is the part the additive slopes cannot imitate. Multiplying
    #: every volatility by the same factor puts more *points* on a wing already
    #: quoted at 41% than on a 31% at-the-money, so the surface steepens on its
    #: own — which is what a real sell-off does and what a parallel shift
    #: cannot produce at any slope.
    vol_coord: bool = False
    #: The relative volatility shock per unit of price fall, ``10`` in IB's
    #: documented form: a 20% fall triples volatility before damping.
    vol_coord_down: float = 10.0
    #: Per unit of price rise, ``1`` in IB's documented form.
    vol_coord_up: float = 1.0
    #: Decay of ``VR(t) = exp(-decay * t)``. IB documents that the function
    #: exists and is decreasing but not what it is, so this is **fitted, not
    #: published**: it reproduces a live Risk Navigator Vol.Coord. curve to an
    #: RMS of roughly 3% of that curve's own depth, against four times worse
    #: for the best two-parameter additive fit. It is one number calibrated on
    #: one book, and it should be refitted against your own Risk Navigator
    #: before being trusted on another.
    #:
    #: Defaults to whatever ``calibrate_vol_coord`` last stored, and to the
    #: factory fit only when nothing has been stored. A calibration that had to
    #: be quoted back on every call was one nobody quoted back.
    vol_coord_decay: float = field(
        default_factory=lambda: calibration.decay(DEFAULT_VOL_COORD_DECAY)
    )
    #: Tenor beyond which the decay above was never calibrated. Positions past
    #: it still price, and are named in `warnings` so the extrapolation is not
    #: silent.
    vol_coord_calibrated_to_years: float = field(
        default_factory=lambda: calibration.calibrated_to_years(
            DEFAULT_VOL_COORD_CALIBRATED_YEARS
        )
    )
    date_offset_days: int = 0
    #: Which per-point P&L breakdowns to return; see :data:`BREAKDOWNS`.
    breakdown: str = "symbol"
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
        if self.breakdown not in BREAKDOWNS:
            raise ValueError(
                f"breakdown must be one of {BREAKDOWNS}, got {self.breakdown!r}"
            )
        if self.date_offset_days < 0:
            raise ValueError(
                "date_offset_days rolls the valuation date forward; a negative value "
                "would value the book in the past, where this engine has no prices. "
                f"Got {self.date_offset_days}."
            )
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
class VolScenario:
    """One volatility regime on the same portfolio and the same shock axis.

    A scenario changes nothing about the positions or the surface — only how
    the volatility *level* answers the shock. That is deliberate: the point of
    running several is to make the vol assumption a visible axis of the result
    rather than a choice buried in one number, so everything else has to be
    held identical between them.
    """

    name: str
    #: Volatility points added per 1% fall, as in :attr:`StressConfig.vol_slope_down`.
    #: Zero is the constant-volatility curve.
    vol_slope_down: float = 0.0
    #: Volatility points removed per 1% rise. Positive means volatility falls
    #: into a rally, which is the usual direction.
    vol_slope_up: float = 0.0
    #: A flat shift of the whole surface for this regime, in points, applied at
    #: every shock including zero. Unlike the slopes this moves the P&L at zero
    #: shock, so a scenario that uses it does not start the curve at zero.
    vol_bump: float = 0.0
    #: Use IB's relative, term-damped Vol.Coord. model for this curve instead
    #: of the additive slopes above. The two do not combine: with this on, the
    #: slopes are ignored and ``vol_bump`` still applies.
    vol_coord: bool = False

    def overrides(self) -> dict[str, Any]:
        return {
            "vol_slope_down": self.vol_slope_down,
            "vol_slope_up": self.vol_slope_up,
            "vol_bump": self.vol_bump,
            "vol_coord": self.vol_coord,
        }


#: −40% to +10% in 2% steps. Wide enough on the downside that a short-gamma
#: book's trough falls inside the window rather than on its edge, and short on
#: the upside because that half of the curve is nearly straight.
DEFAULT_SHOCKS: tuple[float, ...] = tuple(round(-0.40 + i * 0.02, 4) for i in range(26))

#: The two curves Risk Navigator itself draws, and nothing invented on top.
#:
#: ``const`` is the constant-volatility case — its blue line, and the one with
#: an external check. ``vol_coord`` is IB's own volatility-coordinated model,
#: its ``Vol.Coord.`` curve, in the documented relative form.
#:
#: An earlier default set carried two additive slopes, 0.7 and 1.4 points per
#: 1% fall. They are gone because they were invented rather than measured, and
#: because a parallel shift is the wrong *shape*: on a ratio book it made
#: the curve monotonically worse where Risk Navigator turns it back up. Additive
#: slopes remain available for a regime you want to state by hand.
DEFAULT_VOL_SCENARIOS: tuple["VolScenario", ...] = (
    VolScenario("const"),
    VolScenario("vol_coord", vol_coord=True),
)


# --------------------------------------------------------------------------
# when the curve is valued
# --------------------------------------------------------------------------


def parse_valuation_date(raw: str) -> date:
    """An ISO date, and nothing cleverer.

    No weekend or holiday adjustment: "the curve at 30 September" means the
    30th, and a server that silently answered for the 2nd of October because
    the 30th happened to be a Saturday would be lying about the axis it drew.
    Time to expiry is ACT/365 throughout, so a non-trading valuation date is
    arithmetically fine and simply carries a day more decay than the previous
    session close.
    """
    try:
        return date.fromisoformat(raw.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"valuation dates are ISO calendar dates, e.g. '2026-09-30'; got {raw!r}"
        ) from exc


def offset_to(target: date, asof: date | None = None) -> int:
    """Calendar days from the valuation baseline to ``target``.

    An absolute date is sugar over ``date_offset_days``, not a second mode:
    both curves are still measured *from today*, priced at today's spot and
    today's volatility with time rolled forward. That is what makes the answer
    "what this book earns or loses between now and then", which is the question
    a date was being counted out by hand to ask.
    """
    base = asof or date.today()
    days = (target - base).days
    if days < 0:
        raise ValueError(
            f"{target.isoformat()} is {-days} day(s) before the valuation baseline "
            f"{base.isoformat()}. This engine rolls time forward from today's prices; "
            "it cannot value a book in the past."
        )
    return days


def resolve_offsets(
    *,
    date_offset_days: int = 0,
    valuation_date: str | None = None,
    date_offsets: Sequence[int] | None = None,
    valuation_dates: Sequence[str] | None = None,
    asof: date | None = None,
) -> list[int]:
    """The one list of day offsets a run is valued at, from whichever of the
    four ways of saying it the caller used.

    They are mutually exclusive rather than combined. Two of them given at once
    is a caller who believes one thing and asked for another, and picking a
    winner would hide that: ``date_offset_days=3`` beside
    ``valuation_date='2026-09-30'`` has no reading that is obviously right.
    """
    given = [
        name
        for name, value in (
            ("date_offset_days", date_offset_days or None),
            ("valuation_date", valuation_date),
            ("date_offsets", list(date_offsets) if date_offsets else None),
            ("valuation_dates", list(valuation_dates) if valuation_dates else None),
        )
        if value
    ]
    if len(given) > 1:
        raise ValueError(
            "these say when the curve is valued and only one of them can: got "
            + ", ".join(given)
            + ". Use date_offsets or valuation_dates for a family of dates."
        )

    if valuation_dates:
        offsets = [offset_to(parse_valuation_date(d), asof) for d in valuation_dates]
    elif valuation_date:
        offsets = [offset_to(parse_valuation_date(valuation_date), asof)]
    elif date_offsets:
        offsets = [int(d) for d in date_offsets]
    else:
        offsets = [int(date_offset_days)]

    for offset in offsets:
        if offset < 0:
            raise ValueError(
                f"date offsets roll the valuation date forward; {offset} would value "
                "the book in the past, where this engine has no prices."
            )
    if len(set(offsets)) != len(offsets):
        raise ValueError(
            f"the same valuation date was asked for twice: offsets {offsets}. Two "
            "identical curves are not a comparison."
        )
    return offsets


def valuation_date_of(offset: int, asof: date | None = None) -> str:
    """The calendar date an offset lands on, so a curve says when it is valued
    rather than making the reader add days to today."""
    return ((asof or date.today()) + timedelta(days=offset)).isoformat()


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
    #: Settlement date, for options only. This is the key the P&L breakdown
    #: groups on, and it is the *settlement* date rather than the last trading
    #: date for the same reason the skew buckets are: a quarterly and a weekly
    #: that settle the same morning are one expiry to a risk book, however
    #: differently TWS lists them.
    expiry: date | None = None
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

    @property
    def expiry_key(self) -> str:
        """The bucket this position falls in under the expiry breakdown.

        ``ES 2026-10-30`` for an option; ``ES (future)``, ``AAPL (equity)`` for
        everything that has no expiry to speak of. Two things follow from
        giving those a key of their own rather than dropping them: the
        breakdown sums to the point's total, so a reader can check it against
        `pnl` instead of trusting it, and a key without a date on it is
        visibly not an expiry.
        """
        if self.expiry is not None:
            return f"{self.symbol} {self.expiry.isoformat()}"
        return f"{self.symbol} ({self.asset_class})"

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
    base.expiry = C.settlement_date(contract)
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
            unit.expiry = C.settlement_date(contract)
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


async def build_surfaces(
    units: Sequence[RiskUnit], cfg: StressConfig
) -> tuple[dict[str, pricing.VolSurface], list[str]]:
    """One :class:`~ibkr_risk_mcp.pricing.VolSurface` per underlying root, built
    from the portfolio's own implied volatilities.

    A surface rather than a bag of independent smiles, because the two axes fail
    differently. Along **strike**, three quotes on one expiry are the minimum
    that has a slope; with fewer, that expiry does not pin down a smile. Along
    **tenor**, a book routinely holds one lonely strike on a back month and a
    full ladder on the front — and the back month's *shape* is far better
    approximated by the front's, carried across in total variance, than by the
    flat line a single quote implies.

    So an expiry contributes its shape to the surface only when it has three
    distinct strikes, and every option is then read off the surface at its own
    tenor, interpolated between the tenors that do. An underlying where no
    expiry defines a smile gets no surface at all and is repriced sticky_strike,
    which is said out loud rather than papered over with a flat smile.
    """
    warnings: list[str] = []
    if cfg.vol_mode != "sticky_moneyness":
        return {}, warnings

    grouped: dict[tuple[str, str], list[RiskUnit]] = {}
    for u in units:
        if u.asset_class == "option" and u.priceable and u.skew_key:
            grouped.setdefault(u.skew_key, []).append(u)

    surfaces: dict[str, pricing.VolSurface] = {}
    thin: list[tuple[str, str]] = []
    for key, group in sorted(grouped.items()):
        symbol, expiry = key
        strikes = [u.strike for u in group]
        distinct = len({round(float(s), 6) for s in strikes})
        if distinct < 3 and cfg.fetch_skew:
            fetched = await _fetch_skew(group[0], key)
            if fetched is not None:
                surfaces.setdefault(symbol, pricing.VolSurface()).add(fetched)
                continue
        if distinct < 3:
            thin.append(key)
            continue
        forward = group[0].forward(0.0, group[0].years or 0.0, cfg.rate)
        surfaces.setdefault(symbol, pricing.VolSurface()).add(
            pricing.VolSkew.from_strikes(
                group[0].years or 0.0, forward, strikes, [u.iv for u in group]
            )
        )

    for symbol, expiry in thin:
        held = len({u.strike for u in grouped[(symbol, expiry)]})
        if symbol in surfaces:
            warnings.append(
                f"{symbol} {expiry}: only {held} strike(s) held on this expiry, so its "
                f"smile shape was interpolated from the other {symbol} tenors in total "
                "variance. The volatility level is still IB's own for each contract; "
                "only the moneyness response is borrowed."
            )
        else:
            warnings.append(
                f"{symbol} {expiry}: only {held} strike(s) held, and no other {symbol} "
                "expiry defines a smile either, so this expiry was repriced "
                "sticky_strike. Pass fetch_skew=true to pull neighbouring strikes "
                "from IB."
            )
    return surfaces, warnings


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


def surface_report(surfaces: dict[str, pricing.VolSurface]) -> list[dict[str, Any]]:
    """Every quote that went into the surface, so the caller can see what the
    repricing actually read.

    Without this the sticky-moneyness result is unfalsifiable: a surface that
    came back empty reprices exactly like sticky_strike, and from the outside
    the two are indistinguishable. An empty list here means the moneyness
    response was never applied, whatever ``volMode`` says.
    """
    out: list[dict[str, Any]] = []
    for symbol, surface in sorted(surfaces.items()):
        for years in surface.tenors:
            skew = surface.skews[years]
            out.append(
                {
                    "underlying": symbol,
                    "yearsToExpiry": round(float(years), 6),
                    "forward": round(float(skew.forward), 4),
                    "points": [
                        {
                            "strike": round(float(skew.forward * math.exp(k)), 4),
                            "iv": round(float(v), 6),
                        }
                        for k, v in zip(skew.log_moneyness, skew.vols)
                    ],
                }
            )
    return out


def smile_shift(
    unit: RiskUnit,
    shock: float,
    cfg: StressConfig,
    surfaces: dict[str, pricing.VolSurface],
    years_then: float | None = None,
) -> float:
    """How far the smile alone moves this strike's volatility, in points.

    Read as a **difference** — the surface at the shocked (tenor, moneyness)
    minus the surface at today's — rather than as an absolute lookup. That
    matters for two reasons.

    It keeps IB's own volatility as the anchor. IB publishes a volatility for
    *this exact contract*, out of a model that prices American exercise; a
    fitted surface is an interpolation of several of them. Reading the level off
    the fit would discard the better number in favour of a worse one, on every
    contract, at every shock.

    And it keeps the curve exactly zero at zero shock. An absolute lookup only
    returns a unit's own volatility where the fit happens to pass through its
    strike — true for a strike that helped define its own expiry's smile, false
    for one whose shape was borrowed from another tenor. A difference is zero at
    zero shock by construction, for every contract.
    """
    surface = surfaces.get(unit.symbol)
    if surface is None or unit.strike is None:
        return 0.0
    years_now = float(unit.years or 0.0)
    if years_then is None:
        years_then = years_now
    now = surface.iv(years_now, float(unit.strike), unit.forward(0.0, years_now, cfg.rate))
    then = surface.iv(years_then, float(unit.strike), unit.forward(shock, years_then, cfg.rate))
    if now is None or then is None:
        return 0.0
    return float(then - now)


def shocked_vol(
    unit: RiskUnit,
    shock: float,
    cfg: StressConfig,
    surfaces: dict[str, pricing.VolSurface],
    years_then: float | None = None,
) -> float:
    """The volatility to reprice this contract at, under this shock.

    ``sticky_strike`` — the strike keeps its volatility. This is the default and
    it is what Risk Navigator's default curve does. It is also the conservative
    choice, since it does not let a falling market hand a position a lower
    volatility than the one it holds today.

    ``sticky_moneyness`` — the smile travels with the forward, so the strike
    picks up the volatility belonging to its new moneyness, at its own tenor,
    interpolated across strike *and* expiry. It arrives as the shift computed by
    :func:`smile_shift`, on top of IB's volatility rather than in place of it.

    Neither mode moves the *level* of the surface: one pins volatility to the
    strike, the other slides the strike along the smile the portfolio has today.
    The level's own response to the shock is :func:`vol_response`, and it is
    added on top of whichever mode is in use — the two answer different
    questions and do not overlap.

    ``shock`` here is the move of *this* position's underlying, already scaled
    by its beta, so a position attenuated on the shock axis gets an attenuated
    volatility response to match.
    """
    base = float(unit.iv or 0.0)
    if cfg.vol_mode == "sticky_moneyness":
        base += smile_shift(unit, shock, cfg, surfaces, years_then)
    if cfg.vol_coord:
        years = float(unit.years or 0.0) if years_then is None else years_then
        base *= vol_coord_factor(shock, years, cfg)
        return max(base + cfg.vol_bump, pricing.MIN_VOL)
    return max(base + cfg.vol_bump + vol_response(shock, cfg), pricing.MIN_VOL)


def vol_coord_factor(shock: float, years: float, cfg: StressConfig) -> float:
    """IB's volatility-coordinated shock, as a multiplier on each volatility.

    Two things distinguish it from the additive slopes, and both matter.

    **It is relative.** The nominal shock multiplies the volatility a contract
    already has, so a wing quoted at 41% picks up more points than a 31%
    at-the-money from the very same scenario. That is a steepening surface, and
    it falls out of the form rather than being bolted on. A parallel shift in
    points cannot produce it at any slope — which is why, on a book that is
    short the middle and long both wings, the additive model made the curve
    monotonically worse where Risk Navigator turns it back up.

    **It is damped across tenors.** ``VR(t)`` is 1 at zero and decreasing, so
    the front month takes the shock almost whole and a six-month option takes a
    fraction of it. Without that the model is unusable: undamped, a 20% fall
    triples every volatility on the board.

    The asymmetry is IB's documented one — a fall moves volatility ten times
    as hard as a rise of the same size. ``VR`` itself is not published, and
    ``vol_coord_decay`` is this server's fit to a live curve, not IB's number.
    """
    nominal = cfg.vol_coord_down * (-shock) if shock < 0 else -cfg.vol_coord_up * shock
    vr = math.exp(-cfg.vol_coord_decay * max(years, 0.0))
    return max(1.0 + nominal * vr, 0.0)


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
    surfaces: dict[str, pricing.VolSurface],
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
        vol = shocked_vol(unit, eff, cfg, surfaces, years_then)
        shocked = unit.model_price(eff, vol, years_then, cfg.rate)
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


def trough_by_expiry(
    rows: Sequence[Mapping[str, Any]], portfolio_trough_shock: float
) -> list[dict[str, Any]]:
    """Each expiry's own worst point, and what it contributes at the
    portfolio's.

    These are two different questions and the gap between them answers a third.
    ``pnl`` is where that expiry alone is worst, which says what it can cost.
    ``pnlAtPortfolioTrough`` is what it is doing at the shock that actually
    defines the account's risk floor, which is what says whether closing it
    would move that floor. A short expiry whose own minimum sits at −35% while
    the book troughs at −22% is not the one to buy back, and reading only the
    first column would nominate it.

    Sorted worst-first, so the expiry holding the trough down is the first row.
    """
    keys = sorted({k for row in rows for k in row.get("pnl_by_expiry", {})})
    out: list[dict[str, Any]] = []
    for key in keys:
        points = [(float(r["shock"]), float(r["pnl_by_expiry"].get(key, 0.0))) for r in rows]
        worst = min(points, key=lambda p: p[1])
        at_trough = next(
            (pnl for shock, pnl in points if shock == portfolio_trough_shock), None
        )
        row: dict[str, Any] = {
            "key": key,
            "shock": worst[0],
            "pnl": round(worst[1], 2),
        }
        if at_trough is not None:
            row["pnlAtPortfolioTrough"] = round(at_trough, 2)
        out.append(row)
    out.sort(key=lambda r: r["pnl"])
    return out


def run_curve(
    units: Sequence[RiskUnit],
    cfg: StressConfig,
    surfaces: dict[str, pricing.VolSurface],
) -> dict[str, Any]:
    """The P&L curve and its trough for one set of units.

    Which breakdowns come back is ``cfg.breakdown``'s to decide. Both are
    accumulated either way — the arithmetic is a dictionary add per position
    and costs nothing next to the repricing — but only the ones asked for are
    serialised, because the payload is where this actually gets expensive.
    """
    want_symbol = cfg.breakdown in ("symbol", "both")
    want_expiry = cfg.breakdown in ("expiry", "both")
    rows: list[dict[str, Any]] = []
    for shock in cfg.shocks:
        by_class: dict[str, float] = {}
        by_symbol: dict[str, float] = {}
        by_expiry: dict[str, float] = {}
        total = 0.0
        for unit in units:
            pnl = unit_pnl(unit, shock, cfg, surfaces)
            total += pnl
            by_class[unit.asset_class] = by_class.get(unit.asset_class, 0.0) + pnl
            by_symbol[unit.symbol] = by_symbol.get(unit.symbol, 0.0) + pnl
            key = unit.expiry_key
            by_expiry[key] = by_expiry.get(key, 0.0) + pnl
        row: dict[str, Any] = {
            "shock": round(float(shock), 6),
            "pnl_total": round(float(total), 2),
            "pnl_by_asset_class": {k: round(float(v), 2) for k, v in sorted(by_class.items())},
        }
        if want_symbol:
            row["pnl_by_symbol"] = {k: round(float(v), 2) for k, v in sorted(by_symbol.items())}
        if want_expiry:
            row["pnl_by_expiry"] = {k: round(float(v), 2) for k, v in sorted(by_expiry.items())}
        rows.append(row)
    points = [(r["shock"], r["pnl_total"]) for r in rows]
    best = max(points, key=lambda p: p[1])
    trough = trough_of(points)
    out: dict[str, Any] = {
        "curve": rows,
        "trough": trough,
        "peak": {"shock": float(best[0]), "pnl": round(float(best[1]), 2)},
    }
    if want_expiry:
        out["troughByExpiry"] = trough_by_expiry(rows, float(trough["shock"]))
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


def scope_warnings(
    units: Sequence[RiskUnit], cfg: StressConfig, *, include_vol: bool = True
) -> list[str]:
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
        # The pointer has to name a key the response actually carries, so it
        # follows `breakdown` rather than naming pnl_by_symbol unconditionally.
        where = {
            "symbol": "in `pnl_by_symbol`",
            "expiry": "in `pnl_by_expiry`",
            "both": "in `pnl_by_symbol` and `pnl_by_expiry`",
            "none": "not broken out at all, since `breakdown` is 'none'",
        }[cfg.breakdown]
        out.append(
            f"Positions on {', '.join(attenuated)} were repriced at a beta-scaled move of "
            "their own underlying rather than at the full shock. That is what makes this "
            "curve readable when one underlying does not belong on the axis — but it does "
            "not *measure* those positions, it stands them down. A short strangle "
            "attenuated to a fifth of the equity move still carries its whole gap risk, "
            f"and none of that risk is on this curve. Their contribution here is {where}; "
            "their own scenario is not modelled by this server."
        )

    if not include_vol:
        # stress_curve varies the slope per curve, so one sentence about "the"
        # slope would be wrong for every scenario but one. It says it itself.
        return out

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


def decay_source(cfg: StressConfig) -> str:
    """Where the decay in force came from, in one line.

    Three cases and they are not interchangeable: your own fit, this server's
    factory fit, or a number the caller typed. The stored calibration is only
    claimed when the decay actually matches it — a caller who overrides
    ``vol_coord_decay`` by hand must not have somebody else's provenance
    attached to their number.
    """
    stored = calibration.load()
    if stored and abs(cfg.vol_coord_decay - float(stored["decay"])) < 1e-9:
        return calibration.provenance() or "a stored calibration with no provenance"
    if cfg.vol_coord_decay == DEFAULT_VOL_COORD_DECAY:
        return (
            "this server's factory fit to one Risk Navigator screenshot on one account "
            "— not published by IB, and with no claim on your book. Refit it with the "
            "calibrate_vol_coord tool."
        )
    return "passed in by the caller; this server did not fit it and cannot vouch for it"


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
        "volCoord": cfg.vol_coord,
        "volCoordModel": (
            "IB's volatility-coordinated form: every volatility is multiplied by "
            f"(1 + Y*VR(t)), Y = {cfg.vol_coord_down:g}*|shock| on a fall and "
            f"-{cfg.vol_coord_up:g}*shock on a rise, VR(t) = exp(-{cfg.vol_coord_decay:g}*t). "
            "The asymmetry is IB's documented one; the decay is this server's fit to a "
            "live Vol.Coord. curve and is not published by IB. Because the shock is "
            "relative, the surface steepens on its own."
            if cfg.vol_coord
            else None
        ),
        "volCoordDecay": cfg.vol_coord_decay if cfg.vol_coord else None,
        "volCoordDecaySource": decay_source(cfg) if cfg.vol_coord else None,
        "dateOffsetDays": cfg.date_offset_days,
        "valuationDate": valuation_date_of(cfg.date_offset_days, cfg.asof),
        "pnlBreakdown": cfg.breakdown,
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
    surfaces, warnings = await build_surfaces(units, cfg)

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
    if cfg.vol_coord:
        warnings.extend(vol_coord_warnings(units, cfg))

    result = run_curve(units, cfg, surfaces)
    reconciliation = reconcile(holdings, ib, connection.require_account())
    return {
        **result,
        "reconciliation": reconciliation,
        "reconciled": reconciliation.get("reconciled", False),
        "assumptions": assumptions(cfg),
        "volSurfaceUsed": surface_report(surfaces),
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

    surfaces, warnings = await build_surfaces(combined, cfg)
    if leg_problems:
        warnings.append(
            f"{len(leg_problems)} leg(s) could not be priced and are not in the "
            "'with legs' curve — see `legProblems`. The comparison below is of the "
            "portfolio against the legs that did resolve."
        )
    warnings.extend(scope_warnings(combined, cfg))

    base = run_curve(base_units, cfg, surfaces)
    withlegs = run_curve(combined, cfg, surfaces)
    diff_points: list[dict[str, Any]] = []
    for b, w in zip(base["curve"], withlegs["curve"]):
        point: dict[str, Any] = {
            "shock": b["shock"],
            "pnl_total": round(float(w["pnl_total"] - b["pnl_total"]), 2),
        }
        # The per-expiry difference is the row that answers "did the structure
        # land where the damage is". A leg bought against one expiry shows up
        # under that expiry alone, and an expiry it did not touch reads zero,
        # which is how you tell a hedge from a change of subject.
        if "pnl_by_expiry" in b:
            keys = set(b["pnl_by_expiry"]) | set(w["pnl_by_expiry"])
            point["pnl_by_expiry"] = {
                k: round(w["pnl_by_expiry"].get(k, 0.0) - b["pnl_by_expiry"].get(k, 0.0), 2)
                for k in sorted(keys)
            }
        diff_points.append(point)
    diff_pairs = [(r["shock"], r["pnl_total"]) for r in diff_points]
    diff_trough = trough_of(diff_pairs)
    difference: dict[str, Any] = {
        "curve": diff_points,
        "trough": diff_trough,
    }
    if cfg.breakdown in ("expiry", "both"):
        difference["troughByExpiry"] = trough_by_expiry(
            diff_points, float(diff_trough["shock"])
        )
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
        "volSurfaceUsed": surface_report(surfaces),
        "reconciliation": reconciliation,
        "reconciled": reconciliation.get("reconciled", False),
        "assumptions": assumptions(cfg),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# the multi-regime curve
# --------------------------------------------------------------------------


def reference_underlying(units: Sequence[RiskUnit], cfg: StressConfig) -> dict[str, Any] | None:
    """The instrument whose price the shock axis is quoted against.

    The curve's x-axis is a percentage, which is unambiguous but hard to read
    against a chart: "the trough is at −18%" is a weaker statement than "the
    trough is at 5764". Naming one underlying and printing its shocked level at
    every point closes that gap.

    It is a *label*, not a model input — nothing is priced off it. The engine
    still shocks every in-scope underlying by the same percentage, each at its
    own price, so a book on two underlyings is not being collapsed onto one
    here. The heaviest in-scope exposure wins, which on an index book is the
    front-month future the options hang off.
    """
    best: dict[str, dict[str, float]] = {}
    for unit in units:
        if not in_scope(unit, cfg):
            continue
        name = unit.und_symbol or unit.symbol
        if unit.asset_class == "option" and unit.priceable:
            price = float(unit.und_price or 0.0)
            weight = abs(unit.position * unit.multiplier * price)
        elif unit.asset_class == "future" and unit.notional and unit.position and unit.multiplier:
            price = float(unit.notional) / (unit.position * unit.multiplier)
            weight = abs(float(unit.notional))
        elif unit.asset_class == "equity" and unit.market_value and unit.position:
            price = float(unit.market_value) / unit.position
            weight = abs(float(unit.market_value))
        else:
            continue
        if price <= 0:
            continue
        row = best.setdefault(name, {"weight": 0.0, "price": price, "heaviest": 0.0})
        row["weight"] += weight
        # The price is taken from the single heaviest position rather than
        # averaged: two ES expiries hang off two different futures, and their
        # mean is a level neither of them trades at.
        if weight >= row["heaviest"]:
            row["heaviest"] = weight
            row["price"] = price
    if not best:
        return None
    name, row = max(best.items(), key=lambda kv: kv[1]["weight"])
    return {"symbol": name, "spot": round(row["price"], 4)}


def curve_points(
    rows: Sequence[dict[str, Any]],
    spot: float | None,
    net_liquidation: float | None,
) -> list[dict[str, Any]]:
    """The engine's rows, restated in the units a reader plots in.

    ``pnl_pct_of_nlv`` is the one that travels: a 43,000 loss means nothing
    without the account beside it, and a reader comparing two dates or two
    accounts needs the fraction rather than the amount. It is omitted rather
    than defaulted when NetLiquidation is unavailable — a percentage of an
    assumed denominator is worse than no percentage.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        shock = float(row["shock"])
        pnl = float(row["pnl_total"])
        point: dict[str, Any] = {
            "shock": round(shock, 6),
            "shock_pct": round(shock * 100.0, 4),
            "pnl": round(pnl, 2),
            "pnl_by_asset_class": row["pnl_by_asset_class"],
        }
        for key in ("pnl_by_symbol", "pnl_by_expiry"):
            if key in row:
                point[key] = row[key]
        if spot:
            point["underlying"] = round(spot * (1.0 + shock), 4)
        if net_liquidation:
            point["portfolio_value"] = round(net_liquidation + pnl, 2)
            point["pnl_pct_of_nlv"] = round(pnl / net_liquidation, 6)
        out.append(point)
    return out


async def stress_curve(
    cfg: StressConfig,
    scenarios: Sequence[VolScenario],
    date_offsets: Sequence[int] | None = None,
) -> dict[str, Any]:
    """One curve per volatility regime and valuation date, over one loading of
    the portfolio.

    The regimes differ only in how the volatility *level* responds to the shock,
    which is the assumption Risk Navigator hides. ``vol_slope_down=0`` is the
    constant-volatility curve — its blue line, and the one to check first: if
    that does not line up, the surface lookup is wrong and no other regime is
    worth reading.

    Everything expensive happens once. Positions, greeks and the volatility
    surface are loaded a single time and every scenario is repriced against the
    same snapshot, so the curves are differences in assumption rather than
    differences in market data — which running the single-curve tool three times
    could not guarantee, since the book moves between calls.

    ``date_offsets`` extends that guarantee to the time axis. "Today against
    Monday, when the August wings expire" used to take two calls, and the book
    and the market moved between them, so part of the difference between the
    two curves was not the three days at all. Given several offsets the
    scenarios are crossed with them and every curve comes out of the same
    positions and the same prices, leaving time as the only thing that changed.
    """
    cfg.validate()
    if not scenarios:
        raise ValueError("stress_curve needs at least one volatility scenario")
    names = [s.name for s in scenarios]
    if len(set(names)) != len(names):
        raise ValueError(f"scenario names must be unique, got {names}")
    offsets = resolve_offsets(
        date_offset_days=cfg.date_offset_days,
        date_offsets=date_offsets,
        asof=cfg.asof,
    )
    for scenario in scenarios:
        for offset in offsets:
            replace(cfg, **scenario.overrides(), date_offset_days=offset).validate()

    ib = await connection.get()
    holdings = await MD.load_holdings(with_greeks=True)
    units = [unit_from_holding(h, cfg.asof) for h in holdings]
    surfaces, warnings = await build_surfaces(units, cfg)
    surface_rows = surface_report(surfaces)

    unpriceable = [u for u in units if u.asset_class == "option" and not u.priceable]
    if unpriceable:
        warnings.append(
            f"{len(unpriceable)} option position(s) had no usable model greeks and were "
            "held flat across every shock, in every scenario — each curve understates "
            "the risk by whatever they carry. They are listed in `positions` with a note."
        )
    implied_locally = [u for u in units if u.asset_class == "option" and u.priceable and u.note]
    if implied_locally:
        warnings.append(
            f"{len(implied_locally)} option position(s) are repriced from a volatility "
            "implied locally off the mark price, because IB published no model greeks "
            "for them — commonly a missing market data entitlement. They are in the "
            "curves, but they are this server's numbers rather than IB's. See `positions`."
        )
    other = {u.symbol for u in units if u.asset_class == "other"}
    if other:
        warnings.append(
            f"Held flat because this server does not model them: {', '.join(sorted(other))}."
        )
    if cfg.vol_mode == "sticky_moneyness" and not surface_rows:
        warnings.append(
            "volMode is 'sticky_moneyness' and `volSurfaceUsed` is EMPTY: no expiry in "
            "this portfolio held three strikes, so no smile could be built and every "
            "option was repriced sticky_strike instead. The moneyness response is not in "
            "these curves at all. Pass fetch_skew=true, or read the result as "
            "sticky_strike."
        )
    warnings.extend(scope_warnings(units, cfg, include_vol=False))
    if any(sc.vol_coord for sc in scenarios):
        warnings.extend(vol_coord_warnings(units, cfg))
    # This used to be unconditional, from when every scenario was an additive
    # slope. Left that way it described a parallel shift flat across tenors on a
    # run whose only volatility model was vol_coord — which is relative and
    # damped by tenor, so the sentence contradicted the thing it was attached
    # to. A warning that misdescribes the model in use is worse than no warning.
    if any(sc.vol_slope_down or sc.vol_slope_up for sc in scenarios):
        warnings.append(
            "One or more curves use an additive volatility slope, which is your "
            "assumption rather than a measurement, and is applied as a parallel shift of "
            "the whole surface, flat across tenors. A real surface steepens in a "
            "sell-off, which understates a long out-of-the-money put, and a long-dated "
            "position moves less than the front month a slope is usually calibrated on. "
            "vol_coord has neither limitation and is the better choice unless you "
            "specifically want a flat regime."
        )
    warnings.append(
        "Each curve is one assumption about volatility, and which is worst depends on "
        "where you are on the axis rather than being fixed. Compare them against each "
        "other rather than reading any one of them as the answer."
    )
    if any(offsets):
        warnings.append(
            "One or more curves are valued forward in time, at "
            + ", ".join(
                f"{valuation_date_of(o, cfg.asof)} (+{o}d)" for o in offsets if o
            )
            + ". Time is the only thing that moves: they are priced off TODAY's spot "
            "and TODAY's implied volatilities with the clock advanced, so what they "
            "show is decay and the change in convexity that comes with it, not a "
            "forecast of where the market will be. Anything expiring inside the window "
            "is carried to a hundredth of a day rather than removed from the book, so "
            "it settles at intrinsic value rather than disappearing."
        )

    reconciliation = reconcile(holdings, ib, connection.require_account())
    net_liq = reconciliation.get("netLiquidation")
    reference = reference_underlying(units, cfg)
    spot = float(reference["spot"]) if reference else None

    # `assumptions` describes one config, and here there are several. The
    # volatility terms are the only ones that differ, so they are replaced by a
    # pointer to the curves rather than left reporting the base config's zeros
    # — which would read as "volatility is held constant" on a result whose
    # whole purpose is that it is not.
    shared = assumptions(cfg)
    for key in ("volSlopeDown", "volSlopeUp", "volBump"):
        shared.pop(key, None)
    shared["volatilityLevel"] = (
        "varies by curve: each entry under `curves` carries its own volSlopeDown "
        "(volatility points added per 1% fall), volSlopeUp and volBump, applied as a "
        "parallel shift of the whole surface. A curve with volSlopeDown 0 is the "
        "constant-volatility case — the one to check against Risk Navigator first."
    )
    shared["volScenarios"] = [
        {
            "name": s.name,
            "volSlopeDown": s.vol_slope_down,
            "volSlopeUp": s.vol_slope_up,
            "volBump": s.vol_bump,
        }
        for s in scenarios
    ]
    # Same reasoning as the volatility terms above: with a family of dates the
    # base config's single offset describes none of the curves.
    dates = [
        {"offsetDays": o, "valuationDate": valuation_date_of(o, cfg.asof)} for o in offsets
    ]
    if len(offsets) > 1:
        shared.pop("dateOffsetDays", None)
        shared["valuationDates"] = dates

    curves: list[dict[str, Any]] = []
    for scenario in scenarios:
        for offset in offsets:
            scfg = replace(cfg, **scenario.overrides(), date_offset_days=offset)
            result = run_curve(units, scfg, surfaces)
            points = curve_points(result["curve"], spot, net_liq)
            valued_at = valuation_date_of(offset, cfg.asof)
            entry: dict[str, Any] = {
                "name": scenario.name,
                # `name` stays the scenario's alone, so a consumer keyed on it
                # does not change meaning the day a second date is asked for.
                # `label` is what distinguishes two curves in one result.
                "label": scenario.name if len(offsets) == 1 else f"{scenario.name} @ {valued_at}",
                "dateOffsetDays": offset,
                "valuationDate": valued_at,
                "volSlopeDown": scenario.vol_slope_down,
                "volSlopeUp": scenario.vol_slope_up,
                "volBump": scenario.vol_bump,
                "volCoord": scenario.vol_coord,
                "points": points,
                "trough": result["trough"],
                "peak": result["peak"],
            }
            if "troughByExpiry" in result:
                entry["troughByExpiry"] = result["troughByExpiry"]
            if "troughRefined" in result:
                entry["troughRefined"] = result["troughRefined"]
            worst = min(points, key=lambda p: p["pnl"])
            entry["minPnl"] = worst["pnl"]
            entry["minAtShockPct"] = worst["shock_pct"]
            if "pnl_pct_of_nlv" in worst:
                entry["minPnlPctOfNlv"] = worst["pnl_pct_of_nlv"]
            if "underlying" in worst:
                entry["minAtUnderlying"] = worst["underlying"]
            curves.append(entry)

    return {
        "underlying": reference,
        "netLiquidation": net_liq,
        "valuationDates": dates,
        "curves": curves,
        "volSurfaceUsed": surface_rows,
        "reconciliation": reconciliation,
        "reconciled": reconciliation.get("reconciled", False),
        "assumptions": shared,
        "excluded": excluded_report(units, cfg),
        "positions": _position_report(units, cfg),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# calibrating vol_coord
# --------------------------------------------------------------------------


def vol_coord_warnings(units: Sequence[RiskUnit], cfg: StressConfig) -> list[str]:
    """What the reader has to know before believing a ``vol_coord`` curve.

    Two separate admissions, and neither should have to be inferred from
    `assumptions`. The decay is a fit to somebody else's Risk Navigator; and
    the fit had nothing to say about tenors longer than the book it was fitted
    on, where an exponential runs to zero and quietly reports a long-dated
    option as having no volatility risk at all.
    """
    out: list[str] = []
    stored = calibration.load()
    if stored and abs(cfg.vol_coord_decay - float(stored["decay"])) < 1e-9:
        # A calibrated run still says where its number came from. The warning
        # changes from "distrust this" to "here is what it was fitted against",
        # because a fit is only as current as the book it was fitted on.
        out.append(
            f"vol_coord is running on YOUR calibration, decay {cfg.vol_coord_decay:g}, "
            + (calibration.provenance() or "provenance unrecorded")
            + ". Refit with the calibrate_vol_coord tool when the book's tenor mix "
            "changes materially, or when the const curve stops lining up with Risk "
            "Navigator's blue line."
        )
    elif cfg.vol_coord_decay == DEFAULT_VOL_COORD_DECAY:
        out.append(
            f"vol_coord is running on the factory decay of {DEFAULT_VOL_COORD_DECAY:g}. IB "
            "documents that VR(t) exists and is decreasing but not what it is, so this "
            "number is a FIT — to one Risk Navigator screenshot, on one account, from "
            "nine points read off a chart by eye. The 10x/1x asymmetry above it is IB's "
            "own and is on firmer ground than the damping is. Recalibrate against your "
            "own Risk Navigator with the calibrate_vol_coord tool — read four or more "
            "points off its Vol.Coord. curve and pass them in — before treating a "
            "vol_coord curve as anything but indicative."
        )

    limit = cfg.vol_coord_calibrated_to_years
    beyond = sorted(
        {
            u.label
            for u in units
            if u.asset_class == "option"
            and u.priceable
            and in_scope(u, cfg)
            and float(u.years or 0.0) > limit
        }
    )
    if beyond:
        worst = max(
            float(u.years or 0.0)
            for u in units
            if u.asset_class == "option" and u.priceable and in_scope(u, cfg)
        )
        vr = math.exp(-cfg.vol_coord_decay * worst)
        out.append(
            f"{len(beyond)} option position(s) expire beyond {limit:.2f} years, which is "
            f"as far as the vol_coord damping was ever calibrated — the longest is "
            f"{worst:.2f} years, where VR(t) is {vr:.3f}. Past the calibration the "
            "exponential is extrapolation and it decays to nothing: at that tenor this "
            f"model moves volatility by {vr * 100:.1f}% of the nominal shock, so those "
            "positions are being priced as though a crash barely touched their "
            "volatility. That is almost certainly wrong, and it understates the risk of "
            f"anything long-dated. Affected: {', '.join(beyond[:8])}"
            + (f" and {len(beyond) - 8} more" if len(beyond) > 8 else "")
            + "."
        )
    return out


def calibrate_vol_coord(
    units: Sequence[RiskUnit],
    cfg: StressConfig,
    targets: Mapping[float, float],
    surfaces: dict[str, pricing.VolSurface] | None = None,
) -> dict[str, Any]:
    """Fit ``vol_coord_decay`` to a Risk Navigator ``Vol.Coord.`` curve.

    ``targets`` maps a shock, as a fraction, to the portfolio P&L Risk
    Navigator shows there. Read them off its Vol.Coord. line — the curve, not
    the blue one — and give at least four, spread across the range you care
    about rather than bunched near the money.

    This exists so the decay stops being a constant somebody once fitted and
    becomes a number you can rederive. What comes back is not only the fit but
    what to distrust about it: the residual at every point, the tenor range the
    targets actually constrain, and the most extreme volatility the fitted model
    produces. A decay that reproduces the curve by pricing a wing at 150% has
    fitted the chart rather than the market, and that is visible here rather
    than three layers down in a P&L.
    """
    from scipy.optimize import least_squares

    if len(targets) < 3:
        raise ValueError(
            "a decay cannot be fitted to fewer than three points; four or more spread "
            "across the range is what makes the answer mean anything"
        )
    surfaces = surfaces or {}
    shocks = sorted(targets)
    observed = [float(targets[s]) for s in shocks]

    def curve_at(decay: float) -> list[float]:
        fitted = replace(cfg, vol_coord=True, vol_coord_decay=float(decay))
        rows = run_curve(units, replace(fitted, shocks=shocks), surfaces)["curve"]
        return [float(r["pnl_total"]) for r in rows]

    result = least_squares(
        lambda p: [m - o for m, o in zip(curve_at(p[0]), observed)],
        x0=[DEFAULT_VOL_COORD_DECAY],
        bounds=([0.0], [200.0]),
        # run_curve rounds P&L to the cent, and the default finite-difference
        # step is relative and around 1e-8 — small enough that the perturbed
        # curve rounds to exactly the same numbers, the gradient reads as zero
        # and the fit returns x0 looking like it converged. The step has to be
        # coarse enough to move a cent.
        diff_step=1e-3,
    )
    decay = float(result.x[0])
    modelled = curve_at(decay)
    residuals = [m - o for m, o in zip(modelled, observed)]
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))

    priced = [
        u
        for u in units
        if u.asset_class == "option" and u.priceable and in_scope(u, cfg)
    ]
    tenors = [float(u.years or 0.0) for u in priced]
    deepest = min(shocks)
    fitted_cfg = replace(cfg, vol_coord=True, vol_coord_decay=decay)
    repriced = [
        (shocked_vol(u, deepest, fitted_cfg, surfaces), u.label, float(u.iv or 0.0))
        for u in priced
    ]
    repriced.sort(reverse=True)

    warnings: list[str] = []
    if repriced and repriced[0][0] > 1.5:
        warnings.append(
            f"At {deepest:+.0%} this decay prices {repriced[0][1]} at "
            f"{repriced[0][0]:.0%} implied volatility, up from {repriced[0][2]:.0%}. A fit "
            "that reproduces the curve by putting a wing above 150% has fitted the chart "
            "rather than the market. Treat the decay as unusable and check whether the "
            "targets were read off the right line."
        )
    if tenors and max(tenors) < 1.0:
        warnings.append(
            f"These targets only constrain tenors out to {max(tenors):.2f} years, because "
            "that is all the portfolio holds. VR(t) beyond it is extrapolation, and an "
            "exponential extrapolates to zero — meaning no volatility response at all on "
            "a long-dated position. Set vol_coord_calibrated_to_years to "
            f"{max(tenors):.3f} so the engine says so when it is asked to price past it."
        )

    return {
        "decay": round(decay, 4),
        "rms": round(rms, 2),
        "points": [
            {
                "shock": round(s, 6),
                "target": round(o, 2),
                "model": round(m, 2),
                "residual": round(r, 2),
            }
            for s, o, m, r in zip(shocks, observed, modelled, residuals)
        ],
        "calibratedToYears": round(max(tenors), 4) if tenors else None,
        "tenorsCovered": sorted({round(t, 3) for t in tenors}),
        "mostExtremeVol": (
            {
                "label": repriced[0][1],
                "atShock": round(deepest, 6),
                "impliedVolBefore": round(repriced[0][2], 4),
                "impliedVolAfter": round(repriced[0][0], 4),
            }
            if repriced
            else None
        ),
        "warnings": warnings,
    }


async def run_calibration(
    cfg: StressConfig,
    targets: Mapping[float, float],
    persist: bool = True,
) -> dict[str, Any]:
    """Fit the decay against the live account and, unless told not to, keep it.

    This is :func:`calibrate_vol_coord` with the two things around it that make
    it usable from a conversation rather than from a shell: it loads the book
    itself, and it writes the result where every later run will find it.

    It refuses to store a fit taken against a portfolio that does not
    reconcile. A curve that is missing a position is missing it at every shock,
    so the decay that reproduces Risk Navigator from it is absorbing the gap —
    and unlike a bad curve, a bad stored decay is silent: it would go on
    quietly deforming every vol_coord run afterwards with nothing to say it
    came from a broken snapshot. The fit is still returned, with the reason.
    """
    cfg.validate()
    ib = await connection.get()
    account = connection.require_account()
    holdings = await MD.load_holdings(with_greeks=True)
    units = [unit_from_holding(h, cfg.asof) for h in holdings]
    surfaces, warnings = await build_surfaces(units, cfg)

    fit = calibrate_vol_coord(units, cfg, targets, surfaces)
    reconciliation = reconcile(holdings, ib, account)
    reconciled = bool(reconciliation.get("reconciled", False))

    stored: str | None = None
    if not persist:
        warnings.append(
            "Not stored, because persist=false. Pass vol_coord_decay="
            f"{fit['decay']} to stress_curve by hand, or run this again with persist "
            "left at its default to make it the standing calibration."
        )
    elif not reconciled:
        warnings.append(
            "NOT STORED: the portfolio does not reconcile against NetLiquidation, so "
            "this decay was fitted to a book that is missing something. A bad curve "
            "announces itself; a bad stored decay would not, and it would deform every "
            "vol_coord run from here on. Fix the reconciliation and refit."
        )
    else:
        record = {
            "decay": fit["decay"],
            "calibratedToYears": fit["calibratedToYears"],
            "rms": fit["rms"],
            "fittedAt": datetime.now().isoformat(timespec="seconds"),
            "account": account,
            "scope": cfg.scope,
            "volMode": cfg.vol_mode,
            "points": fit["points"],
        }
        stored = str(calibration.save(record))

    return {
        **fit,
        "stored": stored is not None,
        "storedAt": stored,
        "reconciliation": reconciliation,
        "reconciled": reconciled,
        "assumptions": assumptions(replace(cfg, vol_coord=True, vol_coord_decay=fit["decay"])),
        "warnings": [*fit["warnings"], *warnings],
    }
