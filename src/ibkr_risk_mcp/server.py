"""MCP tool definitions.

The docstrings here are the interface. The consumer of this server is a
language model that will never read the implementation, so every trap that
would change a conclusion — AM settlement dates, which quarterly an option is
written on, term structure, reconciliation — is stated in the description of
the tool that hits it, not only in the code.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Literal, Sequence

from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import AliasChoices, AnyHttpUrl, BaseModel, ConfigDict, Field

from . import marketdata as MD
from . import stress as S
from . import contracts as C
from . import margin as M
from . import calibration as CAL
from .auth import verifier_for
from .config import settings
from .connection import AccountAmbiguous, IBUnavailable, connection

logging.basicConfig(level=logging.WARNING)

log = logging.getLogger(__name__)


def _auth_settings() -> AuthSettings | None:
    """The SDK's auth configuration, or None when auth is off.

    Passing this alongside a token verifier is all it takes to become a
    conformant resource server: the SDK mounts
    /.well-known/oauth-protected-resource (RFC 9728) and answers an
    unauthenticated request with the `WWW-Authenticate: Bearer
    resource_metadata=...` challenge that points a client at it. Token
    issuance, authorization-server metadata and client registration stay with
    the provider, where they belong — this server never sees a credential.
    """
    if not settings.mcp_auth:
        return None
    return AuthSettings(
        issuer_url=AnyHttpUrl(settings.mcp_auth_issuer),  # type: ignore[arg-type]
        resource_server_url=AnyHttpUrl(settings.mcp_resource_url),  # type: ignore[arg-type]
        required_scopes=[settings.mcp_auth_scope],
    )


mcp = FastMCP(
    "ibkr-risk-mcp",
    host=settings.mcp_host,
    port=settings.mcp_port,
    streamable_http_path=settings.mcp_path,
    token_verifier=verifier_for(settings),
    auth=_auth_settings(),
    # Session affinity buys this server nothing. Everything it holds between
    # calls — the IB connection, the market data semaphore, the contract
    # details cache, the stored calibration — is process-global by design and
    # is deliberately shared by every caller, because TWS keys API sessions by
    # client id and will not give a second connection to the same one. So a
    # request that lands without a session id has lost nothing it would have
    # had with one, and dropping the requirement removes a whole class of
    # "session expired" failure from a client's path to a P&L curve.
    stateless_http=True,
    instructions="""This server exposes Interactive Brokers' *risk* data: model greeks,
IB's implied volatility surface, what-if margin, and a local stress engine that
rebuilds the portfolio P&L curve across underlying shocks.

It deliberately does not duplicate the official IBKR connector. Positions,
balances, orders, trades, performance, spot and historical prices, option
chains and watchlists all come from there. Use this server only for questions
about risk: where the P&L trough sits, what a hypothetical structure does to
it, what margin it needs, and what IB's volatility surface looks like.

It reads a live account through TWS or IB Gateway, which may be on this machine
or on a host this server reaches over the network. If any tool fails, call
check_connection first: it distinguishes "not running" from "API switch off"
from "client id taken" from "no account logged in", names the machine the
problem is on, and returns a `hint` that says what to do about it. Pass that
hint on rather than guessing — the user may not be sitting at the machine
running TWS, and some of these are not theirs to fix.

Three things about the data are worth knowing before quoting any number:

1. Expiry dates are reported twice, as lastTradeDate and settlementDate, and
   for AM-settled contracts they differ by a day. Two contracts settling the
   same morning can show two different last trading dates. Use settlementDate
   when reasoning about how positions pair up or how much time is left.
2. Every derived figure carries a reconciliation against NetLiquidation. If
   `reconciled` is false, say so and quote the residual — do not present the
   curve as fact.
3. The stress engine's assumptions are returned with every result, in
   `assumptions`. sticky_strike is the default and corresponds to Risk
   Navigator's default curve; Risk Navigator's actual volatility shock model is
   not public, so this is an approximation of it, not a reproduction.

The server is read-only against the account. whatif_order is the only tool that
reaches IB's order path, it sends whatIf=True orders that are never routed, and
it is disabled unless the server was started with IBKR_ENABLE_WHATIF=true.
There is no tool here that can submit a live order. calibrate_vol_coord writes
one small local file — the volatility calibration it fits — and nothing else
here writes anywhere. check_connection reports whether that calibration is in
place; without it every stress curve falls back to a decay fitted against
somebody else's book, which is worth saying out loud when quoting one.""",
)


def tool(*, read_only: bool = True, idempotent: bool = True) -> Callable:
    """Register a tool whose ``success`` flag reflects what actually happened.

    A failure wrapped as a successful call with the error buried in the payload
    is a reply the caller will trust. Connection problems come back with the
    same ``state`` and ``hint`` that ``check_connection`` reports, so a tool
    that fails because TWS is down says so in the same words.
    """

    annotations = ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=False,
        idempotentHint=idempotent,
        openWorldHint=True,
    )

    def register(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                return {"success": True, **(await fn(*args, **kwargs))}
            except M.WhatIfDisabled as exc:
                return {
                    "success": False,
                    "blocked": True,
                    "sentToIb": False,
                    "error": str(exc),
                }
            except AccountAmbiguous as exc:
                return {
                    "success": False,
                    "state": "account_ambiguous",
                    "accounts": exc.accounts,
                    "hint": "Set IBKR_ACCOUNT to one of these account codes and restart "
                    "the server. Two accounts must not be added together.",
                    "error": str(exc),
                }
            except IBUnavailable as exc:
                return {
                    "success": False,
                    "state": exc.state,
                    "hint": exc.hint,
                    "error": str(exc),
                }
            except (ValueError, RuntimeError) as exc:
                return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

        return mcp.tool(annotations=annotations)(wrapper)

    return register


@tool()
async def check_connection() -> dict[str, Any]:
    """Check whether TWS or IB Gateway is reachable and an account is logged in.

    Call this first whenever another tool fails. It separates the four
    situations that all present as "cannot connect" and need different fixes:

    - `not_listening` — nothing is on the port. TWS is not running, or
      IBKR_PORT points at the wrong one; the response lists which of the four
      default ports (7496 TWS live, 7497 TWS paper, 4001/4002 Gateway) are
      answering.
    - `api_not_enabled` — the port answers but the API handshake never
      completes. "Enable ActiveX and Socket Clients" is off in TWS.
    - `client_id_in_use` — another script holds this client id.
    - `not_logged_in` — the API is up but no account is loaded.
    - `connected` — everything is in place.

    `hint` says what to do about the state in each case.

    `calibration` reports whether a fitted vol_coord decay is in use and what
    it was fitted against, or null if the server is running on the factory
    number. Worth checking after any redeploy: the fit lives in a file, losing
    it is silent by design, and every stress_curve afterwards is quietly less
    accurate than the one before.
    """
    probe = await connection.probe()
    accounts = probe.get("accounts") or []
    resolved = connection.account if probe.get("connected") else None
    if probe.get("connected") and len(accounts) > 1 and resolved is None:
        probe = {
            **probe,
            "state": "account_ambiguous",
            "hint": f"Connected, but this login manages {len(accounts)} accounts "
            f"({', '.join(accounts)}) and IBKR_ACCOUNT is not set. Every tool that reads "
            "positions or account values refuses until one is chosen, because IB returns "
            "all of them at once and adding two accounts together is meaningless.",
        }
    return {
        **probe,
        "accountInUse": resolved,
        "config": {
            "host": settings.host,
            "port": settings.port,
            "clientId": settings.client_id,
            "account": settings.account,
            "marketDataType": settings.market_data_type,
            "whatIfEnabled": settings.enable_whatif,
        },
        # Whether a fitted vol_coord decay is in place, and what it was fitted
        # against. None means the factory number is in use — which is a real
        # answer and an easily missed one: `calibration.load()` falls back
        # rather than raising, by design, so a calibration file that was lost
        # in a redeploy costs every later stress run its accuracy and says
        # nothing at all. This is where that becomes visible.
        "calibration": CAL.provenance(),
    }


@tool()
async def get_margin_summary() -> dict[str, Any]:
    """Margin and liquidity, split by segment.

    Returns NetLiquidation, EquityWithLoanValue, FullInitMarginReq,
    FullMaintMarginReq, AvailableFunds, ExcessLiquidity, TotalCashValue,
    BuyingPower and Leverage — each as the account total and, where IB reports
    it, separately for the securities (`-S`) and commodities (`-C`) segments.

    **The segments are the point.** Futures margin must be met in the
    commodities segment; IB covers a shortfall there by sweeping cash out of
    the securities segment. An account whose total excess liquidity looks
    healthy can still be one bad day away from a forced liquidation if the
    shortfall lands in commodities while securities is also falling. Report the
    segment figures, not only the totals, whenever futures are involved.
    """
    return await M.margin_summary()


@tool()
async def get_position_greeks(
    symbol: str | None = Field(
        default=None, description="Restrict to options on this underlying root, e.g. 'ES'."
    ),
) -> dict[str, Any]:
    """IB's model greeks for every option position — its numbers, not ones
    implied locally.

    Returns per position: conid, symbol, secType, right, strike, lastTradeDate,
    settlementDate, daysToExpiry, position, multiplier, undPrice, impliedVol,
    delta, gamma, vega, theta, optPrice and pvDividend.

    Things to know about the values:

    - **undPrice is the forward IB used**, not the index spot. For a futures
      option that is the future's price, which differs from the cash index by
      the basis; any repricing has to start from it.
    - **multiplier comes from the contract.** ES is 50 and MES is 5, and the
      difference is a factor of ten in every exposure figure.
    - **settlementDate, not lastTradeDate, is the expiry.** The quarterly ES
      options settle AM and stop trading the afternoon before, so TWS shows 17
      December for something that expires on the 18th, while a weekly settling
      the same morning shows the 18th. Both are returned; pair positions on
      settlementDate.
    - **Missing greeks are listed, not dropped.** A contract IB never published
      model greeks for appears under `missing` with the reason. The usual
      causes are no market data subscription for that instrument, or a strike
      too illiquid for IB to imply a volatility. Note that *delayed* data does
      carry model greeks, so a missing row is rarely explained by the market
      data type alone.
    """
    holdings = await MD.load_holdings(with_greeks=True, symbol=symbol)
    options = [h for h in holdings if h.is_option]
    rows, missing = [], []
    for h in options:
        info = h.describe()
        if h.greeks is None:
            missing.append({**info, "reason": h.greeks_error})
            continue
        rows.append({**info, **h.greeks})
    return {
        "count": len(rows),
        "data": rows,
        "missing": missing,
        "nonOptionPositions": len(holdings) - len(options),
        "marketDataType": settings.market_data_type,
    }


@tool()
async def get_vol_surface(
    underlying: str = Field(description="Underlying root, e.g. 'ES' or 'SPY'."),
    expiries: list[str] = Field(
        description="Expiries as YYYYMMDD or YYYY-MM-DD. Either the last trading date or "
        "the settlement date works; both resolve to the same contracts."
    ),
    strikes: list[float] | None = Field(
        default=None,
        description="Explicit strikes. Each is snapped to the nearest listed one.",
    ),
    min_strike: float | None = None,
    max_strike: float | None = None,
    rights: list[Literal["P", "C"]] = Field(
        default=["P"], description="Puts by default; pass both for the full smile."
    ),
    sec_type: Literal["STK", "IND", "FUT"] | None = Field(
        default=None,
        description="Which instrument the symbol means. Left empty the search tries STK, "
        "then IND, then FUT — so 'ES' returns Eversource Energy, not the E-mini S&P. Pass "
        "FUT for a futures root. Any collision is reported in `warnings` either way.",
    ),
    max_strikes_per_expiry: int = Field(
        default=25,
        description="Cap per expiry. Each strike is one market data line and IB allows "
        "about fifty at once, so a wide grid over many expiries takes time.",
    ),
    trading_class: str | None = Field(
        default=None,
        description="Needed when one underlying has two contracts expiring the same day, "
        "e.g. 'ES' (quarterly, AM-settled) against 'EW4' (weekly, PM-settled).",
    ),
) -> dict[str, Any]:
    """IB's implied volatility surface for an underlying: a grid of
    (expiry, strike) with impliedVol, delta, optPrice, undPrice and
    daysToExpiry.

    This is the input that makes local repricing deterministic. With IB's own
    volatilities in hand, a constant-volatility scenario needs no proprietary
    model — only Black-76 arithmetic on top of numbers IB published.

    **Do not collapse the surface to one number.** Volatility has a term
    structure: ES at 139 days can sit near 15% at the money while the front
    month prints 12%. Using the front month for a longer tenor understates a
    long-dated position badly. Read the tenor you need, and interpolate between
    tenors in total variance if you must.

    `daysToExpiry` counts to the **settlement** date. For AM-settled expiries
    that is one day past the last trading date TWS shows.

    Strikes with no published volatility come back under `missing` with the
    reason rather than being silently absent — a surface missing its left wing
    looks identical to one that has none.

    **Check which instrument you got.** A bare root is ambiguous: `ES` is the
    E-mini S&P 500 future *and* Eversource Energy on NYSE, and with no
    `sec_type` the stock wins. The resolved contract comes back under
    `underlying`, any collision is listed in `warnings`, and a surface on the
    wrong instrument looks perfectly reasonable until you notice the strikes
    are two orders of magnitude off.
    """
    req = MD.SurfaceRequest(
        underlying=underlying,
        expiries=list(expiries),
        strikes=list(strikes) if strikes else None,
        min_strike=min_strike,
        max_strike=max_strike,
        rights=tuple(rights) or ("P",),
        sec_type=sec_type,
        max_strikes_per_expiry=max_strikes_per_expiry,
        trading_class=trading_class,
    )
    result = await MD.vol_surface(req)
    return {
        "underlying": result.underlying,
        "count": len(result.rows),
        "data": result.rows,
        "missing": result.missing,
        "warnings": result.warnings,
    }


def _stress_config(
    shocks: Sequence[float],
    vol_mode: str,
    vol_bump: float,
    vol_slope_down: float,
    vol_slope_up: float,
    date_offset_days: int,
    betas: dict[str, float] | None,
    default_beta: float,
    bond_rate_shift_bp: float,
    bond_duration_years: float,
    fetch_skew: bool,
    scope: str = "equity",
    risk_groups: dict[str, str] | None = None,
    breakdown: str = "symbol",
) -> S.StressConfig:
    return S.StressConfig(
        shocks=list(shocks),
        vol_mode=vol_mode,
        vol_bump=vol_bump,
        vol_slope_down=vol_slope_down,
        vol_slope_up=vol_slope_up,
        date_offset_days=date_offset_days,
        betas=dict(betas or {}),
        default_beta=default_beta,
        scope=scope,
        risk_groups=dict(risk_groups or {}),
        bond_rate_shift_bp=bond_rate_shift_bp,
        bond_duration_years=bond_duration_years,
        fetch_skew=fetch_skew,
        breakdown=breakdown,
    )


#: Said the same way in three tool signatures, so said once here.
BREAKDOWN_HELP = (
    "Which per-point P&L breakdowns to return. 'symbol' — the default and what this "
    "server has always returned — groups by root, which on an options book collapses "
    "every ES expiry under one 'ES' key. 'expiry' groups by the option's SETTLEMENT "
    "date instead ('ES 2026-10-30'), which is the breakdown that answers 'which expiry "
    "is holding the trough down, and which short do I buy back'. 'both' returns the two "
    "of them and 'none' neither. Positions with no expiry get a key naming their class "
    "('ES (future)', 'AAPL (equity)') so the breakdown still sums to the point's total "
    "and can be checked against it. Responses are large: a book on nine expiries pays "
    "for the second dictionary at every one of twenty-six shocks, so ask for 'both' "
    "only when you want both, and 'none' when you only want the curve."
)

VALUATION_DATE_HELP = (
    "Value the curve at this ISO date, e.g. '2026-09-30', instead of counting the days "
    "out by hand into date_offset_days. Mutually exclusive with it. No calendar magic: "
    "the date is the date, weekend or not, and time to expiry is ACT/365 throughout. "
    "The P&L is still measured FROM today at today's prices and volatilities, with the "
    "clock rolled forward — it is decay, not a forecast."
)


@tool()
async def stress_portfolio(
    shocks: list[float] = Field(
        description="Underlying moves as fractions: -0.10 is a 10% fall. A range like "
        "-0.30 to +0.30 in 0.01 steps is the usual ask."
    ),
    vol_mode: Literal["sticky_strike", "sticky_moneyness"] = Field(
        default="sticky_strike",
        description="sticky_strike keeps each strike's current volatility and is what "
        "Risk Navigator's default curve does. sticky_moneyness slides the smile with the "
        "forward.",
    ),
    vol_bump: float = Field(
        default=0.0,
        description="Added to every volatility, in points, flat along the shock axis: 0.05 "
        "is +5 points at every shock. For volatility that responds to the shock itself, "
        "use vol_slope_down.",
    ),
    vol_slope_down: float = Field(
        default=0.0,
        description="Volatility points added per 1% FALL in the underlying: 1.0 means a "
        "-20% shock reprices at +20 points. Zero — the default — holds the volatility "
        "level flat, which prices the move in the underlying and not the move in "
        "volatility that comes with it. A net short option book loses real money on that "
        "term, so leaving this at zero is the optimistic half of the answer.",
    ),
    vol_slope_up: float = Field(
        default=0.0,
        description="Volatility points removed per 1% RISE. Positive means volatility "
        "falls as the market rallies, which is the usual direction. Separate from "
        "vol_slope_down because the response is not symmetric.",
    ),
    date_offset_days: int = Field(
        default=0, description="Roll the valuation date forward this many days (time decay)."
    ),
    valuation_date: str | None = Field(default=None, description=VALUATION_DATE_HELP),
    breakdown: Literal["symbol", "expiry", "both", "none"] = Field(
        default="symbol", description=BREAKDOWN_HELP
    ),
    scope: Literal["equity", "all"] = Field(
        default="equity",
        description="Which underlyings are on the shock axis. 'equity' — the default — "
        "keeps only equity underlyings and excludes FX, rates and the rest outright, which "
        "is what TWS Risk Navigator's Equity tab does and what makes the curve comparable "
        "to it. Off the equity axis a single percentage shock is meaningless: a currency "
        "future moved 20% prices an exchange rate that has never traded there. 'all' shocks "
        "everything alike. Excluded positions are always listed under `excluded`, never "
        "dropped in silence.",
    ),
    risk_groups: dict[str, str] | None = Field(
        default=None,
        description="Override the risk group of a symbol, e.g. {'TLT': 'rates', 'GLD': "
        "'metals'}. IB publishes no asset class for a bond or gold ETF quoted as a stock, "
        "so those are classified as equity unless named here. Groups: equity, fx, rates, "
        "metals, energy, other.",
    ),
    betas: dict[str, float] | None = Field(
        default=None,
        description="Per-symbol share of the shock, e.g. {'AAPL': 1.2, 'EUR': 0.0}. Applies "
        "to options and futures as well as equities: the beta scales the move of that "
        "position's own underlying, and the option is then repriced there. Key it on the "
        "root ('ES'), the local symbol ('ESZ6 P5800') or the underlying ('ESZ6'); the most "
        "specific match wins. Use it to stand a foreign underlying down off an equity axis "
        "— but read the warning it produces: a beta of 0 removes a position from this "
        "curve, it does not measure its own risk.",
    ),
    default_beta: float = 1.0,
    bond_rate_shift_bp: float = Field(
        default=0.0,
        description="Parallel rate shift in basis points applied to bonds. Zero leaves "
        "them unchanged, which is the default.",
    ),
    bond_duration_years: float = Field(
        default=5.0,
        description="Duration assumed for bonds when a rate shift is applied. IB does not "
        "publish duration, so this is your input and the result is only as good as it.",
    ),
    fetch_skew: bool = Field(
        default=False,
        description="Let sticky_moneyness pull neighbouring strikes from IB when the "
        "portfolio holds too few to define a smile. Costs extra market data requests.",
    ),
) -> dict[str, Any]:
    """Reprice the whole portfolio — options, equities, futures — across
    underlying shocks and return the P&L curve and its **trough**.

    The trough is the primary output: the worst point of the curve and the
    shock at which it sits. `troughRefined` interpolates between grid points
    for where the minimum actually falls, and is labelled as interpolated.

    The model, returned with every result in `assumptions`:

    - **only equity underlyings are on the axis by default** (`scope='equity'`).
      FX, rates and anything else is excluded outright and listed under
      `excluded` with its market value. This is what Risk Navigator's Equity tab
      does, and it is what makes the two comparable — verified against a live
      account, where the engine and Risk Navigator agreed to 7 dollars on 29,000
      at a 15% fall once the FX leg was off both. Off the equity axis the single
      shock is meaningless: the same account's CAD strangle was contributing
      -21,716 at -20% *and* -7,183 at +10%, dominating both tails.
    - all underlyings in scope are shocked by the same percentage at once, which
      is Risk Navigator's own default assumption. `betas` scales that shock per symbol
      and reaches **every** class that responds to one — an option is repriced
      at its own beta-scaled move, not at the index move. Use it to stand a
      foreign underlying down off an equity axis, and read the warning it
      produces: a beta of 0 takes a position off this curve, it does not
      measure that position's own risk.
    - options are repriced with Black-76 on the shocked forward using **IB's**
      implied volatility. Equity options are carried from spot using IB's
      pvDividend, so both kinds go through one pricer.
    - **the volatility level is flat along the shock axis unless you say
      otherwise.** Neither vol mode raises it: sticky_strike pins volatility to
      the strike, sticky_moneyness slides a strike along today's smile. Real
      volatility rises when an index falls, and a net short option book pays for
      that on top of the delta and gamma this curve already counts. `vol_bump`
      does not fill the gap — it is constant across shocks. `vol_slope_down`
      does: 1.0 adds one volatility point per 1% fall. It is your input, not a
      measurement, and it is applied as a parallel shift across every tenor.
    - P&L is model-price-now against model-price-shocked, so the curve is
      exactly zero at zero shock by construction. The gap between the local
      model and IB's own price is reported per position as `modelVsMarket`
      instead of being folded into the curve. Expect it to be small out of the
      money and a couple of percent in the money, where IB prices the early
      exercise that Black-76 has no room for — so the curve understates losses
      slightly once options go deep in the money.
    - bonds are held flat unless `bond_rate_shift_bp` is set; anything this
      server does not model is held flat and named in `warnings`.
    - `breakdown='expiry'` adds `pnl_by_expiry` to every point and
      `troughByExpiry` beside the trough — each expiry's own worst point and
      what it contributes at the *portfolio's* trough. On a book holding one
      root across many expiries those are the two different numbers behind
      "which short is the problem", and `pnl_by_symbol` can answer neither.

    **Check `reconciled` before quoting anything.** At zero shock the portfolio
    is rebuilt from its positions and compared against NetLiquidation; a
    residual over 1% returns `reconciled: false` with the residual attached. A
    curve that does not reconcile is missing something, and the number it gives
    for the trough is missing it too.

    Risk Navigator's own volatility shock model is not published. `sticky_strike`
    is the approximation that corresponds to its default curve, not a
    reproduction of it — expect the shape to match and the last few percent
    not to.
    """
    cfg = _stress_config(
        shocks,
        vol_mode,
        vol_bump,
        vol_slope_down,
        vol_slope_up,
        S.resolve_offsets(
            date_offset_days=date_offset_days, valuation_date=valuation_date
        )[0],
        betas,
        default_beta,
        bond_rate_shift_bp,
        bond_duration_years,
        fetch_skew,
        scope,
        risk_groups,
        breakdown,
    )
    return await S.stress_portfolio(cfg)


class VolScenarioSpec(BaseModel):
    """One volatility regime for :func:`stress_curve`."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="Label for this curve, e.g. 'const' or 'stress'.")
    vol_slope_down: float = Field(
        default=0.0,
        validation_alias=AliasChoices("vol_slope_down", "beta"),
        description="Volatility points added per 1% FALL in the underlying. 0 is the "
        "constant-volatility curve — Risk Navigator's blue line. 1.0 puts a -20% shock "
        "at +20 points. Accepted as `beta` too.",
    )
    vol_slope_up: float = Field(
        default=0.0,
        description="Volatility points removed per 1% RISE. Positive means volatility "
        "falls into a rally, which is the usual direction. Separate because the response "
        "is not symmetric.",
    )
    vol_bump: float = Field(
        default=0.0,
        description="Flat shift of the whole surface for this regime, in points, at every "
        "shock including zero. Unlike the slopes this moves P&L at zero shock, so a "
        "scenario using it does not start its curve at zero.",
    )
    vol_coord: bool = Field(
        default=False,
        description="Use IB's own volatility-coordinated model for this curve — the one "
        "Risk Navigator labels 'Vol.Coord.' — instead of the additive slopes, which it "
        "then ignores. Volatility is multiplied rather than shifted: a fall of X moves it "
        "by 10X relatively and a rise by -X, damped across tenors. Because it is "
        "relative, a wing already quoted high picks up more points than the money and the "
        "surface steepens on its own — which an additive slope cannot do at any value.",
    )


@tool()
async def stress_curve(
    shocks: list[float] | None = Field(
        default=None,
        description="Underlying moves as fractions: -0.20 is a 20% fall. Defaults to "
        "-0.40 to +0.10 in 2% steps, which is wide enough on the downside that a "
        "short-gamma trough falls inside the window rather than on its edge.",
    ),
    vol_scenarios: list[VolScenarioSpec] | None = Field(
        default=None,
        description="One curve per volatility regime. Defaults to the two curves Risk "
        "Navigator itself draws: 'const' (no volatility response, its blue line) and "
        "'vol_coord' (IB's own volatility-coordinated model). Always keep a const curve "
        "in the set — it is the one that can be checked against Risk Navigator, and if it "
        "does not line up nothing else in the result is worth reading. Additive slopes "
        "are still there for a regime you want to state by hand, but prefer vol_coord: a "
        "slope shifts the surface in parallel, which on a book that is short the middle "
        "and long both wings is the wrong shape and not merely the wrong size.",
    ),
    vol_mode: Literal["sticky_strike", "sticky_moneyness"] = Field(
        default="sticky_strike",
        description="sticky_strike — the default — pins each strike to the volatility it "
        "holds today. This is what Risk Navigator's blue curve does, and it is the only "
        "setting under which the slope-0 curve can be checked against it: measured on a "
        "live index ratio book the two agree to within 1-3% at every shock from 0 to "
        "-30%. sticky_moneyness instead rereads each strike's volatility at the moneyness "
        "it lands on after the shock, off the portfolio's own surface, interpolated across "
        "strike and expiry — a defensible model, but a different one, and on that same "
        "book it deepened the trough by a factor of 1.8. Do not compare it to Risk "
        "Navigator.",
    ),
    scope: Literal["equity", "all"] = Field(
        default="equity",
        description="Which underlyings are on the shock axis. 'equity' excludes FX, rates "
        "and the rest outright and lists them under `excluded`; 'all' shocks everything "
        "by the same percentage, which off the equity axis is meaningless.",
    ),
    risk_groups: dict[str, str] | None = Field(
        default=None,
        description="Override a symbol's risk group, e.g. {'TLT': 'rates'}. IB publishes "
        "no asset class for a bond or gold ETF quoted as a stock.",
    ),
    betas: dict[str, float] | None = Field(
        default=None,
        description="Per-symbol share of the PRICE shock — unrelated to a scenario's "
        "volatility slope. Scales the move of that position's own underlying, options "
        "and futures included.",
    ),
    default_beta: float = 1.0,
    date_offset_days: int = Field(
        default=0, description="Roll the valuation date forward this many days (time decay)."
    ),
    date_offsets: list[int] | None = Field(
        default=None,
        description="A FAMILY of valuation dates in one call, as day offsets: [0, 3] is "
        "today and three days out. Every scenario is run at every offset, and all of "
        "them come out of ONE loading of the positions and prices — which is what makes "
        "the comparison mean anything, since calling this tool twice lets the book and "
        "the market move between the two answers. Mutually exclusive with "
        "date_offset_days and valuation_dates.",
    ),
    valuation_dates: list[str] | None = Field(
        default=None,
        description="The same family, given as ISO dates: ['2026-08-28', '2026-08-31']. "
        "Mutually exclusive with the offset forms. No calendar magic — the date is the "
        "date, weekend or not.",
    ),
    breakdown: Literal["symbol", "expiry", "both", "none"] = Field(
        default="symbol", description=BREAKDOWN_HELP
    ),
    vol_coord_decay: float | None = Field(
        default=None,
        description="Term damping of the vol_coord model: VR(t) = exp(-decay * t), so a "
        "front-month contract takes nearly the whole shock and a back month a fraction. "
        "IB documents that this function exists and is decreasing but not what it is, so "
        "this number is FITTED, not published. Left unset it uses your own stored "
        "calibration if calibrate_vol_coord has ever been run, and otherwise the factory "
        "fit of 4.736 — one number calibrated on one book, which has no claim on yours. "
        "`assumptions.volCoordDecaySource` says which of the three you got.",
    ),
    vol_coord_calibrated_to_years: float | None = Field(
        default=None,
        description="How far out in tenor `vol_coord_decay` was actually constrained. "
        "Defaults alongside the decay: your calibration's reach if you have one, and "
        "otherwise 0.345, the four months the shipped fit was constrained over. An "
        "exponential extrapolates to zero, which would price a one-year option as "
        "carrying no volatility risk at all in a crash, so positions past this are "
        "priced anyway and named in `warnings` rather than passing in silence.",
    ),
    bond_rate_shift_bp: float = 0.0,
    bond_duration_years: float = 5.0,
    fetch_skew: bool = Field(
        default=False,
        description="Let the surface pull neighbouring strikes from IB for expiries the "
        "portfolio holds too thinly. Costs extra market data requests, and is what to "
        "reach for when `volSurfaceUsed` comes back thin or empty.",
    ),
) -> dict[str, Any]:
    """The portfolio P&L curve under several volatility regimes at once — the
    data behind a risk graph, for plotting rather than for reading point by
    point.

    Risk Navigator draws two curves: a constant-volatility line and one from its
    own implied-volatility model, which is not documented and cannot be
    reproduced. This returns as many as you ask for, and the volatility
    assumption behind each is a number in the output rather than a black box:
    `volSlopeDown` is volatility points per 1% fall.

    **Read the slope-0 curve first.** It is the constant-volatility case and the
    only one with an external check — it should sit close to Risk Navigator's
    blue line. If it does not, the volatility lookup is wrong and no other
    scenario in the result means anything.

    **Check `volSurfaceUsed`.** It lists every quote the repricing actually
    read, as (underlying, tenor, strike, iv). Empty under `sticky_moneyness`
    means no expiry held three strikes, so no smile could be built and every
    option silently fell back to sticky_strike — the result looks perfectly
    normal and is not the model you asked for. `fetch_skew=true` fixes it at the
    cost of extra market data requests.

    How each curve is built, and where it is weakest:

    - Every scenario reprices **one** loading of the portfolio and **one**
      surface, so the curves differ by assumption alone. Calling the
      single-curve tool three times could not promise that: the book moves
      between calls.
    - The starting volatility is IB's own, per contract, out of a model that
      prices American exercise. The surface is used only for the *change* in
      volatility as a strike slides to new moneyness, which keeps IB's better
      number as the anchor and keeps every curve exactly zero at zero shock.
    - `vol_coord` reproduces IB's own model: volatility is **multiplied**, not
      shifted — a fall of X moves it by 10X relatively, a rise by -X, damped
      across tenors. Being relative is what makes the surface steepen by
      itself, since a wing already quoted at 41% takes more points than a 31%
      at-the-money out of the same scenario. The asymmetry is IB's documented
      one; the damping is fitted here and is not published, so `vol_coord_decay`
      is an input you should refit against your own Risk Navigator.
    - The additive `volSlopeDown` alternative is a **parallel** shift, flat
      across tenors. It cannot steepen at any value, and on a ratio book that
      is the difference between a curve that keeps falling and one that turns
      back up. Prefer `vol_coord` unless you specifically want a flat regime.
    - Equities move by the shock times their beta, futures and options by their
      own underlying's beta-scaled move. Bonds are flat unless
      `bond_rate_shift_bp` is set. FX is off the axis by default and reported
      under `excluded` with its market value — not held flat in silence.
    - Options IB would not model are repriced from a locally implied volatility
      where a mark price exists, flagged per position and in `warnings`, and
      held flat only when even that fails.

    - `date_offsets=[0, 3]` (or `valuation_dates`) crosses the scenarios with a
      family of valuation dates, so "today against Monday, when the August
      wings expire" is one call rather than two that the market moves between.
      Each entry under `curves` carries its own `valuationDate` and
      `dateOffsetDays`; `name` stays the scenario's and `label` distinguishes
      them. Time is the only thing that moves — today's spot and today's
      volatilities with the clock advanced, which is decay and not a forecast.
    - `breakdown='expiry'` puts `pnl_by_expiry` on every point and
      `troughByExpiry` on every curve, keyed on the option's settlement date.
      On a book running one root across many expiries that is the only way to
      see which expiry owns the trough; `pnl_by_symbol` shows one "ES" number
      for all of them.

    `pnl_pct_of_nlv` is on every point, and `netLiquidation` at the top. Quote
    the fraction rather than the amount when comparing two dates or two
    accounts. **Check `reconciled` before quoting any of it.**
    """
    offsets = S.resolve_offsets(
        date_offset_days=date_offset_days,
        date_offsets=date_offsets,
        valuation_dates=valuation_dates,
    )
    cfg = _stress_config(
        list(shocks) if shocks else list(S.DEFAULT_SHOCKS),
        vol_mode,
        0.0,
        0.0,
        0.0,
        offsets[0],
        betas,
        default_beta,
        bond_rate_shift_bp,
        bond_duration_years,
        fetch_skew,
        scope,
        risk_groups,
        breakdown,
    )
    if vol_scenarios is None:
        scenarios = list(S.DEFAULT_VOL_SCENARIOS)
    else:
        # An empty list is not a request for the defaults, it is a caller
        # mistake, and stress_curve says so rather than quietly substituting a
        # set of scenarios nobody asked for.
        scenarios = [
            S.VolScenario(
                name=v.name,
                vol_slope_down=v.vol_slope_down,
                vol_slope_up=v.vol_slope_up,
                vol_bump=v.vol_bump,
                vol_coord=v.vol_coord,
            )
            for v in vol_scenarios
        ]
    # Unset means "whatever the calibration says", which StressConfig already
    # resolved when it was built. Only an explicit value overrides it.
    if vol_coord_decay is not None:
        cfg.vol_coord_decay = vol_coord_decay
    if vol_coord_calibrated_to_years is not None:
        cfg.vol_coord_calibrated_to_years = vol_coord_calibrated_to_years
    return await S.stress_curve(cfg, scenarios, offsets)


@tool()
async def stress_whatif(
    legs: list[C.Leg] = Field(
        description="Hypothetical legs to add. Each is either a conid, or "
        "symbol+secType+expiry+strike+right, with action BUY/SELL and a quantity."
    ),
    shocks: list[float] = Field(description="Underlying moves as fractions, as above."),
    vol_mode: Literal["sticky_strike", "sticky_moneyness"] = "sticky_strike",
    vol_bump: float = 0.0,
    vol_slope_down: float = 0.0,
    vol_slope_up: float = 0.0,
    date_offset_days: int = 0,
    valuation_date: str | None = Field(default=None, description=VALUATION_DATE_HELP),
    breakdown: Literal["symbol", "expiry", "both", "none"] = Field(
        default="symbol", description=BREAKDOWN_HELP
    ),
    scope: Literal["equity", "all"] = "equity",
    risk_groups: dict[str, str] | None = None,
    betas: dict[str, float] | None = None,
    default_beta: float = 1.0,
    bond_rate_shift_bp: float = 0.0,
    bond_duration_years: float = 5.0,
    fetch_skew: bool = False,
) -> dict[str, Any]:
    """The same stress run, with hypothetical legs added — three curves and
    three troughs: the portfolio as it stands, the portfolio plus the legs, and
    the difference.

    This is what replaces reading Risk Navigator's What-If by hand: "if I add
    N puts at strike K expiring E, where does the trough move to?"

    Read all three troughs. **The trough of the difference is not the
    difference of the troughs** — adding protection moves where the worst point
    sits as well as how deep it is, and comparing only the depths hides the
    move. A structure that lifts the bottom by very little may still have
    pushed it from −8% out to −15%, which is the part that matters.

    Hypothetical options are priced off IB's current model greeks for those
    exact contracts, so both curves start from the same volatilities and the
    difference is the structure alone. A leg that cannot be resolved or priced
    is reported in `legProblems` and left out of the second curve; the
    comparison then covers only the legs that did resolve, and says so.

    With `breakdown='expiry'` all three curves are broken out by settlement
    date, and the difference curve's `pnl_by_expiry` is where the structure
    actually landed: the expiry you traded against moves and the ones you did
    not read zero, which is how a hedge is told apart from a change of subject.
    A short expiry whose row goes to roughly nothing in `withLegs` is one the
    structure has closed out.

    Nothing is sent to IB's order path here — this is pure local repricing. For
    what the structure costs in margin, use whatif_order.
    """
    cfg = _stress_config(
        shocks,
        vol_mode,
        vol_bump,
        vol_slope_down,
        vol_slope_up,
        S.resolve_offsets(
            date_offset_days=date_offset_days, valuation_date=valuation_date
        )[0],
        betas,
        default_beta,
        bond_rate_shift_bp,
        bond_duration_years,
        fetch_skew,
        scope,
        risk_groups,
        breakdown,
    )
    return await S.stress_whatif(legs, cfg)


class RiskNavigatorPoint(BaseModel):
    """One reading off Risk Navigator's Vol.Coord. curve."""

    model_config = ConfigDict(populate_by_name=True)

    shock: float = Field(
        description="Where on the x-axis you read it, as a FRACTION: -0.20 is a 20% "
        "fall. Not a percent.",
        validation_alias=AliasChoices("shock", "shock_pct_as_fraction"),
    )
    pnl: float = Field(
        description="The portfolio P&L Risk Navigator shows there, in account currency, "
        "signed: a loss is negative.",
        validation_alias=AliasChoices("pnl", "target"),
    )


@tool(read_only=False, idempotent=False)
async def calibrate_vol_coord(
    targets: list[RiskNavigatorPoint] = Field(
        description="Four or more points read off Risk Navigator's Vol.Coord. curve, "
        "spread across the range you care about. Three is the minimum the fit will "
        "accept and it constrains almost nothing; points bunched near the money "
        "constrain nothing at all. Get them from the Equity tab if you run the default "
        "scope='equity', so the curve you are fitting to excludes FX and rates the same "
        "way this engine does."
    ),
    scope: Literal["equity", "all"] = Field(
        default="equity",
        description="Match the Risk Navigator tab the readings came from. Its Equity tab "
        "is 'equity', the default.",
    ),
    vol_mode: Literal["sticky_strike", "sticky_moneyness"] = Field(
        default="sticky_strike",
        description="Match the run you intend to use the decay in. sticky_strike is the "
        "default everywhere else and is what Risk Navigator's own curve does.",
    ),
    risk_groups: dict[str, str] | None = Field(
        default=None,
        description="Same override as elsewhere, e.g. {'TLT': 'rates'}. Use the same one "
        "you pass to stress_curve, or the fit is against a different book than the runs "
        "that will use it.",
    ),
    betas: dict[str, float] | None = None,
    default_beta: float = 1.0,
    fetch_skew: bool = False,
    persist: bool = Field(
        default=True,
        description="Store the fit as this machine's standing calibration, so every "
        "later stress_curve uses it without being told. Set false to see the fit "
        "without adopting it. A fit against a portfolio that does not reconcile is "
        "never stored, whatever this says.",
    ),
) -> dict[str, Any]:
    """Fit `vol_coord_decay` to YOUR Risk Navigator, and keep the result.

    `vol_coord` reproduces IB's volatility-coordinated model. Its asymmetry — a
    fall moves volatility ten times as hard as a rise — is IB's, documented.
    Its term damping `VR(t)` is not: IB says only that the function exists and
    is decreasing. **The number this server ships was fitted to one screenshot
    of somebody else's Risk Navigator, from nine points read off a chart by
    eye.** It has no claim on your book, every vol_coord result says so in
    `warnings`, and this tool is how you replace it.

    What to do:

    1. In TWS open Risk Navigator's risk graph, on the Equity tab if you use
       the default scope.
    2. Read the **Vol.Coord.** curve — the one that responds to volatility, not
       the constant-volatility line — at four or more shocks spread across the
       range you care about.
    3. Pass them in as `{shock: -0.20, pnl: -28000}` pairs, shocks as
       fractions.

    The fit comes back with the residual at every point and, more usefully,
    with what to distrust about it: the tenor range your positions actually
    constrain, and the most extreme volatility the fitted decay produces. **A
    decay that reproduces the curve by pricing a wing at 150% has fitted the
    chart rather than the market**, and it says so rather than leaving you to
    find out three layers down in a P&L.

    Unless `persist=false`, the fit is written to disk and becomes the default
    `vol_coord_decay` for every later `stress_curve` on this machine — no
    restart, no carrying the number by hand — together with what it was fitted
    against, which is then reported in `assumptions.volCoordDecaySource`. Set
    `IBKR_CALIBRATION_FILE` to move the file. It is the only thing this server
    writes.

    A fit taken against a portfolio that does not reconcile is returned but
    **not** stored. The reason is asymmetry of failure: a curve that is missing
    a position announces itself through `reconciled`, while a decay that
    absorbed the same gap would go on silently deforming every vol_coord run
    afterwards.

    Nothing here trades or quotes. It reads positions and reprices locally.
    """
    points = {float(p.shock): float(p.pnl) for p in targets}
    if len(points) < 3:
        raise ValueError(
            "a decay cannot be fitted to fewer than three distinct shocks; four or more "
            "spread across the range is what makes the answer mean anything"
        )
    if min(points) >= 0:
        raise ValueError(
            "none of these shocks is negative. The volatility response is asymmetric and "
            "almost all of it is on the downside; a fit from the upside alone does not "
            "constrain the damping at all."
        )
    cfg = _stress_config(
        sorted(points),
        vol_mode,
        0.0,
        0.0,
        0.0,
        0,
        betas,
        default_beta,
        0.0,
        5.0,
        fetch_skew,
        scope,
        risk_groups,
    )
    return await S.run_calibration(cfg, points, persist=persist)


@tool(read_only=False, idempotent=True)
async def whatif_order(
    legs: list[C.Leg] = Field(
        description="The structure to evaluate. Each leg is a conid, or "
        "symbol+secType+expiry+strike+right, with action BUY/SELL and a quantity."
    ),
) -> dict[str, Any]:
    """IB's own margin impact for a hypothetical structure. **Nothing reaches
    the market.**

    Each order carries `whatIf=True`, which IB evaluates in its margin engine
    and discards: it is never routed, never acknowledged as live, never appears
    in the order book. The tool is still gated behind `IBKR_ENABLE_WHATIF=true`
    because it is the only thing in this server that touches the order path at
    all; with the gate closed it sends nothing and returns `blocked: true`.

    Returns initMargin, maintMargin and equityWithLoan before/after/change,
    plus commission and any warningText, in two views:

    - `perLeg` — each leg evaluated on its own.
    - `cumulative` — legs 1..k as a combo, for every k, so you can see where
      the offset appears.

    **Read the combined figure, not the sum of the legs.** SPAN offsets the legs
    against each other and against what the account already holds, so the two
    differ — the difference is reported as `offset.spanOffset`. IB's what-if on
    arbitrary multi-leg combos is unreliable and will sometimes return nothing;
    that is reported per step and does not mean the structure is invalid.

    If every call fails with no margin figures, check whether TWS has
    "Read-Only API" enabled — that setting blocks what-if orders too.
    """
    return await M.whatif_order(legs)


def main() -> None:
    """Serve, over whichever transport was configured.

    The HTTP branch runs uvicorn from inside ``mcp.run``, which awaits
    ``Server.serve()`` on the loop it is already on rather than calling
    ``uvicorn.run()``. That matters more than it looks: ``uvicorn.run`` selects
    ``loop="auto"``, which is uvloop wherever uvloop is installed, and ib_async
    drives its own socket transport on whatever loop it finds. Keeping the
    process on plain asyncio keeps ib_async on the loop it is tested against.
    A future deployment that prefers the uvicorn CLI over this entry point
    needs ``--loop asyncio`` to stay equivalent.
    """
    if settings.mcp_transport == "http":
        # basicConfig above pins the root at WARNING, which is right for a
        # subprocess talking MCP over a pipe and wrong for a service somebody
        # has to find the endpoint of. The handler itself has no level, so
        # lifting this logger is enough to let the line through.
        log.setLevel(logging.INFO)
        log.info(
            "serving MCP over streamable HTTP at http://%s:%s%s",
            settings.mcp_host,
            settings.mcp_port,
            settings.mcp_path,
        )
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
