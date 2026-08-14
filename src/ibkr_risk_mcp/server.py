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

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import marketdata as MD
from . import stress as S
from . import contracts as C
from . import margin as M
from .config import settings
from .connection import AccountAmbiguous, IBUnavailable, connection

logging.basicConfig(level=logging.WARNING)

mcp = FastMCP(
    "ibkr-risk-mcp",
    instructions="""This server exposes Interactive Brokers' *risk* data: model greeks,
IB's implied volatility surface, what-if margin, and a local stress engine that
rebuilds the portfolio P&L curve across underlying shocks.

It deliberately does not duplicate the official IBKR connector. Positions,
balances, orders, trades, performance, spot and historical prices, option
chains and watchlists all come from there. Use this server only for questions
about risk: where the P&L trough sits, what a hypothetical structure does to
it, what margin it needs, and what IB's volatility surface looks like.

It needs TWS or IB Gateway running on this machine with the API enabled. If any
tool fails, call check_connection first — it distinguishes "not running" from
"API switch off" from "no account logged in", which have different fixes.

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

The server is read-only. whatif_order is the only tool that reaches IB's order
path, it sends whatIf=True orders that are never routed, and it is disabled
unless the server was started with IBKR_ENABLE_WHATIF=true. There is no tool
here that can submit a live order.""",
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
    date_offset_days: int,
    betas: dict[str, float] | None,
    default_beta: float,
    bond_rate_shift_bp: float,
    bond_duration_years: float,
    fetch_skew: bool,
) -> S.StressConfig:
    return S.StressConfig(
        shocks=list(shocks),
        vol_mode=vol_mode,
        vol_bump=vol_bump,
        date_offset_days=date_offset_days,
        betas=dict(betas or {}),
        default_beta=default_beta,
        bond_rate_shift_bp=bond_rate_shift_bp,
        bond_duration_years=bond_duration_years,
        fetch_skew=fetch_skew,
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
        default=0.0, description="Added to every volatility, in points: 0.05 is +5 points."
    ),
    date_offset_days: int = Field(
        default=0, description="Roll the valuation date forward this many days (time decay)."
    ),
    betas: dict[str, float] | None = Field(
        default=None,
        description="Per-symbol beta for equity positions, e.g. {'AAPL': 1.2}. Options and "
        "futures always move with their own underlying.",
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

    - all underlyings are shocked by the same percentage at once, which is Risk
      Navigator's own default assumption. Equity positions can be scaled with
      `betas`; options and futures move one-for-one with their underlying.
    - options are repriced with Black-76 on the shocked forward using **IB's**
      implied volatility. Equity options are carried from spot using IB's
      pvDividend, so both kinds go through one pricer.
    - P&L is model-price-now against model-price-shocked, so the curve is
      exactly zero at zero shock by construction. The gap between the local
      model and IB's own price is reported per position as `modelVsMarket`
      instead of being folded into the curve. Expect it to be small out of the
      money and a couple of percent in the money, where IB prices the early
      exercise that Black-76 has no room for — so the curve understates losses
      slightly once options go deep in the money.
    - bonds are held flat unless `bond_rate_shift_bp` is set; anything this
      server does not model is held flat and named in `warnings`.

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
        date_offset_days,
        betas,
        default_beta,
        bond_rate_shift_bp,
        bond_duration_years,
        fetch_skew,
    )
    return await S.stress_portfolio(cfg)


@tool()
async def stress_whatif(
    legs: list[C.Leg] = Field(
        description="Hypothetical legs to add. Each is either a conid, or "
        "symbol+secType+expiry+strike+right, with action BUY/SELL and a quantity."
    ),
    shocks: list[float] = Field(description="Underlying moves as fractions, as above."),
    vol_mode: Literal["sticky_strike", "sticky_moneyness"] = "sticky_strike",
    vol_bump: float = 0.0,
    date_offset_days: int = 0,
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

    Nothing is sent to IB's order path here — this is pure local repricing. For
    what the structure costs in margin, use whatif_order.
    """
    cfg = _stress_config(
        shocks,
        vol_mode,
        vol_bump,
        date_offset_days,
        betas,
        default_beta,
        bond_rate_shift_bp,
        bond_duration_years,
        fetch_skew,
    )
    return await S.stress_whatif(legs, cfg)


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
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
