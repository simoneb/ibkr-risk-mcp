"""End-to-end check against a live TWS, exercising the whole server.

Run it with TWS or IB Gateway up and the API enabled:

    uv run python scripts/smoke_test.py
    uv run python scripts/smoke_test.py --shock-range 0.30 --step 0.01

It reads the same environment variables as the server, so a `.env` pointing at
the paper port is enough to try everything safely. Nothing it does can place an
order: the what-if step sends `whatIf=True`, and it picks a strike far out of
the money on purpose so that a misconfigured TWS could not fill it even if the
order somehow escaped — belt and braces on a path that is already inert.

Each step prints PASS, WARN or FAIL and the run exits non-zero if anything
failed. A WARN is for things that are the account's problem rather than the
server's — no option positions to fetch greeks for, no market data entitlement
— because those should not read as a broken build.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from ibkr_risk_mcp import contracts as C  # noqa: E402
from ibkr_risk_mcp import margin as M  # noqa: E402
from ibkr_risk_mcp import marketdata as MD  # noqa: E402
from ibkr_risk_mcp import stress as S  # noqa: E402
from ibkr_risk_mcp.config import settings  # noqa: E402
from ibkr_risk_mcp.connection import connection  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def record(status: str, step: str, detail: str = "") -> None:
    RESULTS.append((status, step, detail))
    print(f"[{status:4}] {step}" + (f"\n       {detail}" if detail else ""), flush=True)


def money(value: Any) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


async def step_connection() -> bool:
    probe = await connection.probe()
    if probe["state"] != "connected":
        record("FAIL", "check_connection", f"{probe['state']}: {probe['hint']}")
        return False
    record(
        "PASS",
        "check_connection",
        f"accounts={probe['accounts']} serverVersion={probe.get('serverVersion')} "
        f"port={settings.port} clientId={settings.client_id}",
    )
    return True


async def step_margin() -> None:
    summary = await M.margin_summary()
    net_liq = (summary["summary"].get("NetLiquidation") or {}).get("total")
    if net_liq is None:
        record("FAIL", "get_margin_summary", "IB reported no NetLiquidation")
        return
    init = (summary["summary"].get("FullInitMarginReq") or {}).get("total")
    excess = (summary["summary"].get("ExcessLiquidity") or {}).get("total")
    segments = sorted(
        {seg for row in summary["summary"].values() for seg in row if seg != "currency"}
    )
    record(
        "PASS",
        "get_margin_summary",
        f"NetLiq={money(net_liq)} InitMargin={money(init)} ExcessLiq={money(excess)} "
        f"segments={segments}",
    )


async def step_greeks() -> list[MD.Holding]:
    holdings = await MD.load_holdings(with_greeks=True)
    options = [h for h in holdings if h.is_option]
    if not holdings:
        record("WARN", "get_position_greeks", "the account holds no positions")
        return holdings
    if not options:
        record(
            "WARN",
            "get_position_greeks",
            f"{len(holdings)} position(s), none of them options — nothing to fetch greeks for",
        )
        return holdings
    missing = [h for h in options if h.greeks is None]
    detail = f"{len(options) - len(missing)}/{len(options)} option positions have model greeks"
    if missing:
        detail += "\n       missing: " + "; ".join(
            f"{h.contract.localSymbol or h.contract.symbol} ({h.greeks_error})"
            for h in missing[:5]
        )
        record("WARN", "get_position_greeks", detail)
    else:
        sample = options[0]
        detail += (
            f"\n       e.g. {sample.contract.localSymbol}: IV="
            f"{sample.greeks['impliedVol']:.4f} delta={sample.greeks['delta']:.4f} "
            f"undPrice={sample.greeks['undPrice']}"
        )
        record("PASS", "get_position_greeks", detail)

    # The settlement-date normalisation is worth showing explicitly: it is the
    # one transformation whose result cannot be checked against TWS by eye.
    shifted = [
        h
        for h in options
        if C.settlement_date(h.contract).isoformat()
        != C.parse_ib_date(h.contract.lastTradeDateOrContractMonth).isoformat()
    ]
    if shifted:
        record(
            "PASS",
            "settlement-date normalisation",
            "; ".join(
                f"{h.contract.localSymbol}: lastTrade="
                f"{h.contract.lastTradeDateOrContractMonth} -> settles="
                f"{C.settlement_date(h.contract)}"
                for h in shifted[:5]
            ),
        )
    return holdings


async def step_stress(shock_range: float, step: float) -> dict[str, Any] | None:
    n = int(round(shock_range / step))
    shocks = [round(i * step, 6) for i in range(-n, n + 1)]
    cfg = S.StressConfig(shocks=shocks)
    result = await S.stress_portfolio(cfg)

    trough = result["trough"]
    refined = result.get("troughRefined")
    detail = (
        f"{len(shocks)} shocks from {shocks[0]:+.0%} to {shocks[-1]:+.0%}\n"
        f"       trough: {trough['pnl']:,.0f} at {trough['shock']:+.1%}"
    )
    if refined:
        detail += f" (interpolated: {refined['pnl']:,.0f} at {refined['shock']:+.2%})"
    zero = next(r for r in result["curve"] if r["shock"] == 0.0)
    detail += f"\n       P&L at zero shock: {zero['pnl_total']:,.2f} (must be 0.00)"
    if result["warnings"]:
        detail += "\n       warnings: " + " | ".join(result["warnings"])
    record("FAIL" if abs(zero["pnl_total"]) > 0.01 else "PASS", "stress_portfolio", detail)

    rec = result["reconciliation"]
    rec_detail = (
        f"NetLiq={money(rec.get('netLiquidation'))} "
        f"derived={money(rec.get('derivedNetLiquidation'))} "
        f"residual={money(rec.get('residual'))}"
    )
    if rec.get("residualPct") is not None:
        rec_detail += f" ({rec['residualPct']:+.3%})"
    if rec.get("reason"):
        rec_detail += f"\n       {rec['reason']}"
    record("PASS" if rec.get("reconciled") else "FAIL", "reconciliation at zero shock", rec_detail)
    return result


async def step_stress_curve() -> None:
    """The multi-regime curve, run the way the tool actually defaults.

    Forcing a non-default vol_mode here was worse than useless: the step passed
    while printing a constant-volatility trough of -91,923 where the default
    gives -52,311, and the number a reader would have taken to Risk Navigator
    was the one that cannot be compared to it. The surface path still needs
    exercising, so it gets its own run below rather than distorting this one.
    """
    cfg = S.StressConfig(shocks=list(S.DEFAULT_SHOCKS))
    result = await S.stress_curve(cfg, list(S.DEFAULT_VOL_SCENARIOS))

    curves = result["curves"]
    lines = []
    for c in curves:
        at = f"{c['minAtShockPct']:+.0f}%"
        if c.get("minAtUnderlying"):
            at += f" ({c['minAtUnderlying']:,.0f})"
        pct = f" [{c['minPnlPctOfNlv']:+.2%} of NLV]" if c.get("minPnlPctOfNlv") else ""
        how = "IB vol-coordinated" if c.get("volCoord") else f"slope {c['volSlopeDown']:g}"
        lines.append(f"{c['name']} ({how}): {c['minPnl']:,.0f} at {at}{pct}")
    detail = f"{len(curves)} curve(s) over {len(curves[0]['points'])} shocks\n       " + (
        "\n       ".join(lines)
    )

    const = next((c for c in curves if c["volSlopeDown"] == 0.0), None)
    if const is None:
        record("WARN", "stress_curve", detail + "\n       no slope-0 curve to check against")
        return
    zero = next((p for p in const["points"] if p["shock"] == 0.0), None)
    if zero is None:
        record("WARN", "stress_curve", detail + "\n       no zero shock on the axis to check")
        return
    detail += f"\n       const P&L at zero shock: {zero['pnl']:,.2f} (must be 0.00)"
    record("FAIL" if abs(zero["pnl"]) > 0.01 else "PASS", "stress_curve", detail)

    # The default is sticky_strike, which builds no surface by design. The
    # surface path is real code and still has to run, so it gets its own pass.
    moneyness = await S.stress_curve(
        S.StressConfig(shocks=[-0.10, 0.0], vol_mode="sticky_moneyness"),
        [S.VolScenario("const")],
    )
    surface = moneyness["volSurfaceUsed"]
    if not surface:
        record(
            "WARN",
            "vol surface used",
            "EMPTY under sticky_moneyness: no expiry held three strikes, so every option "
            "was repriced sticky_strike. Not a server fault if the account holds few "
            "options, but the moneyness response would not be in such a curve.",
        )
        return
    quotes = sum(len(row["points"]) for row in surface)
    record(
        "PASS",
        "vol surface used",
        f"{quotes} quote(s) over {len(surface)} tenor(s): "
        + "; ".join(
            f"{row['underlying']} {row['yearsToExpiry']:.3f}y fwd {row['forward']:,.1f} "
            f"({len(row['points'])} strikes)"
            for row in surface[:5]
        ),
    )


async def step_expiry_and_dates() -> None:
    """The expiry breakdown and a family of valuation dates, in one call.

    Two invariants worth exercising against a real book rather than a fixture.
    The breakdown must sum to each point's own total — that is what makes it
    checkable rather than merely plausible — and two dates asked for at once
    must come back as separate curves off one snapshot.
    """
    result = await S.stress_curve(
        S.StressConfig(shocks=list(S.DEFAULT_SHOCKS), breakdown="expiry"),
        [S.VolScenario("const")],
        date_offsets=[0, 3],
    )
    curves = result["curves"]
    dates = [c["valuationDate"] for c in curves]
    if len(curves) != 2:
        record("FAIL", "stress_curve dates", f"asked for 2 dates, got {len(curves)} curve(s)")
        return

    worst = 0.0
    for curve in curves:
        for point in curve["points"]:
            drift = abs(sum(point["pnl_by_expiry"].values()) - point["pnl"])
            worst = max(worst, drift)

    today = curves[0]
    rows = today.get("troughByExpiry") or []
    top = "; ".join(
        f"{r['key']} {r.get('pnlAtPortfolioTrough', r['pnl']):,.0f}" for r in rows[:4]
    )
    detail = (
        f"{len(curves)} curve(s) valued {dates[0]} and {dates[1]} off one snapshot\n"
        f"       {len(rows)} expiry bucket(s); worst at the trough: {top or 'none'}\n"
        f"       largest gap between the breakdown and its point total: {worst:.4f}"
    )
    # A cent of rounding per key is expected; anything larger means a position
    # is in the total and in no bucket, which would make the breakdown a
    # different portfolio from the curve above it.
    record("FAIL" if worst > 0.05 * max(len(rows), 1) else "PASS", "expiry breakdown", detail)


async def step_stress_whatif(
    holdings: list[MD.Holding], fallback_symbol: str, fallback_sec_type: str
) -> None:
    leg = await _far_otm_leg(holdings) or await _fallback_leg(fallback_symbol, fallback_sec_type)
    if leg is None:
        record("WARN", "stress_whatif", "could not build a hypothetical leg to test with")
        return
    cfg = S.StressConfig(shocks=[round(-0.30 + i * 0.02, 4) for i in range(31)])
    result = await S.stress_whatif([leg], cfg)
    t = result["troughs"]
    # A leg that changes nothing at any shock means the comparison never
    # happened — unpriced, or taken off the axis by the scope. That used to
    # report PASS, which is the worst way for a check to be wrong.
    moved = any(row["pnl_total"] for row in result["difference"]["curve"])
    status = "WARN" if result["legProblems"] else ("PASS" if moved else "FAIL")
    record(
        status,
        "stress_whatif",
        f"base trough {t['base']['pnl']:,.0f} at {t['base']['shock']:+.1%} -> with leg "
        f"{t['withLegs']['pnl']:,.0f} at {t['withLegs']['shock']:+.1%}; the difference "
        f"troughs at {t['difference']['pnl']:,.0f} at {t['difference']['shock']:+.1%}"
        + ("" if moved else "\n       the leg moved the curve by exactly "
           "zero at every shock: it was either not priced or excluded by the "
           "scope, so nothing was actually compared")
        + (f"\n       legProblems: {result['legProblems']}" if result["legProblems"] else ""),
    )


async def _fallback_leg(symbol: str, sec_type: str | None = None) -> C.Leg | None:
    """A deeply out-of-the-money put on a named underlying, for an account that
    holds no options.

    A fresh paper account is empty, and warning out of the surface and what-if
    steps would leave the two most intricate paths in the server unexercised on
    exactly the run where you most want them checked.
    """
    try:
        underlying, alternatives = await MD.resolve_underlying(symbol, sec_type)
    except Exception as exc:
        record("WARN", f"resolve {symbol}", str(exc))
        return None
    if alternatives:
        record(
            "PASS",
            f"ambiguous underlying {symbol}",
            f"resolved as {underlying.secType} "
            f"({underlying.localSymbol or underlying.symbol}); also matches "
            f"{', '.join(alternatives)} — reported rather than silently chosen",
        )
    fop = underlying.secType == "FUT"
    chain = await MD.chain_details(
        symbol=underlying.symbol,
        expiry=underlying.lastTradeDateOrContractMonth
        if fop
        else _nearest_equity_expiry(underlying),
        sec_type="FOP" if fop else "OPT",
        exchange=underlying.exchange if fop else "SMART",
        currency=underlying.currency or "USD",
        rights=("P",),
    )
    if not chain:
        return None
    # About 30% below the money. The very lowest listed strike would be far
    # enough out to carry no margin at all, which tests the plumbing and
    # nothing else.
    centre = MD.listed_centre([d.contract.strike for d in chain]) or 0.0
    target = min(chain, key=lambda d: abs(d.contract.strike - centre * 0.70))
    return C.Leg(conid=target.contract.conId, action="BUY", quantity=1)


def _nearest_equity_expiry(underlying) -> str:
    from datetime import date, timedelta

    target = date.today() + timedelta(days=30)
    while target.weekday() != 4:
        target += timedelta(days=1)
    return target.strftime("%Y%m%d")


async def _far_otm_leg(holdings: list[MD.Holding]) -> C.Leg | None:
    """A single deeply out-of-the-money put on an underlying already held.

    The strike is taken from the listed chain rather than computed. Working out
    "70% of spot rounded to the tick" produces strikes like 5487 that the
    exchange has never listed, and resolve_leg rightly refuses them — snapping
    a strike silently is how you end up reporting risk for a contract nobody
    asked about. The test should pick a real one; the library should keep
    saying no.

    Deliberately far out of the money: the what-if path never reaches a market,
    but an order that could not fill even if it did is a cheaper thing to be
    wrong about.
    """
    options = [h for h in holdings if h.is_option and h.greeks and h.greeks.get("undPrice")]
    # It has to be an *equity* underlying. Taking whatever came first put the
    # leg on the account's CAD option, which scope='equity' then excluded — so
    # the what-if step compared a curve against itself, reported a difference of
    # exactly zero, and passed without testing anything.
    # Two conditions, both learned the hard way. It has to be an *equity*
    # underlying, or scope='equity' excludes the leg and the what-if compares a
    # curve against itself. And it has to be one IB actually prices: the first
    # equity option here was a CSCO strike the account has no data entitlement
    # for, so the leg came back unpriced and the difference was again exactly
    # zero. Prefer an underlying whose greeks arrived from IB.
    options.sort(
        key=lambda h: (
            C.risk_group(h.contract) != "equity",
            bool((h.greeks or {}).get("source")),
        )
    )
    if not options or C.risk_group(options[0].contract) != "equity":
        return None
    ref = options[0]
    und = float(ref.greeks["undPrice"])
    chain = await MD.chain_details(
        symbol=ref.contract.symbol,
        expiry=ref.contract.lastTradeDateOrContractMonth,
        sec_type=ref.contract.secType,
        exchange=ref.contract.exchange,
        currency=ref.contract.currency,
        trading_class=ref.contract.tradingClass,
        rights=("P",),
    )
    if not chain:
        return None
    target = min(chain, key=lambda d: abs(d.contract.strike - und * 0.70))
    return C.Leg(conid=target.contract.conId, action="BUY", quantity=1)


async def step_whatif(
    holdings: list[MD.Holding], fallback_symbol: str, fallback_sec_type: str
) -> None:
    if not settings.enable_whatif:
        record(
            "WARN",
            "whatif_order",
            "skipped: IBKR_ENABLE_WHATIF is not true, so the tool sends nothing",
        )
        return
    leg = await _far_otm_leg(holdings) or await _fallback_leg(fallback_symbol, fallback_sec_type)
    if leg is None:
        record("WARN", "whatif_order", "could not build a hypothetical leg to test with")
        return
    result = await M.whatif_order([leg])
    first = result["perLeg"][0] if result["perLeg"] else {}
    if "error" in first:
        record("FAIL", "whatif_order", first["error"])
        return
    wi = first["whatIf"]
    change = wi.get("initMarginChange")
    detail = (
        f"BUY 1 {first['contract'].get('localSymbol')} — initMarginChange="
        f"{money(change)}, maintMarginChange={money(wi.get('maintMarginChange'))}, "
        f"commission={wi.get('commission')}"
    )
    if wi.get("warningText"):
        detail += f"\n       IB warning: {wi['warningText']}"
    # A long put cannot raise the initial margin requirement by more than its
    # premium; a wildly large number means the figures were misparsed.
    sane = change is not None and -1e7 < change < 1e7
    record("PASS" if sane else "FAIL", "whatif_order", detail)

    # Two legs is where the interesting part is: SPAN offsets them against each
    # other, so the combo costs materially less than the sum of the parts, and
    # a run that only ever tests one leg never sees that happen.
    if leg.conid:
        spread = await _spread_from(leg)
        if spread:
            out = await M.whatif_order(spread)
            offset = out["offset"]
            record(
                "PASS" if offset["spanOffset"] is not None else "WARN",
                "whatif_order (two legs, SPAN offset)",
                f"sum of legs={money(offset['sumOfLegInitMarginChange'])}, "
                f"combined={money(offset['combinedInitMarginChange'])}, "
                f"offset={money(offset['spanOffset'])}",
            )


async def _spread_from(leg: C.Leg) -> list[C.Leg] | None:
    """The given put plus a short one a few strikes higher — an ordinary spread,
    which is what SPAN is built to net off."""
    detail = await MD.details_for_conid(int(leg.conid))
    if detail is None:
        return None
    contract = detail.contract
    chain = await MD.chain_details(
        symbol=contract.symbol,
        expiry=contract.lastTradeDateOrContractMonth,
        sec_type=contract.secType,
        exchange=contract.exchange,
        currency=contract.currency,
        trading_class=contract.tradingClass,
        rights=("P",),
    )
    higher = sorted(d for d in (c.contract.strike for c in chain) if d > contract.strike)
    if len(higher) < 3:
        return None
    return [
        C.Leg(conid=leg.conid, action="BUY", quantity=1),
        C.Leg(
            conid=next(
                c.contract.conId for c in chain if c.contract.strike == higher[2]
            ),
            action="SELL",
            quantity=1,
        ),
    ]


async def step_surface(
    holdings: list[MD.Holding], fallback_symbol: str, fallback_sec_type: str
) -> None:
    options = [h for h in holdings if h.is_option]
    if options:
        ref = options[0].contract
    else:
        leg = await _fallback_leg(fallback_symbol, fallback_sec_type)
        if leg is None:
            record("WARN", "get_vol_surface", "no underlying to pull a surface for")
            return
        detail = await MD.details_for_conid(int(leg.conid))
        if detail is None:
            record("WARN", "get_vol_surface", "no underlying to pull a surface for")
            return
        ref = detail.contract
    req = MD.SurfaceRequest(
        underlying=ref.symbol,
        expiries=[ref.lastTradeDateOrContractMonth],
        rights=("P",),
        max_strikes_per_expiry=7,
        trading_class=ref.tradingClass or None,
        # A futures option's underlying is a future. Leaving this off resolves
        # ES to Eversource Energy and pulls a surface for the wrong instrument
        # without complaining about it.
        sec_type="FUT" if ref.secType == "FOP" else "STK",
    )
    result = await MD.vol_surface(req)
    if not result.rows:
        record(
            "WARN",
            "get_vol_surface",
            f"no volatilities for {ref.symbol} {ref.lastTradeDateOrContractMonth}: "
            + " | ".join(result.warnings or ["unknown"]),
        )
        return
    ivs = [(r["strike"], r["impliedVol"]) for r in result.rows]
    record(
        "PASS",
        "get_vol_surface",
        f"{ref.symbol} {ref.lastTradeDateOrContractMonth}: "
        + ", ".join(f"{k:g}@{v:.3f}" for k, v in sorted(ivs))
        + (f"\n       missing: {len(result.missing)}" if result.missing else ""),
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shock-range", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--probe-underlying",
        default="ES",
        help="Underlying to exercise the surface and what-if steps against when the "
        "account holds no options — as a fresh paper account does not.",
    )
    parser.add_argument(
        "--probe-sec-type",
        default="FUT",
        choices=["STK", "IND", "FUT"],
        help="Which instrument that symbol means. ES is both the E-mini S&P future and "
        "Eversource Energy, and without this the stock wins.",
    )
    args = parser.parse_args()

    print(
        f"ibkr-risk-mcp smoke test — {settings.host}:{settings.port} "
        f"clientId={settings.client_id} marketDataType={settings.market_data_type} "
        f"whatIf={'on' if settings.enable_whatif else 'off'}\n"
    )
    try:
        if not await step_connection():
            return 1
        await step_margin()
        holdings = await step_greeks()
        await step_surface(holdings, args.probe_underlying, args.probe_sec_type)
        await step_stress(args.shock_range, args.step)
        await step_stress_curve()
        await step_expiry_and_dates()
        await step_stress_whatif(holdings, args.probe_underlying, args.probe_sec_type)
        await step_whatif(holdings, args.probe_underlying, args.probe_sec_type)
    finally:
        await connection.disconnect()

    failed = [r for r in RESULTS if r[0] == "FAIL"]
    warned = [r for r in RESULTS if r[0] == "WARN"]
    print(
        f"\n{len(RESULTS) - len(failed) - len(warned)} passed, {len(warned)} warned, "
        f"{len(failed)} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
