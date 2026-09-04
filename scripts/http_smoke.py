"""Drive the server over its HTTP transport, as a real client would.

Several modes, because there are several different questions.

    uv run python scripts/http_smoke.py

The default needs nothing but the code: it starts the server on a spare port,
speaks streamable HTTP to it, and checks that the transport comes up and every
tool registers. That runs on CI, where there is no TWS.

    uv run python scripts/http_smoke.py --live

Calls `check_connection` and then `get_position_greeks` against a real TWS,
**over HTTP**, and then calls `check_connection` again. The point is not the
numbers — `smoke_test.py` already checks those in-process. The point is the
event loop: ib_async drives its own socket and its market data callbacks on
whatever loop it finds, and under stdio that loop was one this code created for
itself. Under HTTP it is uvicorn's. `get_position_greeks` is the tool that
exercises the difference, because it is the one that opens market data
subscriptions, waits on them from inside a request handler, and cancels them
again.

    uv run python scripts/http_smoke.py --concurrent

The one stdio could never ask. A pipe serialises calls by construction: the
client sends one request and waits. HTTP does not, so for the first time two
tool calls can be inside `IBConnection` at once, sharing a single `IB` instance
and a single market data semaphore. That is correct by design — ib_async
multiplexes on request id and the semaphore is process-wide — but "correct by
design" and "exercised" are different claims, and only one of them is worth
deploying on.

    uv run python scripts/http_smoke.py --auth

The bearer-token boundary, end to end, behind a stub identity provider. The
unit tests in `tests/test_auth.py` cover what the verifier makes of a token;
this covers whether the verifier is *attached*, which is the failure that looks
exactly like success from the inside. Needs no TWS — every request it makes is
decided before a tool runs — so it belongs on CI.

    uv run python scripts/http_smoke.py --measure

Times every tool cold and warm, to set timeouts from numbers rather than from
guesses. See `docs/remote.md` for what the numbers came to and what was done
about them.

The server is spawned as a subprocess rather than assumed to be running, so
what gets tested is the entry point that will be in the service unit rather
than a fixture that resembles it. Its output is captured as well as echoed:
IB reports contention on the error channel, and a test that cannot see the
server's log cannot tell a run that stayed under the line limit from one that
was quietly refused half its subscriptions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession

from mcp_remote_auth.smoke import (
    AUTH_SUBJECT,
    NOTABLE_ERRORS,
    RESULTS,
    SERVER_LOG,
    _Done,
    check_auth,
    check_tools,
    client,
    free_port,
    mint,
    payload,
    pump,
    record,
    spawn_server,
    start_jwks_server,
    summary,
    wait_for_port,
)

ROOT = Path(__file__).resolve().parents[1]

#: What the server is supposed to register. Named rather than counted: a count
#: still passes when a tool is renamed and another is dropped.
EXPECTED_TOOLS = {
    "calibrate_vol_coord",
    "check_connection",
    "get_margin_summary",
    "get_position_greeks",
    "get_vol_surface",
    "stress_curve",
    "stress_portfolio",
    "stress_whatif",
    "whatif_order",
}

#: The shock axis the tool's own documentation calls "the usual ask". Used
#: rather than a token three points so the timings below are of a real request.
SHOCKS = [round(-0.30 + 0.01 * i, 2) for i in range(61)]

#: IB error codes worth naming if they show up. Registered with the shared
#: harness so its summary can label them. 322 is the one this whole
#: design is about: it is what TWS returns for market data requests beyond the
#: line limit, and it is what the semaphore in `marketdata.py` exists to
#: prevent. 10091 is an entitlement gap — the account's problem, not the
#: server's, and not evidence of contention.
NOTABLE_ERRORS.update({
    322: "market data line limit exceeded — the semaphore did not hold",
    420: "market data pacing violation",
    10091: "market data entitlement missing (account, not contention)",
    1100: "connection to TWS lost",
    1102: "connection restored",
    200: "no security definition for the request — a strike or expiry that is not listed",
})



async def check_live(session: ClientSession) -> None:
    conn = payload(await session.call_tool("check_connection", {}))
    state = conn.get("state")
    if state != "connected":
        record("FAIL", "check_connection", f"state={state}: {conn.get('hint', '')}")
        record("SKIP", "get_position_greeks", "no connection to fetch greeks over")
        return
    record("PASS", "check_connection", f"account {conn.get('accountInUse')}")

    greeks = payload(await session.call_tool("get_position_greeks", {}))
    if not greeks.get("success"):
        record("FAIL", "get_position_greeks", greeks.get("error", "no error reported"))
    else:
        priced = greeks.get("count", 0)
        missing = greeks.get("missing") or []
        others = greeks.get("nonOptionPositions", 0)
        if priced:
            record(
                "PASS",
                "get_position_greeks",
                f"{priced} option rows with greeks, {len(missing)} without, "
                f"{others} non-option positions",
            )
        elif missing:
            # Worth a WARN and not a FAIL: an entitlement gap is the account's
            # problem, and for this script's purpose the request still went out
            # over HTTP and IB still answered it, which is the thing being
            # tested. The reason is quoted so the two cannot be confused.
            record(
                "WARN",
                "get_position_greeks",
                f"0 of {len(missing)} options returned greeks — "
                f"first reason: {missing[0].get('reason')}",
            )
        else:
            record(
                "WARN",
                "get_position_greeks",
                f"no option positions on this account ({others} non-option)",
            )

    # The one that says whether the market data round trip left the socket
    # intact. A connection that dies during the greeks fetch still returns a
    # populated payload above, because every contract that never answered is
    # reported as missing data rather than as an error.
    again = payload(await session.call_tool("check_connection", {}))
    if again.get("state") == "connected":
        record("PASS", "connection survived", "still connected after the greeks fetch")
    else:
        record("FAIL", "connection survived", f"state={again.get('state')}")


# --------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------


def fingerprint(res: dict[str, Any]) -> dict[str, Any]:
    """The structural shape of a stress result, without its numbers.

    Deliberately not the P&L. Two runs a second apart against a live market
    price different spots and *should* differ in the last figures; comparing
    them for equality would fail on a working server every time the market
    moves. What must not differ is the shape — the same positions, the same
    axis, the same exclusions, the same reconciliation verdict. A portfolio
    load that lost half its rows to a rival caller shows up here and nowhere
    else.
    """
    curve = res.get("curve") or []
    return {
        "shocks": [round(float(p.get("shock", 0)), 6) for p in curve],
        "positions": len(res.get("positions") or []),
        "symbols": sorted({s for p in curve for s in (p.get("pnl_by_symbol") or {})}),
        "excluded": sorted(
            ",".join(g.get("symbols") or []) for g in (res.get("excluded") or [])
        ),
        "reconciled": res.get("reconciled"),
    }


def trough_pnl(res: dict[str, Any]) -> float | None:
    trough = res.get("trough") or {}
    value = trough.get("pnl")
    return float(value) if isinstance(value, (int, float)) else None


async def timed_call(
    port: int,
    tool: str,
    args: dict[str, Any],
    label: str,
    tool_timeout: float,
    token: str | None = None,
) -> tuple[str, float, dict[str, Any]]:
    """One tool call on its own client session, timed.

    A separate session per caller on purpose: two callers sharing one session
    would be testing the client's request multiplexing, and the question here
    is about the server's.
    """
    started = time.perf_counter()
    async with client(port, tool_timeout, token) as session:
        result = payload(await session.call_tool(tool, args, read_timeout_seconds=timedelta(seconds=tool_timeout)))
    return label, time.perf_counter() - started, result


async def after(delay: float, coro_factory) -> tuple[str, float, dict[str, Any]]:
    await asyncio.sleep(delay)
    return await coro_factory()


async def check_concurrency(
    port: int, tool_timeout: float, stagger: float, token: str | None = None
) -> None:
    stress_args = {"shocks": SHOCKS}

    # A baseline first, alone, so "slower under contention" has something to be
    # slower than.
    _, solo_secs, solo = await timed_call(
        port, "stress_portfolio", stress_args, "solo", tool_timeout, token
    )
    if not solo.get("success"):
        record("FAIL", "stress_portfolio (solo)", solo.get("error", "no error reported"))
        record("SKIP", "two overlapping stress runs", "no working baseline to compare")
        return
    record(
        "PASS",
        "stress_portfolio (solo)",
        f"{solo_secs:.1f}s, {len(solo.get('positions') or [])} positions, "
        f"{len(SHOCKS)} shocks, reconciled={solo.get('reconciled')}",
    )
    base = fingerprint(solo)

    # Two of the same tool, overlapping. The stagger is the point: started at
    # exactly the same instant they would both be loading holdings while
    # neither is yet in the market data phase, which is the easy case.
    marker = len(SERVER_LOG)
    (label_a, secs_a, res_a), (label_b, secs_b, res_b) = await asyncio.gather(
        timed_call(port, "stress_portfolio", stress_args, "A", tool_timeout, token),
        after(
            stagger,
            lambda: timed_call(port, "stress_portfolio", stress_args, "B", tool_timeout, token),
        ),
    )
    during = SERVER_LOG[marker:]

    failed = [
        (lab, res) for lab, res in ((label_a, res_a), (label_b, res_b)) if not res.get("success")
    ]
    if failed:
        for lab, res in failed:
            record("FAIL", f"overlapping stress {lab}", res.get("error", "no error reported"))
        return

    record(
        "PASS",
        "two overlapping stress runs",
        f"A {secs_a:.1f}s, B {secs_b:.1f}s (solo {solo_secs:.1f}s), both succeeded",
    )

    for lab, res in ((label_a, res_a), (label_b, res_b)):
        shape = fingerprint(res)
        if shape != base:
            differing = [k for k in base if shape.get(k) != base.get(k)]
            record("FAIL", f"shape of run {lab}", f"differs from the solo baseline in {differing}")
        else:
            record("PASS", f"shape of run {lab}", "identical to the solo baseline")

    # The numbers are reported, not asserted. Against a live market they should
    # move a little between runs; a large gap is worth a human look, and zero is
    # not evidence of anything either way.
    troughs = {lab: trough_pnl(res) for lab, res in (("solo", solo), (label_a, res_a), (label_b, res_b))}
    if all(v is not None for v in troughs.values()):
        spread = max(troughs.values()) - min(troughs.values())  # type: ignore[type-var]
        record(
            "INFO",
            "trough drift across the three runs",
            ", ".join(f"{k}={v:,.0f}" for k, v in troughs.items()) + f" (spread {spread:,.0f})",
        )

    # And what TWS made of being asked twice at once.
    codes: dict[int, int] = {}
    for line in during:
        for match in re.finditer(r"\bError (\d{3,5})\b", line):
            code = int(match.group(1))
            codes[code] = codes.get(code, 0) + 1
    if 322 in codes:
        record(
            "FAIL",
            "market data line limit",
            f"{codes[322]}x error 322 during the overlap — {NOTABLE_ERRORS[322]}",
        )
    elif codes:
        record(
            "PASS",
            "market data line limit",
            "no error 322; other codes seen: "
            + ", ".join(f"{c}x{n}" for c, n in sorted(codes.items())),
        )
    else:
        record("PASS", "market data line limit", "no IB errors at all during the overlap")


def pick_surface_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which underlying to build a surface for, while a stress run is going on.

    Two properties matter and neither is served by taking the first row. It has
    to be the *same* choice on every run against the same book, or a slow
    surface is indistinguishable from an unlucky one; and it should be an
    equity name, because that is what this server is for and what the chain
    around it will actually look like. The first live run of this test drew a
    CAD FX option, which is a legitimate holding and a poor probe.

    Falls back to the whole set when the book holds no equity options at all,
    rather than skipping: contention is still worth measuring on whatever is
    there.
    """
    equity = [r for r in rows if r.get("riskGroup") == "equity"]
    pool = equity or rows
    return sorted(
        pool, key=lambda r: (str(r.get("symbol") or ""), str(r.get("lastTradeDate") or ""))
    )[0]


async def check_mixed(
    port: int, tool_timeout: float, stagger: float, token: str | None = None
) -> None:
    """A stress run overlapping a volatility surface — the harsher case.

    Two stress runs want the same contracts, so the second may be riding the
    first's contract details cache. A surface request is for strikes nobody
    holds, so the two compete for market data lines properly rather than
    sharing them.
    """
    async with client(port, tool_timeout, token) as session:
        greeks = payload(await session.call_tool("get_position_greeks", {}))
    rows = greeks.get("data") or []
    if not rows:
        record("SKIP", "stress + vol surface", "no option positions to pick an underlying from")
        return
    probe = pick_surface_probe(rows)
    underlying, expiry = probe.get("symbol"), probe.get("lastTradeDate")
    if not underlying or not expiry:
        record("SKIP", "stress + vol surface", f"could not read an underlying from {probe!r}")
        return

    marker = len(SERVER_LOG)
    (_, stress_secs, stress_res), (_, surface_secs, surface_res) = await asyncio.gather(
        timed_call(port, "stress_portfolio", {"shocks": SHOCKS}, "stress", tool_timeout, token),
        after(
            stagger,
            lambda: timed_call(
                port,
                "get_vol_surface",
                {"underlying": underlying, "expiries": [expiry], "rights": ["P", "C"]},
                "surface",
                tool_timeout,
                token,
            ),
        ),
    )
    during = SERVER_LOG[marker:]

    if not stress_res.get("success"):
        record("FAIL", "stress under surface load", stress_res.get("error", "no error"))
    else:
        record("PASS", "stress under surface load", f"{stress_secs:.1f}s, still reconciled="
               f"{stress_res.get('reconciled')}")
    attempted = sum(len(re.findall(r"\bError \d{3,5}\b", line)) for line in during)
    if not surface_res.get("success"):
        record("FAIL", "vol surface under stress load", surface_res.get("error", "no error"))
    else:
        points = surface_res.get("count")
        if points is None:
            points = len(surface_res.get("data") or [])
        where = f"{underlying} {expiry} ({probe.get('riskGroup')})"
        if points:
            record(
                "PASS",
                "vol surface under stress load",
                f"{surface_secs:.1f}s, {points} points on {where}",
            )
        else:
            # A surface of nothing is not a passing surface, even when the tool
            # reports success — and the difference between "the chain is not
            # entitled" and "the other caller took every line" is the whole
            # question this test exists to answer. Both leave an empty result;
            # only one leaves error 322.
            record(
                "WARN",
                "vol surface under stress load",
                f"{surface_secs:.1f}s but 0 points on {where} — "
                f"{len(surface_res.get('missing') or [])} strikes reported missing; "
                "an entitlement gap on this chain, not the transport",
            )

    if any("Error 322" in line for line in during):
        record("FAIL", "market data lines under mixed load", "error 322 during the overlap")
    elif attempted:
        record(
            "PASS",
            "market data lines under mixed load",
            f"no error 322 across {attempted} subscription responses during the overlap",
        )
    else:
        # No 322 is only reassuring if something was actually asked for.
        record(
            "WARN",
            "market data lines under mixed load",
            "no error 322, but IB answered nothing during the overlap — this proves less "
            "than it looks",
        )


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

#: The widest axis anyone would plausibly ask stress_curve for: -40% to +20% in
#: 1% steps, across both of the volatility scenarios it draws by default.
WIDE_SHOCKS = [round(-0.40 + 0.01 * i, 2) for i in range(61)]


async def check_measure(port: int, tool_timeout: float, token: str | None = None) -> None:
    """Time every tool a client would wait on, cold and warm.

    Cold matters more than warm and is measured first: a client's very first
    call pays for the IB connect, the account subscription and an empty
    contract details cache, all inside one request. That is the number a
    timeout has to clear, and it is the one nobody sees during development
    because by then the server has been up for an hour.
    """
    timings: list[tuple[str, float, str]] = []

    async def timed(session: ClientSession, label: str, tool: str, args: dict[str, Any]) -> dict:
        started = time.perf_counter()
        res = payload(
            await session.call_tool(
                tool, args, read_timeout_seconds=timedelta(seconds=tool_timeout)
            )
        )
        elapsed = time.perf_counter() - started
        note = "" if res.get("success", True) else f"FAILED: {res.get('error', '')}"
        timings.append((label, elapsed, note))
        record("INFO" if not note else "FAIL", label, f"{elapsed:.1f}s {note}".strip())
        return res

    async with client(port, tool_timeout, token) as session:
        # Cold, in the order a client would hit them.
        await timed(session, "check_connection (cold — includes IB connect)", "check_connection", {})
        await timed(session, "get_position_greeks (cold)", "get_position_greeks", {})
        await timed(
            session, "stress_portfolio (cold-ish, 61 shocks)", "stress_portfolio", {"shocks": SHOCKS}
        )
        await timed(
            session,
            "stress_curve (widest: 61 shocks x 2 vol scenarios)",
            "stress_curve",
            {"shocks": WIDE_SHOCKS},
        )
        await timed(session, "get_margin_summary", "get_margin_summary", {})

        greeks = payload(await session.call_tool("get_position_greeks", {}))
        rows = greeks.get("data") or []
        if rows:
            probe = pick_surface_probe(rows)
            await timed(
                session,
                f"get_vol_surface ({probe.get('symbol')}, 1 expiry, both rights)",
                "get_vol_surface",
                {
                    "underlying": probe.get("symbol"),
                    "expiries": [probe.get("lastTradeDate")],
                    "rights": ["P", "C"],
                },
            )

        # Warm, to separate the fixed cost from the per-call one.
        await timed(session, "get_position_greeks (warm)", "get_position_greeks", {})
        await timed(
            session, "stress_portfolio (warm, 61 shocks)", "stress_portfolio", {"shocks": SHOCKS}
        )

    worst = max((t for _, t, _ in timings), default=0.0)
    record(
        "PASS" if worst < 60 else "WARN",
        "measured ceiling",
        f"slowest single call {worst:.1f}s. Client default read gap is 300s "
        f"({worst / 300:.0%} of it); uvicorn imposes no request timeout of its own",
    )


# --------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    port = free_port()
    auth_key = auth_issuer = auth_resource = None
    auth_env: dict[str, str] = {}
    if args.auth:
        auth_key, jwks_port, _server = start_jwks_server()
        auth_issuer = f"http://127.0.0.1:{jwks_port}/"
        auth_resource = f"http://127.0.0.1:{port}/mcp"
        auth_env = {
            "IBKR_MCP_AUTH": "true",
            "IBKR_MCP_AUTH_ISSUER": auth_issuer,
            "IBKR_MCP_AUTH_JWKS_URL": f"{auth_issuer}.well-known/jwks.json",
            "IBKR_MCP_RESOURCE_URL": auth_resource,
            "IBKR_MCP_AUTH_AUDIENCE": auth_resource,
            "IBKR_MCP_ALLOWED_SUBJECTS": AUTH_SUBJECT,
        }
    proc = await spawn_server(
        ROOT,
        "from ibkr_risk_mcp.server import main; main()",
        port,
        "IBKR_MCP_",
        extra_env=auth_env,
    )
    pumping = asyncio.create_task(pump(proc, echo=args.server_log))
    try:
        await wait_for_port(port, proc, args.startup_timeout)
        record("PASS", "server listening", f"127.0.0.1:{port}")

        session_token = None
        if args.auth:
            await check_auth(port, auth_key, auth_issuer, auth_resource)
            # Everything after this runs authenticated, so the live modes
            # exercise the transport as it will actually be deployed rather
            # than as a version of it with the boundary switched off. An hour
            # of validity: long enough that a full concurrency and measurement
            # pass cannot expire mid-run, short enough to still be a token.
            session_token = mint(
                auth_key, sub=AUTH_SUBJECT, aud=auth_resource, iss=auth_issuer, ttl=3600
            )
            if not (args.live or args.concurrent or args.measure):
                raise _Done()

        async with client(port, args.tool_timeout, session_token) as session:
            record(
                "PASS",
                "initialize",
                "session established over streamable HTTP"
                + (", bearer token accepted" if session_token else ""),
            )
            await check_tools(session, EXPECTED_TOOLS)
            if args.live or args.concurrent:
                await check_live(session)
        if args.concurrent:
            await check_concurrency(port, args.tool_timeout, args.stagger, session_token)
            await check_mixed(port, args.tool_timeout, args.stagger, session_token)
        if args.measure:
            await check_measure(port, args.tool_timeout, session_token)
    except _Done:
        pass
    except Exception as exc:
        record("FAIL", "transport", f"{type(exc).__name__}: {exc}")
    finally:
        if proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=10)
        if proc.returncode is None:  # pragma: no cover - only a wedged server
            proc.kill()
            await proc.wait()
        pumping.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pumping

    return summary()


def cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--live",
        action="store_true",
        help="also call check_connection and get_position_greeks against a real TWS",
    )
    ap.add_argument(
        "--concurrent",
        action="store_true",
        help="also run overlapping tool calls against a real TWS (implies --live)",
    )
    ap.add_argument(
        "--auth",
        action="store_true",
        help="start the server behind a stub identity provider and check the bearer-token "
        "boundary end to end. Needs no TWS: every request is decided before a tool runs",
    )
    ap.add_argument(
        "--measure",
        action="store_true",
        help="time every tool cold and warm against a real TWS, to set the timeouts from "
        "numbers rather than from guesses",
    )
    ap.add_argument(
        "--stagger",
        type=float,
        default=1.0,
        help="seconds between the two overlapping calls, so the second lands while the "
        "first is in its market data phase rather than beside it (default: 1.0)",
    )
    # Generous, because it is an upper bound rather than a wait: the port is
    # polled and the run continues the moment it answers. A warm checkout gets
    # there in under three seconds; a cold one importing numpy, scipy and
    # ib_async for the first time — a CI runner, every time — has been seen to
    # take an order of magnitude longer.
    ap.add_argument("--startup-timeout", type=float, default=90.0)
    ap.add_argument(
        "--tool-timeout",
        type=float,
        default=900.0,
        help="client-side ceiling on one tool call. Deliberately far above anything "
        "measured, so that a slow answer is reported as a slow answer rather than as a "
        "timeout (default: 900)",
    )
    ap.add_argument(
        "--server-log",
        action="store_true",
        help="echo the server's own output as it arrives; it is captured either way",
    )
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(cli())
