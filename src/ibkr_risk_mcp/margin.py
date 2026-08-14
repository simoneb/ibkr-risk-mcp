"""Margin: IB's own answer to "what would this cost me", and the account
summary that says what there is to spend.

``whatif_order`` is the only tool in this server that touches the order path.
An order carrying ``whatIf=True`` is evaluated by IB's margin engine and
discarded — it is never routed, never acknowledged as live, and never appears
in the order book — but it is still the order channel, so it is gated behind
``IBKR_ENABLE_WHATIF`` rather than trusted to stay harmless by construction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from ib_async import Contract, Order

from . import contracts as C
from . import marketdata as MD
from .config import settings
from .connection import connection

log = logging.getLogger(__name__)

#: The figures IB returns on an OrderState, in the order that reads best.
_WHATIF_FIELDS = (
    "initMarginBefore",
    "initMarginAfter",
    "initMarginChange",
    "maintMarginBefore",
    "maintMarginAfter",
    "maintMarginChange",
    "equityWithLoanBefore",
    "equityWithLoanAfter",
    "equityWithLoanChange",
)

#: Account tags the summary always reports, whether or not IB volunteers them.
SUMMARY_TAGS = (
    "NetLiquidation",
    "EquityWithLoanValue",
    "FullInitMarginReq",
    "FullMaintMarginReq",
    "AvailableFunds",
    "ExcessLiquidity",
    "TotalCashValue",
    "BuyingPower",
    "Leverage",
)

#: IB's segment suffixes: securities and commodities. The distinction is the
#: whole point of reporting them — futures margin has to be met in the
#: commodities segment, and IB covers a shortfall there by sweeping cash from
#: securities, so the two falling together is a different event from either
#: falling alone.
SEGMENTS = {"": "total", "-S": "securities", "-C": "commodities"}


class WhatIfDisabled(RuntimeError):
    """whatif_order was called while the gate is closed."""


def require_whatif_enabled() -> None:
    if not settings.enable_whatif:
        raise WhatIfDisabled(
            "whatif_order is disabled because the server was not started with "
            "IBKR_ENABLE_WHATIF=true. Nothing was sent to IB — no order was created, "
            "previewed or routed."
        )


def _parse(value: Any) -> tuple[float | None, str | None]:
    """IB returns every margin figure as a string, and returns the sentinel
    1.7976931348623157E308 for "not applicable".

    A failure to parse is reported, not swallowed: a margin number that quietly
    became None reads downstream as zero impact, which is the one wrong answer
    that looks reassuring.
    """
    if value is None or value == "":
        return None, None
    text = str(value)
    try:
        f = float(text)
    except ValueError:
        return None, f"IB returned {text!r}, which is not a number"
    if abs(f) > 1e300:
        return None, None  # IB's explicit "no value" sentinel
    return f, None


def order_state_dict(state: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    problems: list[str] = []
    for field in _WHATIF_FIELDS:
        value, problem = _parse(getattr(state, field, None))
        out[field] = value
        if problem:
            problems.append(f"{field}: {problem}")

    commission, problem = _parse(getattr(state, "commission", None))
    if problem:
        problems.append(f"commission: {problem}")
    out["commission"] = commission
    out["commissionCurrency"] = getattr(state, "commissionCurrency", None) or None
    out["warningText"] = getattr(state, "warningText", None) or None
    out["status"] = getattr(state, "status", None) or None
    if problems:
        out["parseProblems"] = problems
    return out


#: ``transmit`` must be True on a what-if order. This is counter-intuitive
#: enough to be worth stating plainly: with transmit=False, TWS rejects the
#: request outright — error 321, "What-If order should have transmit flag set
#: to TRUE" — and, because it rejects rather than answers, the call simply
#: never comes back. Verified against live TWS; it is not a setting to soften
#: for safety.
#:
#: What keeps the order off the market is ``whatIf=True``, which makes IB run
#: it through the margin engine and discard it. ``transmit`` only decides
#: whether a *real* order would be staged or sent, and there is no real order
#: here.
_WHATIF_TRANSMIT = True


def _order_for(leg: C.Leg, contract: Contract, limit_price: float | None = None) -> Order:
    order = Order(
        action=leg.action,
        totalQuantity=abs(leg.quantity),
        orderType="LMT" if limit_price is not None else "MKT",
        whatIf=True,
        transmit=_WHATIF_TRANSMIT,
        tif="DAY",
    )
    if limit_price is not None:
        order.lmtPrice = round(limit_price, 4)
    return order


#: What to tell the user when IB gives nothing back. The same list applies
#: whether it answers with an empty state or does not answer at all.
_NO_ANSWER_CAUSES = (
    "Common causes: TWS has 'Read-Only API' enabled, which blocks what-if orders too; "
    "the account has no market data subscription for this contract, so IB cannot price "
    "it; the contract is not tradable from this account; or the market is closed and IB "
    "declined to price a market order."
)


class WhatIfSilent(RuntimeError):
    """IB accepted a what-if and never answered.

    Distinguished from an ordinary failure because it is not worth repeating:
    when TWS is dropping these, every subsequent leg will time out the same
    way, and a four-leg structure would sit there for a minute before saying
    so. The first one stops the rest.
    """


async def _whatif(contract: Contract, order: Order) -> dict[str, Any]:
    """One what-if round trip, bounded.

    ``whatIfOrderAsync`` does not always get a reply — TWS drops the request
    silently in several ordinary situations rather than answering with an error
    — and awaiting it unbounded hangs the whole tool call instead of reporting
    anything. Verified against live TWS, where a what-if on an unpriceable
    contract simply never came back.
    """
    ib = await connection.get()
    try:
        state = await asyncio.wait_for(
            ib.whatIfOrderAsync(contract, order), timeout=settings.whatif_timeout
        )
    except asyncio.TimeoutError:
        raise WhatIfSilent(
            f"IB did not answer the what-if within {settings.whatif_timeout:g}s. It was "
            f"sent and never replied to, rather than rejected. {_NO_ANSWER_CAUSES}"
        ) from None
    if state is None or not getattr(state, "initMarginAfter", ""):
        raise RuntimeError(f"IB returned no margin figures for this order. {_NO_ANSWER_CAUSES}")
    return order_state_dict(state)


async def _whatif_with_fallback(leg: C.Leg, contract: Contract) -> dict[str, Any]:
    """Market order first; a limit order at the model price if IB refuses it.

    Outside trading hours IB will not price a market order and answers with
    nothing at all, which is indistinguishable from a real rejection until the
    limit form succeeds.
    """
    try:
        return await _whatif(contract, _order_for(leg, contract))
    except WhatIfSilent:
        raise
    except Exception as exc:
        price = None
        if C.is_option(contract.secType):
            greeks, _ = await MD.model_greeks(contract, timeout=4.0)
            price = (greeks or {}).get("optPrice")
        else:
            price = await MD.spot_price(contract, timeout=3.0)
        if price is None or price <= 0:
            raise
        result = await _whatif(contract, _order_for(leg, contract, limit_price=price))
        result["pricedAs"] = f"LMT {round(price, 4)} (IB declined a market order: {exc})"
        return result


def _combo(pairs: Sequence[tuple[C.Leg, Contract]]) -> Contract | None:
    """A BAG contract over the legs, or None when they cannot share one.

    IB requires every leg of a combo to be on one exchange in one currency.
    Where they are not, there is no combo to evaluate and the cumulative view
    says so rather than quietly reporting the legs separately again.
    """
    from ib_async import ComboLeg

    exchanges = {c.exchange for _, c in pairs if c.exchange}
    currencies = {c.currency for _, c in pairs}
    if len(exchanges) != 1 or len(currencies) != 1:
        return None
    exchange = exchanges.pop()
    bag = Contract(
        secType="BAG",
        symbol=pairs[0][1].symbol,
        exchange=exchange,
        currency=currencies.pop(),
        comboLegs=[
            ComboLeg(
                conId=contract.conId,
                ratio=abs(leg.quantity),
                action=leg.action,
                exchange=contract.exchange or exchange,
            )
            for leg, contract in pairs
        ],
    )
    return bag


async def whatif_order(legs: Sequence[C.Leg]) -> dict[str, Any]:
    """IB's margin impact of a hypothetical structure, both leg by leg and
    cumulatively.

    Both views are returned because neither is the whole answer. The individual
    figures are what IB will charge for each leg on its own; SPAN then offsets
    the legs against each other and against what is already in the account, so
    the sum of the parts is not the effect of the whole — usually by a lot, and
    in the direction that makes a naive sum look scarier than reality.
    """
    require_whatif_enabled()
    pairs: list[tuple[C.Leg, Contract]] = []
    problems: list[dict[str, Any]] = []
    for leg in legs:
        try:
            pairs.append((leg, await MD.resolve_leg(leg)))
        except Exception as exc:
            problems.append({"leg": leg.model_dump(exclude_none=True), "error": str(exc)})

    per_leg: list[dict[str, Any]] = []
    silent: str | None = None
    for leg, contract in pairs:
        row: dict[str, Any] = {
            "leg": leg.model_dump(exclude_none=True),
            "contract": C.describe(contract),
        }
        if silent:
            row["error"] = f"Not attempted: {silent}"
            row["skipped"] = True
        else:
            try:
                row["whatIf"] = await _whatif_with_fallback(leg, contract)
            except WhatIfSilent as exc:
                # Once TWS is dropping these, it drops all of them. Stop rather
                # than spend the timeout again on every remaining leg.
                silent = str(exc)
                row["error"] = silent
            except Exception as exc:
                row["error"] = str(exc)
        per_leg.append(row)

    cumulative = (
        {"steps": [], "note": f"Not attempted: {silent}"}
        if silent
        else await _cumulative(pairs)
    )

    changes = [
        r["whatIf"]["initMarginChange"]
        for r in per_leg
        if r.get("whatIf") and r["whatIf"].get("initMarginChange") is not None
    ]
    sum_of_legs = round(sum(changes), 2) if changes else None
    combined = None
    if cumulative.get("steps"):
        last = cumulative["steps"][-1]
        combined = (last.get("whatIf") or {}).get("initMarginChange")

    return {
        "sentToIb": True,
        "routed": False,
        "perLeg": per_leg,
        "cumulative": cumulative,
        "offset": {
            "sumOfLegInitMarginChange": sum_of_legs,
            "combinedInitMarginChange": combined,
            "spanOffset": (
                round(sum_of_legs - combined, 2)
                if sum_of_legs is not None and combined is not None
                else None
            ),
            "note": "Positive spanOffset means the structure costs less as a whole than "
            "its legs do separately, which is SPAN recognising the hedge. Quote the "
            "combined figure, not the sum.",
        },
        "legProblems": problems,
    }


async def _cumulative(pairs: Sequence[tuple[C.Leg, Contract]]) -> dict[str, Any]:
    """Legs 1..k as a combo, for k = 1..n.

    IB's support for a what-if on an arbitrary multi-leg combo is uneven: it
    works for recognised strategies on one exchange and returns nothing for
    everything else. Each step is therefore attempted and reported on its own,
    so a partial answer is still an answer.
    """
    if not pairs:
        return {"steps": [], "note": "No legs resolved."}
    steps: list[dict[str, Any]] = []
    for k in range(1, len(pairs) + 1):
        subset = pairs[:k]
        entry: dict[str, Any] = {
            "legs": [leg.model_dump(exclude_none=True) for leg, _ in subset],
        }
        if k == 1:
            leg, contract = subset[0]
            try:
                entry["whatIf"] = await _whatif_with_fallback(leg, contract)
            except WhatIfSilent as exc:
                entry["error"] = str(exc)
                steps.append(entry)
                break
            except Exception as exc:
                entry["error"] = str(exc)
            steps.append(entry)
            continue
        bag = _combo(subset)
        if bag is None:
            entry["error"] = (
                "These legs are on different exchanges or currencies, so IB has no combo "
                "to evaluate. Read the per-leg figures instead and treat their sum as an "
                "upper bound."
            )
            steps.append(entry)
            continue
        order = Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            whatIf=True,
            transmit=_WHATIF_TRANSMIT,
            tif="DAY",
        )
        try:
            entry["whatIf"] = await _whatif(bag, order)
        except WhatIfSilent as exc:
            entry["error"] = str(exc)
            steps.append(entry)
            break
        except Exception as exc:
            entry["error"] = (
                f"IB would not evaluate this combo: {exc}. This is common — its what-if "
                "on arbitrary multi-leg combos is unreliable — and does not mean the "
                "structure is invalid."
            )
        steps.append(entry)
    return {
        "steps": steps,
        "note": "Each step adds one leg to the combo, so the sequence shows where the "
        "margin offset appears rather than only its total.",
    }


# --------------------------------------------------------------------------
# account summary
# --------------------------------------------------------------------------


async def margin_summary() -> dict[str, Any]:
    """The margin picture, split by segment.

    Every tag is reported three times where IB provides it: the account total,
    the securities segment (``-S``) and the commodities segment (``-C``).
    """
    ib = await connection.get()
    account = connection.require_account()
    values = MD.account_values(ib, account)

    summary: dict[str, dict[str, Any]] = {}
    for tag in SUMMARY_TAGS:
        row: dict[str, Any] = {}
        for suffix, name in SEGMENTS.items():
            entries = values.get(tag + suffix)
            if not entries:
                continue
            preferred = [e for e in entries if e["currency"] in ("BASE", "")] or entries
            row[name] = MD.num(preferred[0]["value"])
            if row.get("currency") is None and preferred[0]["currency"]:
                row["currency"] = preferred[0]["currency"]
        if row:
            summary[tag] = row

    cushion = MD.account_number(ib, account, "Cushion")
    net_liq = (summary.get("NetLiquidation") or {}).get("total")
    excess = (summary.get("ExcessLiquidity") or {}).get("total")

    return {
        "account": account,
        "accounts": list(ib.managedAccounts()),
        "summary": summary,
        "cushion": cushion,
        "excessLiquidityPctOfNetLiq": (
            round(excess / net_liq, 6) if net_liq and excess is not None else None
        ),
        "segments": {
            "note": "Futures margin is met in the commodities segment. When it is short, "
            "IB sweeps cash from the securities segment to cover it — so a fall that hits "
            "both segments at once is a different event from one that hits either alone, "
            "and the account total can look comfortable while a segment is not."
        },
        "raw": {
            tag: values[tag]
            for tag in sorted(values)
            if tag.split("-")[0] in SUMMARY_TAGS or tag in ("Cushion", "LookAheadExcessLiquidity")
        },
    }
