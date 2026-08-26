"""Everything that talks to IB's market data feed: model greeks, the implied
volatility surface, spot prices, and the portfolio snapshot the stress engine
runs on.

Two constraints shape this module.

**Rate limits.** IB allows on the order of 50 concurrent market data lines and
100 messages a second. A surface request over six expiries and twenty strikes
is 120 subscriptions; fired at once, TWS answers the first fifty and returns
error 322 for the rest. Every subscription here goes through one semaphore and
is cancelled the moment its data lands, so the line count is bounded by the
semaphore rather than by the size of the request.

**Missing data is data.** ``modelGreeks`` arrives asynchronously and is None
until it does — and stays None for a contract with no market data subscription,
outside trading hours, or with the wrong market data type set. Every function
here reports the contracts it could not get rather than dropping them, because
a surface silently missing its left wing looks exactly like a surface that has
one.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ib_async import Contract, ContractDetails, PortfolioItem, Ticker

from . import contracts as C
from .config import settings
from .connection import IBUnavailable, connection

log = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def market_data_semaphore() -> asyncio.Semaphore:
    """One semaphore for the whole process, created on the running loop rather
    than at import time so it binds to the loop the server actually uses."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.max_market_data_lines)
    return _semaphore


def num(value: Any) -> float | None:
    """IB reports absent numbers as None, as NaN, and as the sentinel
    -1.7976931348623157e+308. All three mean "no value" and none of them should
    reach a caller as a float."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or abs(f) > 1e300:
        return None
    return f


# --------------------------------------------------------------------------
# contract resolution
# --------------------------------------------------------------------------

_details_cache: dict[int, ContractDetails] = {}


async def details_for_conid(conid: int) -> ContractDetails | None:
    """Full contract details by conId, cached for the process lifetime.

    This is where ``underConId`` comes from — the only trustworthy answer to
    "which future is this option written on". Contract definitions do not
    change, so caching them is free.
    """
    if conid in _details_cache:
        return _details_cache[conid]
    ib = await connection.get()
    found = await ib.reqContractDetailsAsync(Contract(conId=conid))
    if not found:
        return None
    _remember(found)
    return found[0]


def _remember(found: Sequence[ContractDetails]) -> list[ContractDetails]:
    """Cache details and hand IB's authoritative expiration to the contracts
    module, so every later date question is answered from IB's own field rather
    than from the class-and-date heuristic."""
    for detail in found:
        conid = detail.contract.conId
        if conid:
            _details_cache[conid] = detail
        C.remember_details(detail)
    return list(found)


async def qualified(contract: Contract) -> Contract:
    """A contract IB has confirmed, with conId and exchange filled in.

    ``reqMktData`` on an unqualified contract returns nothing at all — no error,
    no ticks — which is the single most common reason a surface comes back
    empty.
    """
    if contract.conId and contract.exchange:
        return contract
    ib = await connection.get()
    resolved = await ib.qualifyContractsAsync(contract)
    if not resolved:
        raise ValueError(f"IB does not recognise the contract {contract}")
    return resolved[0]


async def resolve_underlying(
    symbol: str, sec_type: str | None = None
) -> tuple[Contract, list[str]]:
    """Find the underlying instrument for a symbol, and say what else it could
    have been.

    Returns the chosen contract and a list of the other security types the
    symbol also resolves to. That second value is not decoration: **ES is both
    the E-mini S&P 500 future and Eversource Energy on NYSE**, and searching
    stocks first returns a utility with a 75-dollar strike ladder in answer to
    a question about index futures. Confirmed live — it is exactly what
    happened here before this returned the alternatives.

    With no ``sec_type`` the search still runs STK, then IND, then FUT, since
    that is right far more often than not. The difference is that a collision
    is now reported rather than resolved in silence.
    """
    ib = await connection.get()
    order: list[tuple[str, Contract]]
    if sec_type:
        probe = Contract(symbol=symbol, secType=sec_type, currency="USD")
        if sec_type == "STK":
            probe.exchange = "SMART"
        order = [(sec_type, probe)]
    else:
        order = [
            ("STK", Contract(symbol=symbol, secType="STK", exchange="SMART", currency="USD")),
            ("IND", Contract(symbol=symbol, secType="IND", currency="USD")),
            ("FUT", Contract(symbol=symbol, secType="FUT", currency="USD")),
        ]

    matches: list[tuple[str, ContractDetails]] = []
    for kind, candidate in order:
        try:
            found = _remember(await ib.reqContractDetailsAsync(candidate))
        except Exception:  # IB raises on a malformed request; try the next form
            continue
        if not found:
            continue
        if kind == "FUT":
            # Front contract by expiry, not the order IB happened to return.
            found.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
        matches.append((kind, found[0]))

    if not matches:
        raise ValueError(
            f"Could not resolve {symbol!r} as an underlying. Pass sec_type "
            "('STK', 'IND', 'FUT') to disambiguate."
        )
    chosen_kind, chosen = matches[0]
    alternatives = [
        f"{kind} ({d.contract.localSymbol or d.contract.symbol}"
        f"{', ' + d.contract.primaryExchange if d.contract.primaryExchange else ''})"
        for kind, d in matches[1:]
    ]
    if alternatives:
        log.warning(
            "%s resolved as %s; it also matches %s",
            symbol,
            chosen_kind,
            ", ".join(alternatives),
        )
    return chosen.contract, alternatives


async def chain_details(
    symbol: str,
    expiry: str,
    sec_type: str = "OPT",
    exchange: str | None = None,
    currency: str = "USD",
    trading_class: str | None = None,
    rights: Sequence[str] = ("P",),
) -> list[ContractDetails]:
    """Every option contract on ``symbol`` for one expiry, straight from
    ``reqContractDetails``.

    Enumerating the chain this way rather than from ``reqSecDefOptParams`` is
    deliberate: the details carry ``underConId``, so the "which quarterly is
    this written on" question is answered by IB instead of inferred from the
    date. It also returns contracts already qualified, ready for
    ``reqMktData``.

    ``expiry`` may be given as either the last trading date or the settlement
    date; both forms are matched, which matters for AM-settled expiries where
    the two differ by a day.
    """
    ib = await connection.get()
    out: list[ContractDetails] = []
    for right in rights:
        probe = Contract(
            symbol=symbol,
            secType=sec_type,
            currency=currency,
            right=right,
            lastTradeDateOrContractMonth=_ib_date(expiry),
        )
        if exchange:
            probe.exchange = exchange
        if trading_class:
            probe.tradingClass = trading_class
        try:
            out.extend(_remember(await ib.reqContractDetailsAsync(probe)))
        except Exception as exc:
            log.warning("chain lookup failed for %s %s %s: %s", symbol, expiry, right, exc)
    if out:
        return out
    # The requested date was a settlement date that does not exist as a last
    # trading day: ask for the whole month and match on settlement instead.
    wanted = _ib_date(expiry)
    month_probe_rights = rights or ("P",)
    for right in month_probe_rights:
        probe = Contract(
            symbol=symbol,
            secType=sec_type,
            currency=currency,
            right=right,
            lastTradeDateOrContractMonth=wanted[:6],
        )
        if exchange:
            probe.exchange = exchange
        try:
            found = _remember(await ib.reqContractDetailsAsync(probe))
        except Exception:
            continue
        out.extend(
            d
            for d in found
            if C.settlement_date(d.contract).strftime("%Y%m%d") == wanted
        )
    return out


def _ib_date(expiry: str) -> str:
    return (expiry or "").strip().replace("-", "")


# --------------------------------------------------------------------------
# streaming quotes
# --------------------------------------------------------------------------


#: Generic tick 221 is IB's **mark price** — the value it computes for a
#: contract that has not traded, from the bid/ask and its own model. On an
#: illiquid option or a quiet future it is the only price that arrives at all,
#: so asking for it is the difference between a priced contract and an empty
#: ticker.
MARK_PRICE_TICK = "221"


#: IB error codes that mean "this request will never be answered". Waiting out
#: a timeout on any of them is pure delay: the reply has already arrived, and
#: it was a refusal.
#:
#: 354 no market data subscription; 10091 and 10089 need a subscription for the
#: API specifically; 200 no security definition; 10168/10197 no data during
#: competing-session or connectivity loss; 322 too many requests.
_FATAL_MKTDATA_ERRORS = {354, 200, 322, 10089, 10091, 10167, 10168, 10197}


async def _await_ticker(
    contract: Contract,
    ready,
    timeout: float,
    tick_list: str = "",
) -> tuple[Ticker | None, str | None]:
    """Subscribe, wait for ``ready(ticker)``, cancel. Returns the ticker or a
    reason it never became ready.

    Two things keep this quick. IB answers a market data request either within
    a second or never, so the timeout is short. And when it is never, IB
    usually says so immediately with an error — 354 for a missing subscription
    is the common one — which is watched for here and ends the wait on the
    spot. Without that, a book with no entitlement pays the full timeout on
    every single contract, which is how a "quick" check turns into minutes.

    The cancel is in a ``finally`` because an abandoned subscription keeps its
    market data line for the life of the connection, and fifty of those is the
    whole allowance.
    """
    ib = await connection.get()
    refusal: list[str] = []

    def on_error(reqId: int, code: int, msg: str, errContract: Any) -> None:
        if code in _FATAL_MKTDATA_ERRORS and getattr(errContract, "conId", None) == contract.conId:
            text = f"IB refused the request ({code}): {msg.split('.')[0]}"
            if code in (354, 10089, 10091):
                text += (
                    ". This is an account entitlement, not a fault: the instrument needs a "
                    "market data subscription. Delayed data often covers it and does carry "
                    "model greeks — try IBKR_MARKET_DATA_TYPE=3"
                )
            refusal.append(text)

    async with market_data_semaphore():
        ib.errorEvent += on_error
        ticker = ib.reqMktData(contract, tick_list, False, False)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            while loop.time() < deadline:
                if ready(ticker):
                    return ticker, None
                if refusal:
                    return None, refusal[0]
                await asyncio.sleep(0.05)
        finally:
            ib.cancelMktData(contract)
            ib.errorEvent -= on_error
    return None, refusal[0] if refusal else _no_data_reason(ticker)


def _no_data_reason(ticker: Ticker | None) -> str:
    """Say why nothing arrived, in terms of something the user can change."""
    if ticker is None:
        return "no ticker was created for the contract"
    if num(ticker.last) is None and num(ticker.close) is None and num(ticker.bid) is None:
        return (
            "no market data of any kind arrived — most likely the account has no data "
            "subscription for this instrument, or the market is closed. Try "
            "IBKR_MARKET_DATA_TYPE=3: delayed data does carry IB's model greeks, and on "
            "an unsubscribed account it is often the only thing that returns anything"
        )
    return (
        "prices arrived but IB published no model greeks — usual on an illiquid strike "
        "with no two-sided market, since IB will not imply a volatility from nothing"
    )


async def spot_price(contract: Contract, timeout: float = 3.0) -> float | None:
    """A usable price for a contract: mark, then last, then mid, then close.

    The mark comes first because it is the one IB will produce for something
    that has not traded — a quiet future, an illiquid strike — where last is
    empty and the close is stale. Requesting it costs one generic tick and
    turns "no price at all" into a number in the cases that matter.

    Only used to centre a default strike band and to value non-option legs.
    Every price that reaches an option calculation comes from
    ``modelGreeks.undPrice`` instead, which is the forward IB's own model used
    and therefore the one its greeks are consistent with.
    """

    def ready(t: Ticker) -> bool:
        return (
            num(t.markPrice) is not None
            or num(t.last) is not None
            or (num(t.bid) is not None and num(t.ask) is not None)
        )

    ticker, _ = await _await_ticker(contract, ready, timeout, tick_list=MARK_PRICE_TICK)
    if ticker is None:
        return None
    for value in (
        num(getattr(ticker, "markPrice", None)),
        num(ticker.last),
        num(ticker.marketPrice()),
        num(ticker.close),
    ):
        if value is not None and value > 0:
            return value
    bid, ask = num(ticker.bid), num(ticker.ask)
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2
    return None


@dataclass
class GreeksRow:
    contract: Contract
    greeks: dict[str, Any] | None
    error: str | None = None


async def model_greeks(
    contract: Contract, timeout: float | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """IB's own model output for one option: implied volatility, the four
    greeks, the model price, the underlying forward and the present value of
    dividends.

    These are IB's numbers, not this server's. That is the point — a curve
    rebuilt from IB's inputs can be compared to what Risk Navigator shows;
    one rebuilt from locally implied volatilities cannot.
    """
    timeout = timeout if timeout is not None else settings.greeks_timeout

    # Both fields, not just the volatility. IB fills the model computation in
    # over successive ticks, and a snapshot taken the moment impliedVol lands
    # often still has undPrice unset — measured live, where a sizeable minority
    # came back with a volatility and no underlying. Without the forward there
    # is nothing to reprice against, so such a position is held flat across
    # every shock and the curve quietly loses it.
    def ready(t: Ticker) -> bool:
        if t.modelGreeks is None:
            return False
        return (
            num(t.modelGreeks.impliedVol) is not None
            and num(t.modelGreeks.undPrice) is not None
        )

    ticker, reason = await _await_ticker(contract, ready, timeout)
    if ticker is None or ticker.modelGreeks is None:
        return None, reason
    g = ticker.modelGreeks
    return {
        "impliedVol": num(g.impliedVol),
        "delta": num(g.delta),
        "gamma": num(g.gamma),
        "vega": num(g.vega),
        "theta": num(g.theta),
        "optPrice": num(g.optPrice),
        "pvDividend": num(g.pvDividend),
        "undPrice": num(g.undPrice),
    }, None


async def model_greeks_batch(
    contract_list: Sequence[Contract], timeout: float | None = None, retries: int = 1
) -> list[GreeksRow]:
    """Greeks for many contracts at once, bounded by the market data
    semaphore. Order matches the input.

    Stragglers get a second attempt. On a large book the first pass queues
    behind the semaphore and some contracts run out their timeout waiting
    rather than because IB has nothing to say — measured on a live account with
    a few dozen option positions, where a first pass left a handful short and a
    retry got them. Retrying only the ones that came back empty costs a few seconds and
    is the difference between a curve that carries the whole portfolio and one
    quietly missing a quarter of it.
    """

    async def one(c: Contract, t: float | None) -> GreeksRow:
        try:
            greeks, err = await model_greeks(c, t)
        except IBUnavailable:
            raise
        except Exception as exc:  # a bad contract must not sink the batch
            return GreeksRow(c, None, f"{type(exc).__name__}: {exc}")
        return GreeksRow(c, greeks, err)

    rows = list(await asyncio.gather(*(one(c, timeout) for c in contract_list)))
    for _ in range(max(retries, 0)):
        pending = [i for i, row in enumerate(rows) if row.greeks is None]
        if not pending:
            break
        base = timeout if timeout is not None else settings.greeks_timeout
        retried = await asyncio.gather(*(one(rows[i].contract, base * 1.5) for i in pending))
        for i, row in zip(pending, retried):
            if row.greeks is not None:
                rows[i] = row
    return rows


# --------------------------------------------------------------------------
# portfolio snapshot
# --------------------------------------------------------------------------


@dataclass
class Holding:
    """One position, with everything the stress engine needs attached.

    ``marketValue`` is IB's, never recomputed. Bonds are quoted as a percentage
    of nominal, so ``position * marketPrice * multiplier`` overstates them a
    hundredfold; taking IB's figure sidesteps the trap entirely and the
    reconciliation check at zero shock is what proves it worked.
    """

    contract: Contract
    position: float
    market_price: float | None
    market_value: float | None
    average_cost: float | None
    unrealized_pnl: float | None
    asset_class: str
    multiplier: float
    greeks: dict[str, Any] | None = None
    greeks_error: str | None = None
    und_conid: int | None = None
    und_symbol: str | None = None

    @property
    def is_option(self) -> bool:
        return self.asset_class == "option"

    def describe(self) -> dict[str, Any]:
        out = C.describe(self.contract)
        out.update(
            {
                "position": self.position,
                "assetClass": self.asset_class,
                "marketPrice": self.market_price,
                "marketValue": self.market_value,
                "averageCost": self.average_cost,
                "unrealizedPnl": self.unrealized_pnl,
                "undConId": self.und_conid,
                "undSymbol": self.und_symbol,
            }
        )
        return out


def _portfolio_items(ib, account: str) -> list[PortfolioItem]:
    """Positions for one account, filtered here as well as at the API.

    ``ib.portfolio("")`` returns every managed account's positions, so the
    account is never allowed to be empty by the time it reaches this. The
    second filter is belt and braces against older ib_async versions whose
    ``portfolio()`` takes no argument at all.
    """
    try:
        items = list(ib.portfolio(account))
    except TypeError:  # older ib_async takes no account argument
        items = list(ib.portfolio())
    return [i for i in items if not i.account or i.account == account]


async def load_holdings(with_greeks: bool = True, symbol: str | None = None) -> list[Holding]:
    """The current portfolio as :class:`Holding` rows, greeks attached.

    Positions with zero quantity are dropped — IB keeps reporting a position
    the rest of the session after it is closed, with quantity 0 and a stale
    price, and carrying those into a stress run adds noise with no exposure.
    """
    ib = await connection.get()
    account = connection.require_account()
    items = [i for i in _portfolio_items(ib, account) if i.position]

    holdings: list[Holding] = []
    for item in items:
        contract = item.contract
        klass = C.asset_class(contract.secType)
        if symbol and contract.symbol.upper() != symbol.upper():
            continue
        holdings.append(
            Holding(
                contract=contract,
                position=float(item.position),
                market_price=num(item.marketPrice),
                market_value=num(item.marketValue),
                average_cost=num(item.averageCost),
                unrealized_pnl=num(item.unrealizedPNL),
                asset_class=klass,
                multiplier=C.contract_multiplier(contract),
            )
        )

    options = [h for h in holdings if h.is_option]
    if options:
        await asyncio.gather(*(_attach_underlying(h) for h in options))
    if options and with_greeks:
        rows = await model_greeks_batch([h.contract for h in options])
        for holding, row in zip(options, rows):
            holding.greeks = row.greeks
            holding.greeks_error = row.error
        portfolio_prices = underlying_prices_from_portfolio(holdings)
        await _backfill_underlying_price(options, portfolio_prices)
        await _imply_missing_greeks(options, portfolio_prices)
    return holdings


def underlying_prices_from_portfolio(holdings: Sequence["Holding"]) -> dict[int, float]:
    """Prices for anything the account already holds, keyed by conId.

    Every position IB reports carries its own mark on the portfolio update, and
    that arrives over the **account** channel, which no market data entitlement
    gates. So a book holding GOOGL stock next to a GOOGL option already knows
    what GOOGL is worth, and asking again over ``reqMktData`` is both slower and,
    on an account without the API subscription, refused outright — error 10089
    on the very underlying whose price is sitting in the snapshot.

    Measured on a live account: two long put positions were being held flat
    across every shock, worth 2,826 at a 15% fall and 4,414 at 20%, purely
    because the spot needed to imply their volatility was fetched instead of
    read. This is the difference between pricing them and dropping them.

    A futures option gets the same benefit when the account also holds the
    future, since ``underConId`` points straight at it.
    """
    prices: dict[int, float] = {}
    for holding in holdings:
        conid = getattr(holding.contract, "conId", None)
        if not conid or holding.is_option:
            continue
        price = holding.market_price
        if price is None and holding.market_value and holding.position:
            denominator = holding.position * holding.multiplier
            price = holding.market_value / denominator if denominator else None
        if price and price > 0:
            prices[int(conid)] = float(price)
    return prices


#: Set to False the first time ``reqCalcImpliedVolatility`` misbehaves. On the
#: TWS build measured here it answers with error 320 and drops the connection,
#: so one failure is one failure too many to repeat.
_ib_implied_vol_usable = True


async def _ib_implied_vol(
    contract: Contract, option_price: float, under_price: float, timeout: float = 4.0
) -> dict[str, Any] | None:
    """IB's model run on prices we supply, rather than on a live subscription.

    ``reqCalcImpliedVolatility`` is a calculation request: both prices go in as
    arguments, so it does not need the streaming entitlement that
    ``modelGreeks`` does. What comes back is the same ``OptionComputation``
    IB's own model produces, American exercise included — strictly better than
    implying a European volatility locally, when it answers.

    Returns None on anything unexpected. This is already the fallback path; it
    is not worth failing a portfolio over.

    Gated behind ``IBKR_USE_IB_IMPLIED_VOL`` and off by default: on the TWS
    build measured here the request comes back as error 320 and TWS closes the
    connection, which loses the whole portfolio load rather than one option's
    greeks. After the first failure it is not tried again for the rest of the
    process, so a book with twenty unpriced strikes cannot pay that cost twenty
    times over.
    """
    global _ib_implied_vol_usable
    if not settings.use_ib_implied_vol or not _ib_implied_vol_usable:
        return None
    try:
        ib = await connection.get()
        computation = await asyncio.wait_for(
            ib.calculateImpliedVolatilityAsync(contract, option_price, under_price),
            timeout=timeout,
        )
    except Exception as exc:
        _ib_implied_vol_usable = False
        log.warning(
            "calculateImpliedVolatility failed for %s (%s). Not attempting it again this "
            "session; volatilities will be implied locally instead.",
            contract.localSymbol,
            exc,
        )
        return None
    # ib_async returns an OptionComputation here, except when it returns a list
    # of them — seen live. Take the first either way rather than trusting the
    # documented shape.
    if isinstance(computation, (list, tuple)):
        computation = computation[0] if computation else None
    if computation is None or num(getattr(computation, "impliedVol", None)) is None:
        return None
    return {
        "impliedVol": num(getattr(computation, "impliedVol", None)),
        "delta": num(getattr(computation, "delta", None)),
        "gamma": num(getattr(computation, "gamma", None)),
        "vega": num(getattr(computation, "vega", None)),
        "theta": num(getattr(computation, "theta", None)),
        "optPrice": num(getattr(computation, "optPrice", None)) or option_price,
        "pvDividend": num(getattr(computation, "pvDividend", None)) or 0.0,
        "undPrice": num(getattr(computation, "undPrice", None)) or under_price,
        "source": "IB's own option model via calculateImpliedVolatility, fed the mark "
        "price — IB published no streaming model greeks for this contract",
    }


async def _imply_missing_greeks(
    options: Sequence[Holding], portfolio_prices: dict[int, float] | None = None
) -> None:
    """Last resort for an option IB priced but would not model.

    IB declines to publish model greeks for plenty of ordinary contracts — an
    illiquid strike with no two-sided market, or an instrument the account is
    not entitled to model data for. Measured on a live book, a minority of the
    option positions arrived with a mark price and no greeks. Held flat,
    they would silently shrink the risk the curve reports, and the trough is
    the one number this server exists to get right.

    Two ways out, in order of preference.

    **Ask IB to do the sum.** ``reqCalcImpliedVolatility`` takes the option
    price and the underlying price as *inputs* and returns IB's own model
    output — the same American-exercise model behind ``modelGreeks``, so the
    answer is consistent with every other number here. It is a pure
    calculation, not a subscription, which is exactly why it can succeed where
    the streaming greeks were refused.

    **Otherwise imply it locally**, which is a European volatility on an
    American option and absorbs the early-exercise premium into the vol —
    acceptable for a scenario curve, not for a quote.

    Either way the row is stamped with its ``source``, which the stress engine
    surfaces as a warning and the position report shows per position. Neither
    is passed off as IB's streaming model output.
    """
    from . import pricing

    needy = [h for h in options if h.greeks is None]
    if not needy:
        return

    # IB's own forward first, then anything the account holds outright. Both
    # beat a fresh quote: the first is the number IB's model actually used, the
    # second cannot be refused for want of an entitlement.
    known_underlying: dict[int, float] = {}
    for holding in options:
        price = (holding.greeks or {}).get("undPrice")
        if holding.und_conid and price:
            known_underlying.setdefault(holding.und_conid, float(price))
    for conid, price in (portfolio_prices or {}).items():
        known_underlying.setdefault(conid, price)

    rate = settings.risk_free_rate
    for holding in needy:
        contract = holding.contract
        if not contract.strike or not contract.right:
            continue
        spot = known_underlying.get(holding.und_conid or 0)
        if spot is None and holding.und_conid:
            detail = await details_for_conid(holding.und_conid)
            if detail is not None:
                spot = await spot_price(detail.contract, timeout=3.0)
                if spot:
                    known_underlying[holding.und_conid] = spot
        option_price = holding.market_price
        if option_price is None or option_price <= 0:
            option_price = await spot_price(contract, timeout=3.0)
        if not spot or not option_price or option_price <= 0:
            continue

        from_ib = await _ib_implied_vol(contract, option_price, spot)
        if from_ib is not None:
            holding.greeks = from_ib
            continue
        # Falling through to the local model is the ordinary case, not an
        # error: the IB path is opt-in and off by default.

        years = C.years_to_expiry(contract)
        is_forward = contract.secType == "FOP"
        forward = spot if is_forward else pricing.forward_from_spot(spot, years, rate)
        right = contract.right[:1].upper()
        iv = pricing.implied_vol_black76(option_price, forward, float(contract.strike), years, rate, right)
        if iv is None or iv <= 0:
            holding.greeks_error = (
                f"{holding.greeks_error or 'IB published no model greeks'}; a local "
                f"implied volatility could not be found either — the mark of "
                f"{option_price:g} is outside the no-arbitrage range for this contract"
            )
            continue
        greeks = pricing.black76_greeks(forward, float(contract.strike), years, iv, rate, right)
        if not is_forward:
            greeks = pricing.black_scholes_greeks(
                spot, float(contract.strike), years, iv, rate, right
            )
        holding.greeks = {
            "impliedVol": iv,
            **greeks,
            "optPrice": option_price,
            "pvDividend": 0.0,
            "undPrice": spot,
            "source": "implied locally from the mark price — IB published no model "
            "greeks for this contract",
        }


async def _backfill_underlying_price(
    options: Sequence[Holding], portfolio_prices: dict[int, float] | None = None
) -> None:
    """Give a forward to any option whose greeks arrived without one.

    Waiting for ``undPrice`` handles almost every case, but a contract that
    times out with a volatility and no underlying would otherwise be dropped
    from the curve entirely — and every option on the same underlying shares
    that number, so it can be fetched once and reused. Quoting the underlying
    directly is a slightly different figure from the forward IB's model used,
    which is why it is recorded as ``undPriceSource`` rather than passed off as
    IB's own.
    """
    needy = [h for h in options if h.greeks and h.greeks.get("undPrice") is None]
    if not needy:
        return
    by_underlying: dict[int, list[Holding]] = {}
    for holding in needy:
        if holding.und_conid:
            by_underlying.setdefault(holding.und_conid, []).append(holding)

    # One holding per underlying already has a good price more often than not;
    # reuse it before spending a market data line.
    known: dict[int, float] = {}
    for holding in options:
        price = (holding.greeks or {}).get("undPrice")
        if holding.und_conid and price:
            known.setdefault(holding.und_conid, float(price))
    for conid, price in (portfolio_prices or {}).items():
        known.setdefault(conid, price)

    for conid, group in by_underlying.items():
        price = known.get(conid)
        source = "a position the account already holds on this underlying"
        if price is None:
            detail = await details_for_conid(conid)
            if detail is None:
                continue
            price = await spot_price(detail.contract, timeout=3.0)
            source = "a direct quote on the underlying, not IB's model forward"
        if price is None:
            continue
        for holding in group:
            holding.greeks["undPrice"] = price
            holding.greeks["undPriceSource"] = source


async def _attach_underlying(holding: Holding) -> None:
    """Requalify the position's contract and fill in its underlying.

    Both halves matter.

    **Requalifying is not cosmetic.** The contracts on ``PortfolioItem`` come
    back from IB with ``exchange`` empty, and ``reqMktData`` on a contract
    without an exchange returns nothing at all — no ticks, no error, no greeks.
    Observed against a live account: every option position reported zero model
    greeks, while a surface request on the very same underlying returned values
    without trouble, because those contracts had come from
    ``reqContractDetails`` already qualified. Swapping in the canonical
    contract is what makes the position greeks arrive.

    **The underlying is never inferred from the expiry.** ES options expiring
    after the front quarterly has rolled are written on the next one, so the 30
    September end-of-month options belong to ESZ6 and not to ESU6.
    """
    try:
        detail = await details_for_conid(holding.contract.conId)
    except Exception as exc:
        log.warning("contract details failed for %s: %s", holding.contract, exc)
        return
    if detail is None:
        return
    if detail.contract.exchange:
        holding.contract = detail.contract
    holding.und_conid = detail.underConId or None
    holding.und_symbol = detail.contract.symbol
    if holding.und_conid:
        try:
            und = await details_for_conid(holding.und_conid)
        except Exception:
            und = None
        if und is not None:
            holding.und_symbol = und.contract.localSymbol or und.contract.symbol


def account_values(ib, account: str) -> dict[str, list[dict[str, Any]]]:
    """Account values grouped by tag, keeping every segment.

    IB reports the same tag three times — once for the whole account, once
    suffixed ``-S`` for the securities segment and once ``-C`` for commodities
    — and the segments are the interesting part: futures margin has to be met
    in the commodities segment, which IB funds by sweeping the securities one.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for av in ib.accountValues(account):
        grouped.setdefault(av.tag, []).append(
            {"value": av.value, "currency": av.currency, "account": av.account}
        )
    return grouped


def account_number(ib, account: str, tag: str, currency: str = "BASE") -> float | None:
    """One account figure as a float, preferring the base-currency row."""
    rows = [av for av in ib.accountValues(account) if av.tag == tag]
    if not rows:
        return None
    preferred = [av for av in rows if av.currency in (currency, "")] or rows
    return num(preferred[0].value)


# --------------------------------------------------------------------------
# volatility surface
# --------------------------------------------------------------------------


@dataclass
class SurfaceRequest:
    underlying: str
    expiries: list[str]
    strikes: list[float] | None = None
    min_strike: float | None = None
    max_strike: float | None = None
    rights: tuple[str, ...] = ("P",)
    sec_type: str | None = None
    max_strikes_per_expiry: int = 25
    band: float = 0.15
    trading_class: str | None = None


@dataclass
class SurfaceResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    underlying: dict[str, Any] = field(default_factory=dict)


def listed_centre(strikes: Sequence[float]) -> float | None:
    """A stand-in for the underlying price, taken from the chain itself.

    Exchanges list strikes around the money and extend the wings as the
    underlying moves, so the median listed strike is close to it. This is only
    ever used to centre a default band when no quote is available — an account
    without the data subscription, or a request made out of hours — because
    refusing to return a surface at all in that case is worse than returning
    one centred approximately. Every volatility in the result is still IB's.
    """
    values = sorted(s for s in strikes if s and s > 0)
    if not values:
        return None
    return float(values[len(values) // 2])


def _pick_strikes(
    available: Sequence[float], req: SurfaceRequest, centre: float | None
) -> list[float]:
    strikes = sorted({float(s) for s in available if s and s > 0})
    if req.strikes:
        wanted = sorted({float(s) for s in req.strikes})
        # Snap each requested strike to the nearest listed one so a caller that
        # asks for 6000 on a 25-point grid gets 6000, not nothing.
        return [min(strikes, key=lambda s: abs(s - w)) for w in wanted] if strikes else wanted
    lo = req.min_strike if req.min_strike is not None else (
        centre * (1 - req.band) if centre else None
    )
    hi = req.max_strike if req.max_strike is not None else (
        centre * (1 + req.band) if centre else None
    )
    if lo is not None:
        strikes = [s for s in strikes if s >= lo]
    if hi is not None:
        strikes = [s for s in strikes if s <= hi]
    if len(strikes) > req.max_strikes_per_expiry and centre:
        strikes = sorted(
            sorted(strikes, key=lambda s: abs(s - centre))[: req.max_strikes_per_expiry]
        )
    return strikes[: req.max_strikes_per_expiry]


async def vol_surface(req: SurfaceRequest) -> SurfaceResult:
    """Pull IB's implied volatility grid for an underlying.

    With this in hand, constant-volatility repricing is deterministic: the
    volatility for every (expiry, strike) is IB's, so the only thing the local
    model contributes is Black-76 arithmetic.
    """
    result = SurfaceResult()
    underlying, alternatives = await resolve_underlying(req.underlying, req.sec_type)
    centre = await spot_price(underlying)
    result.underlying = {**C.describe(underlying), "price": centre}
    if alternatives:
        result.underlying["alsoMatches"] = alternatives
        result.warnings.append(
            f"{req.underlying!r} was resolved as {underlying.secType} "
            f"({underlying.localSymbol or underlying.symbol}), but it also matches "
            f"{', '.join(alternatives)}. If that is the wrong instrument, pass sec_type. "
            "ES in particular is both the E-mini S&P future and Eversource Energy."
        )

    fop = underlying.secType == "FUT"
    sec_type = "FOP" if fop else "OPT"
    exchange = underlying.exchange if fop else "SMART"

    for expiry in req.expiries:
        chain = await chain_details(
            symbol=underlying.symbol,
            expiry=expiry,
            sec_type=sec_type,
            exchange=exchange,
            currency=underlying.currency or "USD",
            trading_class=req.trading_class,
            rights=req.rights,
        )
        if not chain:
            result.warnings.append(
                f"No contracts found for {underlying.symbol} {expiry}. Check the date "
                "(it may be a settlement date with no listing) or pass trading_class."
            )
            continue

        listed = [d.contract.strike for d in chain]
        band_centre = centre
        if band_centre is None and not req.strikes and req.min_strike is None:
            band_centre = listed_centre(listed)
            if band_centre is not None:
                result.warnings.append(
                    f"No quote for {underlying.symbol}, so the {expiry} band is centred on "
                    f"the median listed strike ({band_centre:g}) instead of the underlying "
                    "price. Pass strikes, or min_strike and max_strike, to control it."
                )
        strikes = _pick_strikes(listed, req, band_centre)
        wanted = set(strikes)
        selected = [d for d in chain if float(d.contract.strike) in wanted]
        if not selected:
            result.warnings.append(f"No listed strikes for {expiry} inside the requested band.")
            continue

        rows = await model_greeks_batch([d.contract for d in selected])
        for detail, row in zip(selected, rows):
            info = C.describe(detail.contract)
            if row.greeks is None:
                result.missing.append({**info, "reason": row.error})
                continue
            result.rows.append(
                {
                    **info,
                    "undConId": detail.underConId or None,
                    "impliedVol": row.greeks["impliedVol"],
                    "delta": row.greeks["delta"],
                    "gamma": row.greeks["gamma"],
                    "vega": row.greeks["vega"],
                    "theta": row.greeks["theta"],
                    "optPrice": row.greeks["optPrice"],
                    "pvDividend": row.greeks["pvDividend"],
                    "undPrice": row.greeks["undPrice"],
                    "yearsToExpiry": C.years_to_expiry(detail.contract),
                }
            )
    return result


async def skew_for(
    symbol: str,
    expiry_contract: Contract,
    forward: float,
    strike_band: float = 0.15,
    max_strikes: int = 15,
    sec_type: str | None = None,
) -> list[dict[str, Any]]:
    """A single expiry's smile around a forward, for the sticky-moneyness path.

    Kept separate from :func:`vol_surface` because the stress engine needs one
    tenor at a time and a narrow band: the point is to know the slope near the
    money, not to publish the whole grid.
    """
    req = SurfaceRequest(
        underlying=symbol,
        expiries=[expiry_contract.lastTradeDateOrContractMonth],
        rights=("P",),
        band=strike_band,
        max_strikes_per_expiry=max_strikes,
        sec_type=sec_type,
        trading_class=expiry_contract.tradingClass or None,
    )
    try:
        out = await vol_surface(req)
    except Exception as exc:
        log.warning("skew fetch failed for %s: %s", symbol, exc)
        return []
    return out.rows


# --------------------------------------------------------------------------
# hypothetical legs
# --------------------------------------------------------------------------


async def resolve_leg(leg: C.Leg) -> Contract:
    """Turn a :class:`~ibkr_risk_mcp.contracts.Leg` into a qualified contract.

    A conId short-circuits everything. Otherwise the descriptive fields are
    matched against the listed chain, which is what makes a settlement date
    usable in place of a last trading date, and what surfaces the case where
    one underlying has two contracts expiring the same morning: if the fields
    match more than one trading class the error names them rather than picking.
    """
    if leg.conid:
        detail = await details_for_conid(leg.conid)
        if detail is None:
            raise ValueError(f"IB does not recognise conId {leg.conid}")
        return detail.contract

    if not leg.symbol or not leg.secType:
        raise ValueError("A leg needs either conid, or symbol and secType.")

    ib = await connection.get()
    if leg.secType in ("STK", "FUT"):
        probe = Contract(
            symbol=leg.symbol,
            secType=leg.secType,
            currency=leg.currency,
            exchange=leg.exchange or ("SMART" if leg.secType == "STK" else ""),
        )
        if leg.expiry:
            probe.lastTradeDateOrContractMonth = _ib_date(leg.expiry)
        found = _remember(await ib.reqContractDetailsAsync(probe))
        if not found:
            raise ValueError(f"No contract matches {leg.symbol} {leg.secType}")
        found.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
        return found[0].contract

    if leg.expiry is None or leg.strike is None or leg.right is None:
        raise ValueError("An option leg needs expiry, strike and right (or a conid).")

    chain = await chain_details(
        symbol=leg.symbol,
        expiry=leg.expiry,
        sec_type=leg.secType,
        exchange=leg.exchange,
        currency=leg.currency,
        trading_class=leg.tradingClass,
        rights=(leg.right,),
    )
    matches = [d for d in chain if abs(float(d.contract.strike) - float(leg.strike)) < 1e-6]
    if not matches:
        listed = sorted({float(d.contract.strike) for d in chain})
        near = sorted(listed, key=lambda s: abs(s - float(leg.strike)))[:5]
        raise ValueError(
            f"No {leg.symbol} {leg.expiry} {leg.right} at strike {leg.strike}. "
            f"Nearest listed: {sorted(near)}"
        )
    classes = {d.contract.tradingClass for d in matches}
    if len(classes) > 1:
        raise ValueError(
            f"{leg.symbol} {leg.expiry} {leg.strike}{leg.right} matches several trading "
            f"classes ({', '.join(sorted(classes))}) — these settle differently even when "
            "they expire the same morning. Pass tradingClass or a conid."
        )
    return matches[0].contract


async def resolve_legs(legs: Sequence[C.Leg]) -> list[tuple[C.Leg, Contract]]:
    resolved = await asyncio.gather(*(resolve_leg(leg) for leg in legs))
    return list(zip(legs, resolved))


def batches(items: Iterable, size: int) -> Iterable[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
