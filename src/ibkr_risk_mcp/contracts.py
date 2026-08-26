"""Expiry and underlying normalisation.

Two facts about IB's contract data cause more wrong answers than anything else
in this server, and both are handled here rather than at the call sites:

1. ``lastTradeDateOrContractMonth`` is not reliably the settlement date. For
   AM-settled contracts — the quarterly ES options, trading class ``ES``, which
   settle to the Special Opening Quotation on the third Friday — some TWS
   builds report the previous afternoon's date instead, so an option settling
   on 18 December shows as the 17th while a PM-settled weekly expiring the same
   morning shows as the 18th. Sorting positions by last trade date then splits
   one expiry into two, and time to expiry computed from it is a day short.

   IB's own answer to this is ``ContractDetails.realExpirationDate``, added for
   exactly this case, with ``lastTradeTime`` distinguishing the two settlements
   (08:30 for the AM quarterly against 15:00 for a PM weekly on the same day).
   Where the details have been seen, they are used; where only a bare contract
   is available, the date heuristic below stands in. Verified against TWS
   server 178, which reports 20261218 for the December quarterly with
   ``lastTradeTime='08:30:00'`` — correct dates, AM settlement visible only in
   the time.

2. The underlying of a futures option is not implied by its expiry, and two
   options expiring the same morning need not share one. Confirmed on live
   data: the 18 December 2026 ``ES`` quarterly is written on ESZ6, while the
   ``EW3`` weekly expiring that same day is written on ESH7; the 30 September
   end-of-month options are on ESZ6, not ESU6. ``underConId`` from the contract
   details is the only reliable source, and nothing here ever guesses it from a
   date.

Everything in this module works on plain values or on objects with the ib_async
attribute names, so it is testable without TWS.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

#: Classes whose *quarterly* expirations settle in the morning to an opening
#: quotation, and therefore stop trading the business day before settlement.
#: The membership test is deliberately paired with a third-Friday check below:
#: these classes also carry weeklies and end-of-month options, which are
#: PM-settled and must not be shifted.
AM_SETTLED_CLASSES = frozenset(
    {
        "ES",  # E-mini S&P 500 quarterly options
        "NQ",  # E-mini Nasdaq-100 quarterly options
        "RTY",  # E-mini Russell 2000 quarterly options
        "SPX",  # cash index, standard (SPXW weeklies are PM-settled)
        "NDX",
        "RUT",
        "DJX",
        "XSP",
    }
)

#: Days per year used for every time-to-expiry in this server. ACT/365 fixed,
#: stated once so the pricing tests and the stress engine cannot drift apart.
DAYS_PER_YEAR = 365.0


class HasExpiry(Protocol):
    lastTradeDateOrContractMonth: str
    tradingClass: str


class Leg(BaseModel):
    """One leg of a hypothetical structure, for what-if margin and what-if
    stress.

    Give either ``conid`` — which is unambiguous and always preferred — or the
    descriptive fields. ``expiry`` accepts the last trading date or the
    settlement date in ``YYYYMMDD`` or ``YYYY-MM-DD`` form; for AM-settled
    expiries the two differ by a day and both resolve to the same contract.
    """

    conid: int | None = Field(
        default=None, description="IB contract id. Preferred: it resolves without ambiguity."
    )
    symbol: str | None = Field(default=None, description="Underlying root, e.g. ES or SPY.")
    secType: Literal["OPT", "FOP", "STK", "FUT"] | None = Field(
        default=None, description="OPT equity option, FOP futures option, STK stock, FUT future."
    )
    expiry: str | None = Field(default=None, description="YYYYMMDD or YYYY-MM-DD.")
    strike: float | None = None
    right: Literal["C", "P"] | None = None
    exchange: str | None = Field(
        default=None, description="Needed for futures and futures options, e.g. CME."
    )
    currency: str = "USD"
    tradingClass: str | None = Field(
        default=None,
        description="Disambiguates same-day expiries on one underlying, e.g. ES (quarterly, "
        "AM-settled) against EW4 (weekly, PM-settled).",
    )
    action: Literal["BUY", "SELL"] = "BUY"
    quantity: int = Field(default=1, ge=1)

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.action == "BUY" else -self.quantity


def parse_ib_date(raw: str) -> date:
    """Parse the date formats TWS puts in ``lastTradeDateOrContractMonth``:
    ``YYYYMMDD``, ``YYYYMM`` (contract month, taken as the first of the month)
    and the ``YYYYMMDD HH:MM:SS`` form some contract details carry."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty expiry")
    text = text.split()[0]
    if len(text) == 8:
        return datetime.strptime(text, "%Y%m%d").date()
    if len(text) == 6:
        return datetime.strptime(text, "%Y%m").date()
    raise ValueError(f"unrecognised IB date: {raw!r}")


def next_business_day(day: date) -> date:
    """The next weekday. There is no exchange holiday calendar here: if an
    AM-settled expiry ever falls the day after a holiday the shift is wrong by
    one day, which is visible in ``settlementDate`` rather than hidden."""
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def is_third_friday(day: date) -> bool:
    return day.weekday() == 4 and 15 <= day.day <= 21


def is_am_settled(trading_class: str, last_trade: date) -> bool:
    """Whether this contract settles the morning after its last trading day,
    judged from the class and the date alone.

    True only for a class known to have AM-settled quarterlies *and* an expiry
    whose next business day is the third Friday — so a weekly carrying the same
    trading class is left alone, and a build that already reports the
    settlement date is not shifted a second time.

    This is the fallback. :func:`remember_details` supplies IB's own answer
    where the contract details have been fetched.
    """
    if (trading_class or "").upper() not in AM_SETTLED_CLASSES:
        return False
    return is_third_friday(next_business_day(last_trade))


#: conId -> (settlement date, settles in the morning), as IB reported it in
#: ContractDetails. Populated by :func:`remember_details` as the market data
#: layer resolves contracts, and consulted by everything below.
#:
#: A registry rather than a parameter threaded through every signature: the
#: details are fetched in one place and wanted in a dozen, and keeping the
#: lookup here is what lets this module stay free of any IB import and remain
#: testable without TWS.
_known: dict[int, tuple[date, bool]] = {}


def remember_details(details: Any) -> None:
    """Record IB's authoritative expiration for a contract.

    ``realExpirationDate`` exists precisely because
    ``lastTradeDateOrContractMonth`` can be the last trading day rather than
    the expiration. ``lastTradeTime`` then says which settlement it is: 08:30
    is the Special Opening Quotation, 15:00 is the afternoon close. Two
    contracts expiring the same day differ in that field and nowhere else.
    """
    contract = getattr(details, "contract", None)
    conid = getattr(contract, "conId", 0)
    if not conid:
        return
    raw_real = (getattr(details, "realExpirationDate", "") or "").strip()
    raw_time = (getattr(details, "lastTradeTime", "") or "").strip()
    try:
        last_trade = parse_ib_date(contract.lastTradeDateOrContractMonth)
    except ValueError:
        return
    settles = parse_ib_date(raw_real) if raw_real else None
    morning = False
    if raw_time[:2].isdigit():
        morning = int(raw_time[:2]) < 12
    if settles is None:
        settles = (
            next_business_day(last_trade)
            if is_am_settled(getattr(contract, "tradingClass", ""), last_trade)
            else last_trade
        )
    _known[int(conid)] = (settles, morning or settles != last_trade)


def forget_details() -> None:
    """Drop the registry. Only for tests — contract definitions do not change."""
    _known.clear()


def settlement_date(contract: Any) -> date:
    """The date the contract actually settles, which is what expiry arithmetic
    must use. Equal to the last trade date for everything PM-settled."""
    known = _known.get(getattr(contract, "conId", 0) or 0)
    if known:
        return known[0]
    last_trade = parse_ib_date(contract.lastTradeDateOrContractMonth)
    if is_am_settled(getattr(contract, "tradingClass", ""), last_trade):
        return next_business_day(last_trade)
    return last_trade


def expiry_info(contract: Any, asof: date | None = None) -> dict[str, Any]:
    """Both dates and the time to expiry, always reported together.

    ``daysToExpiry`` counts calendar days to *settlement*. ``lastTradeDate`` is
    kept alongside it because it is what TWS shows and what the user will read
    off the screen; presenting only one of the two is what makes two contracts
    settling the same morning look like two different expiries.

    ``amSettled`` is true when the contract settles in the morning, whether or
    not that moves the date — on current TWS builds the December ES quarterly
    reports the right date and gives itself away only through its 08:30 last
    trade time.
    """
    asof = asof or date.today()
    last_trade = parse_ib_date(contract.lastTradeDateOrContractMonth)
    known = _known.get(getattr(contract, "conId", 0) or 0)
    settles = known[0] if known else settlement_date(contract)
    am = known[1] if known else settles != last_trade
    days = (settles - asof).days
    return {
        "lastTradeDate": last_trade.isoformat(),
        "settlementDate": settles.isoformat(),
        "amSettled": am,
        "daysToExpiry": days,
        "yearsToExpiry": max(days, 0) / DAYS_PER_YEAR,
    }


def years_to_expiry(contract: Any, asof: date | None = None, offset_days: int = 0) -> float:
    """Time to settlement in years, ACT/365, floored at a fraction of a day.

    An expired or same-day contract is priced at a hundredth of a day rather
    than at zero: intrinsic value is the right answer there, and the pricers
    return it, but a hard zero also makes every vega and gamma vanish, which
    reads as "no risk" when the truth is "no time".
    """
    asof = asof or date.today()
    days = (settlement_date(contract) - asof).days - offset_days
    return max(days, 0.01) / DAYS_PER_YEAR


def is_option(sec_type: str) -> bool:
    return sec_type in ("OPT", "FOP")


def asset_class(sec_type: str) -> str:
    """Bucket a secType into the classes the stress engine shocks differently.

    Everything unrecognised lands in ``other`` and is held flat under shock —
    reported as such, never quietly folded into the equity bucket.
    """
    return {
        "OPT": "option",
        "FOP": "option",
        "STK": "equity",
        "FUND": "equity",
        "ETF": "equity",
        "FUT": "future",
        "BOND": "bond",
        "BILL": "bond",
        "CASH": "cash",
        "CRYPTO": "other",
    }.get(sec_type, "other")


#: The three-letter ISO codes IB uses as the ``symbol`` of a currency future
#: and its options. IB names the *currency*, not the exchange ticker: the 6E is
#: ``symbol="EUR"`` and the 6C is ``symbol="CAD"``, so a currency code on a FUT
#: or FOP is a reliable tell that the underlying is an exchange rate.
CURRENCY_SYMBOLS = frozenset(
    """AUD BRL CAD CHF CNH CZK DKK EUR GBP HKD HUF ILS INR JPY KRW MXN NOK NZD
    PLN RUB SEK SGD TRY USD ZAR""".split()
)

#: Futures roots by the factor they track, for the roots a book like this
#: actually holds. Everything not listed falls through to equity, which is the
#: right default for an index future and the wrong one for anything exotic —
#: hence :func:`risk_group` being reported rather than assumed.
_FUTURE_GROUPS: dict[str, str] = {
    root: group
    for group, roots in {
        "rates": "ZT ZF ZN TN ZB UB GE ZQ SR1 SR3 FGBS FGBM FGBL FGBX",
        "metals": "GC MGC SI SIL HG PA PL",
        "energy": "CL MCL NG RB HO BZ QM QG",
    }.items()
    for root in roots.split()
}


def risk_group(contract: Any) -> str:
    """Which risk factor a position responds to — as opposed to what kind of
    instrument it is.

    :func:`asset_class` answers "is this an option". This answers "is this an
    equity". The two are orthogonal, and conflating them is what puts a CAD
    strangle and an ES ratio spread in one bucket called ``option`` and then
    shocks both by the same equity percentage. Measured on a live account, that
    one CAD strangle contributed −21,716 at a 20% fall *and* −7,183 at a 10%
    rise, against −29,027 and +2,408 for an entire ES campaign: it dominated
    both tails of a curve that was supposed to be about equities.

    TWS Risk Navigator draws the same line — its Equity tab excludes FX and
    fixed income — which is why its curve and this one only agree once the
    non-equity legs are stood down.

    **This is a table and one heuristic, not a deduction.** IB publishes no
    reliable asset class for futures, and a bond or gold ETF quoted as ``STK``
    lands in ``equity`` with no API field to say otherwise. So the group is
    reported in every result that uses it, and can be overridden per symbol.
    """
    sec_type = getattr(contract, "secType", "") or ""
    symbol = (getattr(contract, "symbol", "") or "").upper()
    if sec_type == "CASH":
        return "fx"
    if sec_type in ("BOND", "BILL"):
        return "rates"
    if sec_type in ("FUT", "FOP"):
        if symbol in CURRENCY_SYMBOLS:
            return "fx"
        return _FUTURE_GROUPS.get(symbol, "equity")
    if sec_type in ("STK", "OPT", "FUND", "ETF", "IND"):
        return "equity"
    if sec_type == "CMDTY":
        return "metals"
    return "other"


def contract_multiplier(contract: Any, default: float = 1.0) -> float:
    """The contract multiplier as a number, read from the contract and never
    assumed. ES is 50 and MES is 5; hardcoding either is how a micro position
    ends up reported ten times too large."""
    raw = getattr(contract, "multiplier", "") or ""
    text = str(raw).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def describe(contract: Any) -> dict[str, Any]:
    """The identifying fields of a contract, in the shape every tool returns
    them."""
    out: dict[str, Any] = {
        "conid": getattr(contract, "conId", None),
        "symbol": getattr(contract, "symbol", None),
        "localSymbol": getattr(contract, "localSymbol", None) or None,
        "secType": getattr(contract, "secType", None),
        "exchange": getattr(contract, "exchange", None) or None,
        "currency": getattr(contract, "currency", None),
        "tradingClass": getattr(contract, "tradingClass", None) or None,
        "multiplier": contract_multiplier(contract),
        "riskGroup": risk_group(contract),
    }
    if is_option(out["secType"] or ""):
        out["right"] = getattr(contract, "right", None) or None
        out["strike"] = getattr(contract, "strike", None) or None
        out.update(expiry_info(contract))
    elif getattr(contract, "lastTradeDateOrContractMonth", ""):
        out.update(expiry_info(contract))
    return out
