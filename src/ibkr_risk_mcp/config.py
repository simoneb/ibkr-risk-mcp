import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


#: The ports TWS and IB Gateway listen on out of the box, in the order it is
#: worth probing them. check_connection scans these when the configured port is
#: dead, because "wrong port" and "TWS not running" look identical otherwise and
#: only one of them is the user's problem to fix.
KNOWN_PORTS: tuple[tuple[int, str], ...] = (
    (7496, "TWS live"),
    (7497, "TWS paper"),
    (4001, "IB Gateway live"),
    (4002, "IB Gateway paper"),
)


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    #: TWS keys API sessions by client id and refuses a second connection using
    #: one already in use, so this must not collide with any other script
    #: pointed at the same TWS. 0 is special — it receives orders placed
    #: manually in TWS — which is exactly why it is not the default here.
    client_id: int
    #: Optional account code. Only needed on a multi-account login, where IB
    #: returns portfolio and account values for every account at once and
    #: summing them is meaningless.
    account: str | None
    #: Whether whatif_order may reach IB's order path at all. `whatIf=True`
    #: orders are evaluated by IB's margin engine and never routed to a market,
    #: but this server is read-only by default and that is worth being able to
    #: prove rather than assert. It lives in the process environment so the
    #: model driving the tools cannot change it.
    enable_whatif: bool
    #: 1 live, 2 frozen, 3 delayed, 4 delayed-frozen.
    #:
    #: Delayed data *does* carry model greeks — IB publishes them on the delayed
    #: option-computation tick and ib_async maps them to the same field —
    #: verified against live TWS, where type 3 returned an implied volatility on
    #: an account that got nothing at all from types 1 and 2. So 3 is a real
    #: fallback for an account without the live subscription, not a broken mode:
    #: the numbers are simply a quarter of an hour old, which matters for a
    #: fast-moving book and not at all for the shape of a P&L curve.
    market_data_type: int
    #: How long to wait for IB to push modelGreeks for one contract.
    #:
    #: Short on purpose. IB answers a market data request within about a second
    #: or not at all, so a long timeout buys nothing and costs it on every
    #: contract that was never going to answer. The refusals it does send —
    #: error 354 and friends — end the wait immediately regardless.
    greeks_timeout: float
    #: IB allows roughly 100 messages/second and around 50 concurrent market
    #: data lines on a default account. Both surface requests and greeks
    #: requests go through a semaphore of this size, and each subscription is
    #: cancelled the moment its data lands.
    max_market_data_lines: int
    #: Risk-free rate used for discounting in the local repricing. IB does not
    #: publish the rate behind its own model, so this is an input, not a
    #: measurement: it moves option values by little over the horizons this
    #: server deals with, but it is not zero either.
    risk_free_rate: float
    connect_timeout: float
    #: How long to wait for IB to answer a what-if order. It is not always
    #: answered at all — TWS silently drops the request when "Read-Only API" is
    #: on, and on a contract the account cannot price — and an unbounded wait
    #: there hangs the tool call rather than reporting anything. Kept short: IB
    #: replies in well under a second when it replies.
    whatif_timeout: float


def load_settings() -> Settings:
    return Settings(
        host=os.environ.get("IBKR_HOST", "127.0.0.1"),
        port=_int_env("IBKR_PORT", 7496),
        client_id=_int_env("IBKR_CLIENT_ID", 17),
        account=os.environ.get("IBKR_ACCOUNT") or None,
        enable_whatif=_bool_env("IBKR_ENABLE_WHATIF", False),
        market_data_type=_int_env("IBKR_MARKET_DATA_TYPE", 1),
        greeks_timeout=_float_env("IBKR_GREEKS_TIMEOUT", 4.0),
        max_market_data_lines=_int_env("IBKR_MAX_MKT_DATA_LINES", 40),
        risk_free_rate=_float_env("IBKR_RISK_FREE_RATE", 0.04),
        connect_timeout=_float_env("IBKR_CONNECT_TIMEOUT", 6.0),
        whatif_timeout=_float_env("IBKR_WHATIF_TIMEOUT", 5.0),
    )


settings = load_settings()
