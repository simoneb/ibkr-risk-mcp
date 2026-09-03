import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

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


#: The transports :func:`ibkr_risk_mcp.server.main` knows how to run.
TRANSPORTS: tuple[str, ...] = ("stdio", "http")


def _transport_env(name: str, default: str) -> str:
    """The configured transport, or a refusal to start.

    Falling back to stdio on an unrecognised value would be the worse failure:
    a service unit with ``IBKR_MCP_TRANSPORT=streamable-http`` in it would come
    up, find no stdin worth reading, exit cleanly, and look from the outside
    like a server that ran and stopped — with nothing listening on the port and
    nothing in the log to say why.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value not in TRANSPORTS:
        raise ValueError(
            f"{name}={raw!r} is not a transport this server has. "
            f"Use one of: {', '.join(TRANSPORTS)}."
        )
    return value


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
    #: Whether to ask IB to imply a volatility for an option it would not
    #: publish model greeks for, via ``reqCalcImpliedVolatility``.
    #:
    #: **Off by default, and the reason is not caution.** Measured against live
    #: TWS: ib_async's request for this is answered with error 320 — "Error
    #: reading request. Please use 'Key=Value' format for Misc Options" — and
    #: TWS then **closes the API connection**. A protocol error mid-way through
    #: a portfolio load takes the socket down with it, which is a great deal
    #: worse than the thing it was trying to fix.
    #:
    #: When it does work it is the better answer — IB's own American-exercise
    #: model rather than a European volatility implied locally — so it stays
    #: available for a TWS build that accepts it. It is not something to have on
    #: the default path of a read-only risk server.
    use_ib_implied_vol: bool
    connect_timeout: float
    #: How long to wait for IB to answer a what-if order. It is not always
    #: answered at all — TWS silently drops the request when "Read-Only API" is
    #: on, and on a contract the account cannot price — and an unbounded wait
    #: there hangs the tool call rather than reporting anything. Kept short: IB
    #: replies in well under a second when it replies.
    whatif_timeout: float

    # ----------------------------------------------------------------------
    # How the server is served. Nothing below this line concerns IB.
    #
    #: ``stdio`` for a client that launches this server as a subprocess,
    #: ``http`` for streamable HTTP behind a reverse proxy.
    #:
    #: stdio remains the default on purpose: the desktop extension, the `uvx`
    #: invocation and every existing client configuration keep working with no
    #: change at all, and a server that listens on a port does so because
    #: somebody asked it to.
    mcp_transport: str
    #: The address the HTTP transport binds. Loopback by default — the intended
    #: deployment puts a reverse proxy in front, which terminates TLS and, once
    #: the auth work lands, is no longer the only thing standing in front of an
    #: account's risk data. Binding 0.0.0.0 is a deliberate act, not a default.
    #:
    #: Note the names: ``IBKR_HOST`` and ``IBKR_PORT`` already mean *TWS's*
    #: host and port. These are a different machine's, in the deployment this
    #: exists for, so they get their own prefix rather than overloading a word
    #: that already means something in the same file.
    mcp_host: str
    mcp_port: int
    #: Path the MCP endpoint is mounted at.
    mcp_path: str

    #: Whether the HTTP transport requires a bearer token. Off by default, and
    #: irrelevant under stdio, where the client already launched the process.
    mcp_auth: bool
    #: The identity provider. Its signature, and this exact issuer string, are
    #: what a token has to carry.
    mcp_auth_issuer: str | None
    #: Where that provider publishes its signing keys. Defaults to the
    #: conventional path under the issuer, which is right for most providers
    #: and wrong for enough of them to be worth overriding.
    mcp_auth_jwks_url: str | None
    #: The audience a token must name. Separate from the resource URL because
    #: providers differ: some mint the resource URL itself, others an opaque
    #: API identifier chosen at registration. Defaults to the resource URL.
    #:
    #: Not optional in effect. Without it a token minted by the same provider
    #: for some *other* application would validate here, which is the whole
    #: reason the claim exists.
    mcp_auth_audience: str | None
    #: This server's own public URL, published at
    #: /.well-known/oauth-protected-resource so a client can discover where to
    #: authenticate.
    mcp_resource_url: str | None
    #: Who may use this server, by token subject. There is one account behind
    #: this process and no per-caller data, so this list *is* the authorisation
    #: model — a valid token from the right issuer, naming a subject that is
    #: not here, is a stranger holding the whole position book.
    #:
    #: An empty list never means "everyone". With auth on it means the
    #: configuration is wrong, and the server refuses to start.
    mcp_allowed_subjects: tuple[str, ...]
    #: The scope granted to an allowlisted subject, and required of every
    #: request. Its name does not matter; what matters is that granting it is
    #: this server's decision rather than the provider's.
    mcp_auth_scope: str


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
        use_ib_implied_vol=_bool_env("IBKR_USE_IB_IMPLIED_VOL", False),
        connect_timeout=_float_env("IBKR_CONNECT_TIMEOUT", 6.0),
        whatif_timeout=_float_env("IBKR_WHATIF_TIMEOUT", 5.0),
        mcp_transport=_transport_env("IBKR_MCP_TRANSPORT", "stdio"),
        mcp_host=os.environ.get("IBKR_MCP_HOST", "127.0.0.1"),
        mcp_port=_int_env("IBKR_MCP_PORT", 8765),
        mcp_path=os.environ.get("IBKR_MCP_PATH", "/mcp"),
        **_auth_env(),
    )


def _str_env(name: str) -> str | None:
    raw = os.environ.get(name)
    return raw.strip() or None if raw else None


#: Hosts for which plaintext HTTP is allowed in the auth URLs. Loopback only,
#: and only because the end-to-end auth test stands a stub provider up on it.
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]", "::1")


def _require_secure_url(name: str, value: str | None) -> None:
    """Refuse a plaintext auth endpoint.

    The JWKS URL is the one that matters most and looks the most harmless: it
    carries the public keys every token is checked against, so anyone able to
    answer that request can hand this server a key of their own and mint
    tokens it will believe. Over http that is anyone on the path.
    """
    if not value:
        return
    parsed = urlparse(value)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS:
        return
    raise ValueError(
        f"{name}={value!r} must be https. Plaintext is accepted only on loopback, "
        "for the local auth test."
    )


def _require_resource_matches_path(resource: str | None, path: str) -> None:
    """The published resource URL has to end at the MCP endpoint.

    Claude compares the `resource` field of the protected-resource metadata
    against the URL a user types into the connector dialog, and the comparison
    is exact — path included. A resource of ``https://risk.example.com`` on a
    server whose endpoint is ``/mcp`` produces a document that is internally
    consistent, passes every test here, and fails at connection time with
    nothing pointing at why. Cheaper to catch at startup.
    """
    if not resource:
        return
    resource_path = urlparse(resource).path.rstrip("/")
    if resource_path != path.rstrip("/"):
        raise ValueError(
            f"IBKR_MCP_RESOURCE_URL={resource!r} ends at {resource_path or '/'!r} but the "
            f"MCP endpoint is served at {path!r}. These must agree: the resource URL is "
            "published for discovery and compared, character for character, against the "
            "URL entered in the client."
        )


def _auth_env() -> dict[str, Any]:
    """The bearer-token settings, refusing anything that would half-enable them.

    Auth that is on but misconfigured must not start. The dangerous shape is
    not a server that rejects everyone — that is loud and gets fixed in a
    minute — but one that looks configured and is not: no allowlist, no
    audience, an issuer nobody checked. Each of those is a way for a token this
    server should never have accepted to open an account's book, so each is a
    startup failure rather than a warning in a log nobody reads.
    """
    enabled = _bool_env("IBKR_MCP_AUTH", False)
    issuer = _str_env("IBKR_MCP_AUTH_ISSUER")
    resource = _str_env("IBKR_MCP_RESOURCE_URL")
    subjects = tuple(
        s.strip() for s in (os.environ.get("IBKR_MCP_ALLOWED_SUBJECTS") or "").split(",") if s.strip()
    )
    jwks = _str_env("IBKR_MCP_AUTH_JWKS_URL")
    audience = _str_env("IBKR_MCP_AUTH_AUDIENCE")

    if enabled:
        missing = [
            name
            for name, value in (
                ("IBKR_MCP_AUTH_ISSUER", issuer),
                ("IBKR_MCP_RESOURCE_URL", resource),
                ("IBKR_MCP_ALLOWED_SUBJECTS", subjects),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "IBKR_MCP_AUTH is on but " + ", ".join(missing) + " "
                + ("is" if len(missing) == 1 else "are")
                + " not set. An empty allowlist is not 'allow everyone' — there is one "
                "account behind this server and no way to tell callers apart once a token "
                "is accepted, so the list of subjects is the whole of the authorisation."
            )
        if not jwks:
            jwks = issuer.rstrip("/") + "/.well-known/jwks.json"  # type: ignore[union-attr]
        if not audience:
            audience = resource

        _require_secure_url("IBKR_MCP_AUTH_ISSUER", issuer)
        _require_secure_url("IBKR_MCP_AUTH_JWKS_URL", jwks)
        _require_secure_url("IBKR_MCP_RESOURCE_URL", resource)
        _require_resource_matches_path(resource, os.environ.get("IBKR_MCP_PATH", "/mcp"))

    return {
        "mcp_auth": enabled,
        "mcp_auth_issuer": issuer,
        "mcp_auth_jwks_url": jwks,
        "mcp_auth_audience": audience,
        "mcp_resource_url": resource,
        "mcp_allowed_subjects": subjects,
        "mcp_auth_scope": os.environ.get("IBKR_MCP_AUTH_SCOPE", "risk:read").strip() or "risk:read",
    }


settings = load_settings()
