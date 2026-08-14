"""ib_async lifecycle: one connection per process, reconnected on demand.

TWS keys API sessions by client id and refuses a second connection on an id
already in use, so a connection-per-call design would fight itself the moment
two tools overlap. Instead there is a single ``IB`` instance behind a lock,
opened lazily and reopened if TWS drops it.

The other job of this module is telling apart the ways connecting fails.
"Cannot connect" covers at least four different problems with four different
fixes — TWS not running, TWS running with the API switch off, the client id
already taken, and a session that is up but has no account logged in — and a
caller that cannot distinguish them cannot say anything useful to the user.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ib_async import IB

from .config import KNOWN_PORTS, Settings, settings

log = logging.getLogger(__name__)

#: Ports TWS answers on where the API handshake never completes because
#: "Enable ActiveX and Socket Clients" is off. IB's own error for a client id
#: clash, and the text it arrives with, are matched on separately.
_CLIENT_ID_MARKERS = ("client id is already in use", "already in use")


class AccountAmbiguous(RuntimeError):
    """More than one account is logged in and none was configured.

    This is a refusal, not a fallback. IB answers ``portfolio("")`` and
    ``accountValues("")`` with *every* account at once, so carrying on would
    add two portfolios together and reconcile the total against one account's
    NetLiquidation — a wrong answer that looks like a right one. Verified
    against a live two-account login.
    """

    def __init__(self, accounts: list[str]):
        self.accounts = accounts
        super().__init__(
            "This login manages more than one account "
            f"({', '.join(accounts)}) and IBKR_ACCOUNT is not set. Figures from two "
            "accounts must not be added together, so nothing is reported until one is "
            "chosen. Set IBKR_ACCOUNT to the account code you want."
        )


class IBUnavailable(RuntimeError):
    """A tool needed TWS and could not get it. Carries the same ``state`` and
    ``hint`` that :meth:`IBConnection.probe` reports, so a failing tool tells
    the user the same thing ``check_connection`` would."""

    def __init__(self, state: str, hint: str, detail: str = ""):
        super().__init__(f"{state}: {hint}" + (f" ({detail})" if detail else ""))
        self.state = state
        self.hint = hint
        self.detail = detail


async def port_is_listening(host: str, port: int, timeout: float = 1.5) -> bool:
    """Whether something accepts a TCP connection on the port. This is the
    cheap half of the diagnosis: it separates "nothing is running" from every
    problem that needs TWS to be running in the first place."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:  # pragma: no cover - platform-dependent teardown noise
        pass
    return True


async def scan_known_ports(host: str) -> list[dict[str, Any]]:
    """Which of the four default TWS/Gateway ports are listening. Reported when
    the configured port is dead, because the difference between "TWS is not
    running" and "IBKR_PORT points at the paper port while TWS is live" is
    otherwise invisible."""
    results = await asyncio.gather(
        *(port_is_listening(host, port) for port, _ in KNOWN_PORTS)
    )
    return [
        {"port": port, "description": desc, "listening": ok}
        for (port, desc), ok in zip(KNOWN_PORTS, results)
    ]


class IBConnection:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._ib = IB()
        self._lock = asyncio.Lock()
        self._api_errors: list[str] = []
        self._ib.errorEvent += self._on_error
        self._market_data_type_set = False

    def _on_error(self, reqId: int, code: int, msg: str, contract: Any) -> None:
        # 2104/2106/2158 and friends are "market data farm is fine" notices.
        # They arrive on the error channel and mean nothing went wrong.
        if code in (2104, 2106, 2107, 2108, 2119, 2158, 2100):
            return
        self._api_errors.append(f"{code}: {msg}")
        del self._api_errors[:-20]

    @property
    def raw(self) -> IB:
        """The underlying ib_async client. Only valid after :meth:`get`."""
        return self._ib

    @property
    def connected(self) -> bool:
        return self._ib.isConnected()

    async def get(self) -> IB:
        """The connected client, connecting on first use.

        Raises :class:`IBUnavailable` with a state and a hint rather than a
        transport error, so every tool fails the same legible way.
        """
        if self._ib.isConnected():
            return self._ib
        async with self._lock:
            if self._ib.isConnected():
                return self._ib
            await self._connect()
            return self._ib

    async def _connect(self) -> None:
        cfg = self._cfg
        self._api_errors.clear()
        if not await port_is_listening(cfg.host, cfg.port):
            others = [p for p in await scan_known_ports(cfg.host) if p["listening"]]
            hint = (
                f"Nothing is listening on {cfg.host}:{cfg.port}. Either TWS/IB Gateway is "
                "not running, or it is running with the API switch off — TWS opens no "
                "port at all until 'Enable ActiveX and Socket Clients' is ticked under "
                "File > Global Configuration > API > Settings. Check the port on that "
                "same screen while you are there."
            )
            if others:
                found = ", ".join(f"{p['port']} ({p['description']})" for p in others)
                hint += f" Something is listening on {found} — set IBKR_PORT to match."
            raise IBUnavailable("not_listening", hint)

        try:
            await self._ib.connectAsync(
                cfg.host,
                cfg.port,
                clientId=cfg.client_id,
                timeout=cfg.connect_timeout,
                # A read-only client cannot place orders even by mistake. The
                # what-if path needs the order channel — nothing it sends is
                # routable, but it is still the order channel — so the gate that
                # enables that tool is the same one that opens this.
                readonly=not cfg.enable_whatif,
                account=cfg.account or "",
            )
        except asyncio.TimeoutError:
            raise IBUnavailable(
                "api_not_enabled",
                f"{cfg.host}:{cfg.port} accepts connections but never completed the API "
                "handshake. In TWS: File > Global Configuration > API > Settings, tick "
                '"Enable ActiveX and Socket Clients", and add this machine to "Trusted IPs".',
                "; ".join(self._api_errors[-3:]),
            ) from None
        except Exception as exc:  # ib_async raises plain ConnectionError here
            text = f"{exc} {' '.join(self._api_errors[-3:])}".lower()
            if any(marker in text for marker in _CLIENT_ID_MARKERS):
                raise IBUnavailable(
                    "client_id_in_use",
                    f"clientId {cfg.client_id} is already connected to this TWS. Set "
                    "IBKR_CLIENT_ID to a value no other script uses (and never 0, which "
                    "TWS reserves for manually placed orders).",
                    str(exc),
                ) from None
            raise IBUnavailable(
                "connect_failed",
                f"TWS refused the connection: {exc}. Check the API settings and that the "
                "client id is free.",
                "; ".join(self._api_errors[-3:]),
            ) from None

        if not self._market_data_type_set:
            self._ib.reqMarketDataType(cfg.market_data_type)
            self._market_data_type_set = True

        if not self._ib.managedAccounts():
            await self.disconnect()
            raise IBUnavailable(
                "not_logged_in",
                "The API answered but reports no account. TWS is running without a "
                "logged-in user, or is still loading — log in and retry.",
            )
        await self._ensure_account_updates()

    async def _ensure_account_updates(self) -> None:
        """Make sure ``accountValues`` and ``portfolio`` are populated for the
        account in use.

        IB allows **one** account subscription at a time: requesting a second
        silently cancels the first, which is why a two-account login reports 71
        positions for one and none for the other depending on which was asked
        for last. Observed live. Subscribing only to the resolved account keeps
        that from happening by accident.

        With the account still ambiguous this does nothing rather than
        subscribing to "" — connecting should succeed so that check_connection
        can explain the ambiguity.
        """
        account = self.account
        if account is None:
            return
        if self._ib.accountValues(account):
            return
        try:
            await asyncio.wait_for(self._ib.reqAccountUpdatesAsync(account), timeout=10)
        except Exception as exc:  # ib_async raises broadly; a timeout is one of many
            log.warning("reqAccountUpdates did not complete: %s", exc)

    @property
    def account(self) -> str | None:
        """The account these tools operate on. Configured explicitly, or the
        only managed account when there is exactly one. On a multi-account
        login with nothing configured this stays None, and
        :meth:`require_account` turns that into a refusal."""
        if self._cfg.account:
            return self._cfg.account
        accounts = self._ib.managedAccounts()
        return accounts[0] if len(accounts) == 1 else None

    def require_account(self) -> str:
        """The account, or an error naming the candidates.

        Every tool that reads positions or account values goes through this.
        Passing an empty account code to IB does not mean "the default one", it
        means "all of them", and the difference is a silently combined
        portfolio.
        """
        chosen = self.account
        if chosen:
            return chosen
        accounts = list(self._ib.managedAccounts())
        if not accounts:
            raise IBUnavailable(
                "not_logged_in", "The API reports no account. Log in to TWS and retry."
            )
        raise AccountAmbiguous(accounts)

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()
            await asyncio.sleep(0)

    async def probe(self) -> dict[str, Any]:
        """Diagnose the connection without raising. Always returns a ``state``
        and a ``hint``; never leaves a half-open session behind."""
        cfg = self._cfg
        if self._ib.isConnected():
            accounts = self._ib.managedAccounts()
            if not accounts:
                return {
                    "state": "not_logged_in",
                    "hint": "Connected to the API but no account is logged in to TWS.",
                    "connected": True,
                    "accounts": [],
                }
            return {
                "state": "connected",
                "hint": "TWS is reachable and an account is logged in.",
                "connected": True,
                "accounts": list(accounts),
                "serverVersion": self._ib.client.serverVersion(),
            }
        try:
            await self.get()
        except IBUnavailable as exc:
            out: dict[str, Any] = {
                "state": exc.state,
                "hint": exc.hint,
                "connected": False,
                "accounts": [],
            }
            if exc.detail:
                out["detail"] = exc.detail
            if exc.state == "not_listening":
                out["portScan"] = await scan_known_ports(cfg.host)
            return out
        return {
            "state": "connected",
            "hint": "TWS is reachable and an account is logged in.",
            "connected": True,
            "accounts": list(self._ib.managedAccounts()),
            "serverVersion": self._ib.client.serverVersion(),
        }


#: The process-wide connection. Imported by the other modules rather than
#: passed around, because there is exactly one and threading it through every
#: signature would only obscure that.
connection = IBConnection(settings)
