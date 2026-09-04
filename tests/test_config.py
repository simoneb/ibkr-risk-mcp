"""Where this server's configuration meets the transport's.

The transport and bearer-token settings themselves live in `mcp_remote_auth`
and are tested there. What is worth testing *here* is the seam: this server has
an `IBKR_HOST` and an `IBKR_PORT` that mean TWS's, and an `IBKR_MCP_HOST` and
`IBKR_MCP_PORT` that mean its own listener's. In the deployment these exist
for, those are two different machines, and the day one starts reading the
other's variable is the day the server points its own socket at itself.
"""

from __future__ import annotations

import pytest

from ibkr_risk_mcp.config import load_settings
from mcp_remote_auth import load_remote_settings

PREFIX = "IBKR_MCP_"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start from no transport configuration at all, whatever the developer's
    shell or `.env` happens to hold."""
    for name in ("TRANSPORT", "HOST", "PORT", "PATH", "AUTH", "AUTH_ISSUER",
                 "RESOURCE_URL", "ALLOWED_SUBJECTS", "AUTH_AUDIENCE",
                 "AUTH_JWKS_URL", "AUTH_SCOPE"):
        monkeypatch.delenv(PREFIX + name, raising=False)


def test_the_two_namespaces_do_not_leak_into_each_other(monkeypatch):
    """A remote deployment sets IBKR_HOST to the Gateway and IBKR_MCP_HOST to
    the interface the reverse proxy reaches. If either read the other, the
    server would bind the Gateway's address or dial its own listener."""
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    monkeypatch.setenv("IBKR_PORT", "4001")

    assert (load_settings().host, load_settings().port) == ("10.0.0.5", 4001)
    listener = load_remote_settings(PREFIX)
    assert (listener.host, listener.port) == ("127.0.0.1", 8765)

    monkeypatch.setenv("IBKR_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("IBKR_MCP_PORT", "9000")

    ib = load_settings()
    assert (ib.host, ib.port) == ("10.0.0.5", 4001), "TWS's address must not follow the listener"
    listener = load_remote_settings(PREFIX)
    assert (listener.host, listener.port) == ("0.0.0.0", 9000)


def test_the_prefix_reaches_the_transport_setting(monkeypatch):
    """Guards against the prefix being changed on one side only."""
    monkeypatch.setenv("IBKR_MCP_TRANSPORT", "http")
    assert load_remote_settings(PREFIX).transport == "http"


def test_stdio_remains_the_default(monkeypatch):
    """Every existing client configuration launches this as a subprocess. That
    has to keep working with no change at all."""
    assert load_remote_settings(PREFIX).transport == "stdio"
    assert load_remote_settings(PREFIX).auth is False
