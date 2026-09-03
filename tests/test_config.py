"""How the server is served.

These are about the transport settings only, and specifically about the two
ways getting them wrong is expensive. The first is silence: a mistyped
transport that falls back to stdio produces a service which starts, finds no
stdin, exits zero, and leaves nothing on the port and nothing in the log. The
second is collision: `IBKR_HOST` and `IBKR_PORT` already mean TWS's host and
port, and the day the HTTP listener starts reading them is the day a remote
deployment quietly points its own socket at itself.
"""

from __future__ import annotations

import pytest

from ibkr_risk_mcp.config import TRANSPORTS, load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start from no transport configuration at all, whatever the developer's
    shell or `.env` happens to hold."""
    for name in (
        "IBKR_MCP_TRANSPORT",
        "IBKR_MCP_HOST",
        "IBKR_MCP_PORT",
        "IBKR_MCP_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_to_stdio_on_loopback():
    cfg = load_settings()
    assert cfg.mcp_transport == "stdio"
    assert cfg.mcp_host == "127.0.0.1"
    assert cfg.mcp_path == "/mcp"


@pytest.mark.parametrize("value", ["http", "HTTP", "  http  "])
def test_transport_is_case_and_whitespace_tolerant(monkeypatch, value):
    monkeypatch.setenv("IBKR_MCP_TRANSPORT", value)
    assert load_settings().mcp_transport == "http"


@pytest.mark.parametrize("value", ["streamable-http", "sse", "tcp", "https"])
def test_unknown_transport_refuses_to_start(monkeypatch, value):
    """Plausible wrong answers, not nonsense: "streamable-http" is what the MCP
    SDK calls this transport and is the mistake somebody will actually make."""
    monkeypatch.setenv("IBKR_MCP_TRANSPORT", value)
    with pytest.raises(ValueError) as exc:
        load_settings()
    # The message has to name the alternatives, since the whole point is that
    # the operator is holding a value that looked right.
    assert value in str(exc.value)
    for transport in TRANSPORTS:
        assert transport in str(exc.value)


def test_empty_transport_is_not_an_error(monkeypatch):
    """An unset variable and one set to nothing are the same intent. Service
    managers and `.env` files produce the second constantly."""
    monkeypatch.setenv("IBKR_MCP_TRANSPORT", "")
    assert load_settings().mcp_transport == "stdio"


def test_ib_host_and_port_do_not_leak_into_the_listener(monkeypatch):
    """The two pairs are for two different machines and must stay independent.

    A remote deployment sets IBKR_HOST to the Gateway and IBKR_MCP_HOST to the
    interface Caddy reaches; if either read the other, the server would bind
    the Gateway's address or dial its own listener.
    """
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    monkeypatch.setenv("IBKR_PORT", "4001")
    cfg = load_settings()
    assert (cfg.host, cfg.port) == ("10.0.0.5", 4001)
    assert (cfg.mcp_host, cfg.mcp_port) == ("127.0.0.1", 8765)

    monkeypatch.setenv("IBKR_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("IBKR_MCP_PORT", "9000")
    cfg = load_settings()
    assert (cfg.host, cfg.port) == ("10.0.0.5", 4001)
    assert (cfg.mcp_host, cfg.mcp_port) == ("0.0.0.0", 9000)
