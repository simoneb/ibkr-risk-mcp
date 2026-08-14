"""ibkr-risk-mcp — Interactive Brokers portfolio risk over MCP.

Fills the one gap the official IBKR connector leaves: model greeks, IB's
implied volatility surface, what-if margin, and a local stress engine that
rebuilds the P&L curve across underlying shocks. Positions, balances, orders
and prices come from the official connector; nothing here duplicates them.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    #: Read from the installed distribution rather than written here. The
    #: release workflow bumps pyproject.toml and manifest.json and checks the
    #: two agree; a third copy in the source would drift silently the first
    #: time someone released without remembering it existed.
    __version__ = version("ibkr-risk-mcp")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"
