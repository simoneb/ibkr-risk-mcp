"""ibkr-risk-mcp — Interactive Brokers portfolio risk over MCP.

Fills the one gap the official IBKR connector leaves: model greeks, IB's
implied volatility surface, what-if margin, and a local stress engine that
rebuilds the P&L curve across underlying shocks. Positions, balances, orders
and prices come from the official connector; nothing here duplicates them.
"""

__version__ = "0.1.0"
