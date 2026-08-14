"""Fixtures that stand in for TWS.

The whole point of these is that the repricing layer can be exercised without a
connection: a pricing bug and a market-data problem look identical through a
live socket, and only one of them is this project's to fix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class FakeContract:
    """The subset of ib_async's Contract that this codebase reads."""

    conId: int = 0
    symbol: str = ""
    localSymbol: str = ""
    secType: str = ""
    right: str = ""
    strike: float = 0.0
    lastTradeDateOrContractMonth: str = ""
    tradingClass: str = ""
    multiplier: str = ""
    exchange: str = ""
    currency: str = "USD"


@dataclass
class FakeAccountValue:
    tag: str
    value: str
    currency: str = "BASE"
    account: str = "DU1234567"


@dataclass
class FakeIB:
    """Just enough of ib_async's IB for the reconciliation path."""

    values: list[FakeAccountValue] = field(default_factory=list)

    def accountValues(self, account: str = "") -> list[FakeAccountValue]:
        return self.values

    def managedAccounts(self) -> list[str]:
        return ["DU1234567"]


def load_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def snapshot() -> dict[str, Any]:
    return load_json("portfolio.json")


@pytest.fixture
def holdings(snapshot) -> list:
    from ibkr_risk_mcp import contracts as C
    from ibkr_risk_mcp.marketdata import Holding

    out = []
    for row in snapshot["positions"]:
        contract = FakeContract(
            **{
                k: row[k]
                for k in (
                    "conId",
                    "symbol",
                    "localSymbol",
                    "secType",
                    "lastTradeDateOrContractMonth",
                    "tradingClass",
                    "multiplier",
                    "exchange",
                    "currency",
                )
                if k in row
            },
            right=row.get("right", ""),
            strike=row.get("strike", 0.0),
        )
        out.append(
            Holding(
                contract=contract,
                position=row["position"],
                market_price=row["marketPrice"],
                market_value=row["marketValue"],
                average_cost=row["averageCost"],
                unrealized_pnl=row["unrealizedPNL"],
                asset_class=C.asset_class(row["secType"]),
                multiplier=C.contract_multiplier(contract),
                greeks=row.get("greeks"),
            )
        )
    return out


@pytest.fixture
def fake_ib(snapshot) -> FakeIB:
    return FakeIB(
        values=[
            FakeAccountValue(tag=tag, value=str(value))
            for tag, value in snapshot["account"].items()
        ]
    )
