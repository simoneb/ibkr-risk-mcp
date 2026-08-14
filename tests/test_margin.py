"""Parsing IB's what-if reply.

Every margin figure arrives as a string, and "not applicable" arrives as
1.7976931348623157E308. A number that quietly became None reads downstream as
zero impact, which is the one wrong answer that looks reassuring — so the
distinction between "no value" and "could not read the value" is kept.
"""

from dataclasses import FrozenInstanceError, replace

import pytest

from ibkr_risk_mcp import margin as M
from ibkr_risk_mcp.contracts import Leg


class FakeOrderState:
    def __init__(self, **kw):
        for field in M._WHATIF_FIELDS:
            setattr(self, field, kw.pop(field, "0"))
        self.commission = kw.pop("commission", "1.25")
        self.commissionCurrency = kw.pop("commissionCurrency", "USD")
        self.warningText = kw.pop("warningText", "")
        self.status = kw.pop("status", "PreSubmitted")
        for k, v in kw.items():
            setattr(self, k, v)


class TestParsing:
    def test_strings_become_floats(self):
        assert M._parse("12345.67") == (12345.67, None)

    def test_ibs_sentinel_is_no_value_not_a_number(self):
        value, problem = M._parse("1.7976931348623157E308")
        assert value is None and problem is None

    def test_an_unreadable_value_is_reported_rather_than_swallowed(self):
        value, problem = M._parse("n/a")
        assert value is None
        assert "not a number" in problem

    def test_empty_is_absent(self):
        assert M._parse("") == (None, None)


class TestOrderState:
    def test_every_field_is_carried_through(self):
        state = FakeOrderState(
            initMarginBefore="100000",
            initMarginAfter="112500",
            initMarginChange="12500",
            warningText="Your order size exceeds the size limit",
        )
        out = M.order_state_dict(state)
        assert out["initMarginChange"] == 12500.0
        assert out["commission"] == 1.25
        assert out["warningText"] == "Your order size exceeds the size limit"
        assert "parseProblems" not in out

    def test_parse_failures_are_collected_and_named(self):
        out = M.order_state_dict(FakeOrderState(initMarginChange="unavailable"))
        assert out["initMarginChange"] is None
        assert any("initMarginChange" in p for p in out["parseProblems"])


class TestGate:
    """Settings is frozen on purpose — nothing inside the process should be
    able to open the gate — so these swap the whole object rather than a
    field."""

    @staticmethod
    def _with_gate(monkeypatch, enabled: bool) -> None:
        monkeypatch.setattr(M, "settings", replace(M.settings, enable_whatif=enabled))

    def test_a_closed_gate_sends_nothing(self, monkeypatch):
        self._with_gate(monkeypatch, False)
        with pytest.raises(M.WhatIfDisabled, match="Nothing was sent"):
            M.require_whatif_enabled()

    def test_an_open_gate_passes(self, monkeypatch):
        self._with_gate(monkeypatch, True)
        M.require_whatif_enabled()

    def test_the_setting_cannot_be_flipped_in_place(self):
        with pytest.raises(FrozenInstanceError):
            M.settings.enable_whatif = True  # type: ignore[misc]


class TestOrderShape:
    def test_a_whatif_order_must_transmit(self):
        """Counter-intuitive and load-bearing. With transmit=False TWS rejects
        the request outright — error 321, "What-If order should have transmit
        flag set to TRUE" — and because it rejects rather than answers, the
        call never returns at all. Found against live TWS.

        Nothing about this routes an order: whatIf=True is what makes IB
        evaluate and discard it.
        """
        leg = Leg(conid=1, action="BUY", quantity=2)
        order = M._order_for(leg, _contract())
        assert order.whatIf is True
        assert order.transmit is True

    def test_market_order_by_default_limit_when_priced(self):
        leg = Leg(conid=1, action="SELL", quantity=1)
        assert M._order_for(leg, _contract()).orderType == "MKT"
        priced = M._order_for(leg, _contract(), limit_price=12.3456)
        assert priced.orderType == "LMT" and priced.lmtPrice == 12.3456

    def test_quantity_is_unsigned_and_the_side_carries_the_direction(self):
        order = M._order_for(Leg(conid=1, action="SELL", quantity=3), _contract())
        assert order.totalQuantity == 3 and order.action == "SELL"


def _contract():
    from .conftest import FakeContract

    return FakeContract(conId=1, symbol="ES", secType="FOP", exchange="CME")


class TestCombo:
    def _pair(self, conid, exchange="CME", currency="USD", action="BUY", qty=1):
        from .conftest import FakeContract

        contract = FakeContract(
            conId=conid, symbol="ES", secType="FOP", exchange=exchange, currency=currency
        )
        return Leg(conid=conid, action=action, quantity=qty), contract

    def test_legs_on_one_exchange_make_a_bag(self):
        bag = M._combo([self._pair(1), self._pair(2, action="SELL", qty=2)])
        assert bag.secType == "BAG"
        assert [(l.conId, l.ratio, l.action) for l in bag.comboLegs] == [
            (1, 1, "BUY"),
            (2, 2, "SELL"),
        ]

    def test_legs_across_exchanges_have_no_combo(self):
        """IB has nothing to evaluate here, and returning the legs separately
        again under a 'cumulative' heading would be a wrong answer wearing the
        right label."""
        assert M._combo([self._pair(1, exchange="CME"), self._pair(2, exchange="CBOE")]) is None

    def test_legs_across_currencies_have_no_combo_either(self):
        assert M._combo([self._pair(1), self._pair(2, currency="EUR")]) is None
