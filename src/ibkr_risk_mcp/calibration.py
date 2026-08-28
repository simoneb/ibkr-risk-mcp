"""Where a fitted ``vol_coord`` decay lives between sessions.

``vol_coord_decay`` is the one number in this server that IB neither publishes
nor the account measures: it is a fit, and the value shipped was fitted to
somebody else's Risk Navigator, from nine points read off a chart by eye.
Refitting it against your own is therefore not a tuning step. It is the thing
that turns a vol_coord curve from indicative into yours.

Until now that refit lived in ``scripts/calibrate_vol_coord.py``, and its
result had to be carried by hand into every later call — which meant that in
practice it was not carried at all and every session went back to the factory
number. This module is the other half: the fit is written once and every later
run reads it.

It is the only thing in this server that writes to disk. The file is small,
plain JSON and entirely inspectable, and what it records is not only the
number but what the number was fitted against — targets, residuals, date,
account. A decay with no provenance is precisely the problem the calibration
exists to solve, so one is not written without it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Where the fit goes unless ``IBKR_CALIBRATION_FILE`` says otherwise. Under
#: the home directory rather than beside the code: a calibration belongs to an
#: account and a book, not to a checkout, and it must survive the server being
#: reinstalled from a new tag by ``uvx``.
DEFAULT_PATH = Path.home() / ".ibkr-risk-mcp" / "vol_coord.json"


def path() -> Path:
    """The calibration file, read from the environment on every call.

    Not cached: a test — and a user with two accounts — has to be able to point
    this somewhere else without reimporting the module.
    """
    raw = os.environ.get("IBKR_CALIBRATION_FILE")
    return Path(raw).expanduser() if raw and raw.strip() else DEFAULT_PATH


def load() -> dict[str, Any] | None:
    """The stored fit, or ``None`` if there is none.

    A file that cannot be read or does not carry a usable decay returns
    ``None`` and logs it. The alternative — raising — would take down a stress
    run over a stale file that has nothing to do with the question being asked,
    and the fallback is the factory decay, which is exactly what the caller
    would have had anyway.
    """
    file = path()
    try:
        record = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable vol_coord calibration at %s: %s", file, exc)
        return None
    decay = record.get("decay") if isinstance(record, dict) else None
    if not isinstance(decay, (int, float)) or not decay > 0:
        log.warning("ignoring vol_coord calibration at %s: no usable decay", file)
        return None
    return record


def save(record: dict[str, Any]) -> Path:
    """Write the fit, creating the directory if it is not there.

    Whole-file, not merged: a calibration is one fit against one reading of one
    Risk Navigator, and half of an old one beside half of a new one would be a
    number nobody ever measured.
    """
    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file


def decay(default: float) -> float:
    """The calibrated decay if one was stored, otherwise the factory value."""
    record = load()
    return float(record["decay"]) if record else default


def calibrated_to_years(default: float) -> float:
    """How far out the stored fit was actually constrained.

    Falls back to the shipped figure rather than to infinity: a calibration
    written before this field existed still knows nothing about long tenors,
    and claiming otherwise would silence the one warning that matters most.
    """
    record = load()
    if not record:
        return default
    value = record.get("calibratedToYears")
    return float(value) if isinstance(value, (int, float)) and value > 0 else default


def provenance() -> str | None:
    """One line naming what the stored decay was fitted against, for the
    ``assumptions`` block. ``None`` when nothing is stored."""
    record = load()
    if not record:
        return None
    when = record.get("fittedAt") or "an unrecorded date"
    points = len(record.get("points") or [])
    account = record.get("account")
    return (
        f"calibrated on {when}"
        + (f" against account {account}" if account else "")
        + (f", from {points} Risk Navigator point(s)" if points else "")
        + f", stored at {path()}"
    )
