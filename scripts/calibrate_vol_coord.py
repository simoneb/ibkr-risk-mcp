"""Refit the vol_coord damping against your own Risk Navigator.

``vol_coord`` reproduces IB's volatility-coordinated model. Its asymmetry — a
fall moves volatility ten times as hard as a rise — is documented by IB. Its
term-structure damping ``VR(t)`` is not: IB says only that the function exists
and is decreasing. The value this server ships was fitted to one Risk Navigator
screenshot, on one account, from nine points read off a chart by eye. It has no
claim on your book, and this script is how you replace it.

What to do:

1. In TWS, open Risk Navigator's risk graph on the Equity tab, so the curve
   excludes FX and rates the way ``scope='equity'`` does here.
2. Read the **Vol.Coord.** curve — the one that responds to volatility, not the
   constant-volatility line — at four or more shocks spread across the range you
   care about. Bunching them near the money constrains nothing.
3. Feed them in. The double dash is required: every shock starts with a minus
   sign, and argparse would otherwise read them as flags::

       uv run python scripts/calibrate_vol_coord.py -- \\
           -0.05=-8000 -0.10=-22000 -0.15=-31500 -0.20=-28000 -0.25=-12000

The fit comes back with the residual at every point and, more usefully, with
what to distrust about it: the tenor range your positions actually constrain,
and the most extreme volatility the fitted decay produces. A decay that
reproduces the curve by pricing a wing at 150% has fitted the chart rather than
the market, and it says so.

Nothing here trades, quotes or writes: it reads positions and prices locally.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from ibkr_risk_mcp import marketdata as MD  # noqa: E402
from ibkr_risk_mcp import stress as S  # noqa: E402
from ibkr_risk_mcp.connection import connection  # noqa: E402


def read_point(raw: str) -> tuple[float, float]:
    """``-0.20=-28000`` into ``(-0.20, -28000.0)``."""
    try:
        shock, pnl = raw.split("=")
        return float(shock), float(pnl)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{raw!r} is not SHOCK=PNL, e.g. -0.20=-28000. The shock is a fraction, "
            "not a percent."
        ) from exc


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "points",
        nargs="+",
        type=read_point,
        metavar="SHOCK=PNL",
        help="Readings off Risk Navigator's Vol.Coord. curve, e.g. -0.20=-28000. "
        "Put -- before them: they begin with a minus and would parse as flags.",
    )
    parser.add_argument(
        "--scope",
        default="equity",
        choices=list(S.SCOPES),
        help="Match Risk Navigator's tab. Its Equity tab is 'equity', the default.",
    )
    args = parser.parse_args()

    targets = dict(args.points)
    if min(targets) >= 0:
        print(
            "None of these shocks is negative. The volatility response is asymmetric and "
            "almost all of it is on the downside; a fit from the upside alone will not "
            "constrain the damping.",
            file=sys.stderr,
        )
        return 2

    await connection.get()
    holdings = await MD.load_holdings(with_greeks=True)
    units = [S.unit_from_holding(h) for h in holdings]
    cfg = S.StressConfig(shocks=sorted(targets), scope=args.scope)
    cfg.validate()

    result = S.calibrate_vol_coord(units, cfg, targets)

    print(f"\nvol_coord_decay = {result['decay']}      RMS = {result['rms']:,.0f} USD\n")
    print(f"  {'shock':>7} {'RN target':>12} {'model':>12} {'residual':>11}")
    for row in result["points"]:
        print(
            f"  {row['shock']:>+7.1%} {row['target']:>12,.0f} {row['model']:>12,.0f} "
            f"{row['residual']:>+11,.0f}"
        )

    extreme = result["mostExtremeVol"]
    if extreme:
        print(
            f"\n  most extreme volatility at {extreme['atShock']:+.0%}: {extreme['label']} "
            f"{extreme['impliedVolBefore']:.1%} -> {extreme['impliedVolAfter']:.1%}"
        )
    if result["calibratedToYears"] is not None:
        print(
            f"  tenors constrained: out to {result['calibratedToYears']:.3f} years "
            f"({len(result['tenorsCovered'])} distinct)"
        )

    for warning in result["warnings"]:
        print(f"\n  WARNING: {warning}")

    print(
        "\nTo use it, pass vol_coord_decay="
        f"{result['decay']} and vol_coord_calibrated_to_years="
        f"{result['calibratedToYears']} to stress_curve — the second is what makes the "
        "engine warn instead of extrapolating in silence.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
