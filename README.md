# ibkr-risk-mcp

MCP server exposing **Interactive Brokers' portfolio risk**: IB's model greeks, IB's implied volatility surface, IB's what-if margin, and a local stress engine that rebuilds the P&L-versus-underlying curve and finds its trough.

It deliberately does **not** duplicate the official IBKR connector. Positions, balances, orders, trades, performance, allocation, spot and historical prices, option chains, watchlists and alerts all come from there. This server exists to fill the one gap that connector leaves — risk analysis — and nothing else.

The questions it is built to answer:

- where is the trough of the portfolio's P&L curve across underlying shocks, at constant volatility?
- if I add N puts at strike K expiring E, where does that trough move to?
- how much margin does this hypothetical structure need, now and under stress?
- what does IB's volatility surface look like for this underlying?

Risk Navigator's risk model is not exposed by any API, so the strategy is to pull the *inputs* from IB — per-contract implied volatility, model greeks, what-if margin — and rebuild the curves locally, rather than trying to read Risk Navigator itself.

## Prerequisites

1. **TWS or IB Gateway running and logged in** on this machine. The server talks to its local socket; it never reaches IBKR over the internet.
2. **The API enabled.** File → Global Configuration → API → Settings → tick **Enable ActiveX and Socket Clients**. Until you do, TWS opens no port at all and `check_connection` reports `not_listening`.
3. **Market data** for the instruments you hold. Model greeks come from IB's own option model, so a contract the account cannot price has no greeks and is reported under `missing`. Contrary to what is widely repeated, **delayed data does carry model greeks** — verified against live TWS, where an unsubscribed account got nothing from market data types 1 and 2 and implied volatilities from type 3. If you lack the subscription, set `IBKR_MARKET_DATA_TYPE=3` and read the numbers as a quarter of an hour old.
4. For `whatif_order` only: **Read-Only API must be off** in that same TWS screen. That setting blocks what-if orders too.

Default ports: `7496` TWS live, `7497` TWS paper, `4001` Gateway live, `4002` Gateway paper. If the configured one is dead, `check_connection` scans all four and tells you which is answering.

## Install

```json
{
  "mcpServers": {
    "ibkr-risk": {
      "command": "uvx",
      "args": [
        "--from",
        "https://github.com/simoneb/ibkr-risk-mcp/archive/refs/tags/<TAG>.tar.gz",
        "ibkr-risk-mcp"
      ],
      "env": {
        "IBKR_PORT": "7496",
        "IBKR_CLIENT_ID": "17",
        "IBKR_ENABLE_WHATIF": "false"
      }
    }
  }
}
```

| Client | File |
|---|---|
| Claude Desktop | `%APPDATA%\Claude\claude_desktop_config.json`, or `~/Library/Application Support/Claude/` |
| Claude Code | `claude mcp add` |
| Cursor | `~/.cursor/mcp.json`, or `.cursor/mcp.json` per project |
| VS Code | `.vscode/mcp.json`, keyed under `servers` with `"type": "stdio"` |
| Codex CLI | `~/.codex/config.toml` |

Pin a tag rather than a branch, and prefer the archive URL over `git+https://…` — the git form needs `git` on PATH, and some clients hand the server too small an environment to find one.

For a local checkout:

```
uv venv && uv pip install -e ".[dev]"
uv run python -m ibkr_risk_mcp.server
```

As a Claude Desktop extension, `manifest.json` surfaces host, port, client id, account, market data type, risk-free rate and the what-if gate as a settings form, so none of them need editing by hand.

## Environment variables

| Variable | Default | |
|---|---|---|
| `IBKR_HOST` | `127.0.0.1` | Where TWS listens. A remote host must also be in the API's Trusted IPs |
| `IBKR_PORT` | `7496` | 7496/7497 TWS live/paper, 4001/4002 Gateway live/paper |
| `IBKR_CLIENT_ID` | `17` | Must differ from every other script on this TWS. **Never 0** — TWS reserves that for orders placed by hand |
| `IBKR_ACCOUNT` | — | **Required** on a multi-account login. Without it every position/account tool refuses rather than combining accounts |
| `IBKR_MARKET_DATA_TYPE` | `1` | 1 live, 2 frozen, 3 delayed, 4 delayed-frozen. 3 works and does carry greeks |
| `IBKR_WHATIF_TIMEOUT` | `5` | Seconds to wait for a what-if reply. IB sometimes never sends one |
| `IBKR_ENABLE_WHATIF` | `false` | The gate on `whatif_order`. Also decides whether the connection itself is opened read-only |
| `IBKR_RISK_FREE_RATE` | `0.04` | Used to discount and to carry spot to the forward in the local repricing |
| `IBKR_GREEKS_TIMEOUT` | `4` | Seconds to wait for greeks on one contract. Short by design — IB answers fast or never, and explicit refusals cut the wait short anyway |
| `IBKR_MAX_MKT_DATA_LINES` | `40` | Concurrent market data subscriptions. IB allows about 50 |
| `IBKR_CONNECT_TIMEOUT` | `6` | Seconds for the API handshake |

Copy `.env.example` to `.env` for local runs.

## Tools

| Tool | |
|---|---|
| `check_connection` | Is TWS reachable, and if not, which of the four failure modes it is |
| `get_margin_summary` | NetLiq, margin requirements, available funds and excess liquidity — **per segment** |
| `get_position_greeks` | IB's model greeks for every option position, with both expiry dates |
| `get_vol_surface` | IB's implied volatility grid for an underlying, by expiry and strike |
| `stress_portfolio` | The P&L curve across underlying shocks, and its trough |
| `stress_whatif` | The same curve with hypothetical legs added: base, with-legs, and the difference |
| `whatif_order` | IB's margin impact of a structure, per leg and cumulatively. Needs the gate below |

Each tool carries the protocol's annotations, so a client can group them by permission. Six are marked read-only; `whatif_order` is not, because it puts something on IB's order channel even though nothing is routable.

## The what-if gate

`whatif_order` works **only** with `IBKR_ENABLE_WHATIF=true`. Otherwise it sends nothing and returns `success: false` with `blocked: true`.

An order carrying `whatIf=True` is evaluated by IB's margin engine and discarded — never routed, never acknowledged as live, never in the order book. The gate exists anyway, for two reasons. It is the only thing in this server that touches the order path at all, so being read-only should be provable rather than asserted; and with the gate closed the ib_async connection itself is opened in read-only mode, which makes an order impossible below this server as well as inside it.

**No tool here can submit a live order.** Those belong somewhere else.

## The traps this server handles

These are not hypothetical; each one produces a confidently wrong number if you skip it.

**Expiry dates are reported twice, and AM settlement does not always move the date.** For AM-settled contracts — the quarterly ES options, trading class `ES` — some TWS builds report `lastTradeDateOrContractMonth` as the day *before* settlement, so an 18 December expiry shows as the 17th while a PM weekly expiring the same morning shows as the 18th; time to expiry taken from the last trading day is then a day short.

But that is not universal, and assuming it is would introduce the error it was meant to prevent. Measured on TWS server 178: the December ES quarterly reports `20261218` — the correct date — with the AM settlement visible only in `lastTradeTime='08:30:00'` against the weekly's `15:00:00`. IB's own `ContractDetails.realExpirationDate` exists for exactly this ambiguity and is preferred wherever the details have been fetched; the class-and-date heuristic is only the fallback. Every tool returns both `lastTradeDate` and `settlementDate` plus an `amSettled` flag, and everything downstream uses settlement.

**The underlying is not implied by the expiry — and two options expiring the same morning need not share one.** Confirmed against live TWS: the 18 December 2026 `ES` quarterly is written on ESZ6, while the `EW3` weekly expiring *that same day* is written on ESH7. The 30 September end-of-month options are on ESZ6, not ESU6. `underConId` comes from the contract details and is never inferred from a date.

**Portfolio contracts arrive unqualified.** The contracts on IB's `PortfolioItem` come back with `exchange` empty, and `reqMktData` on a contract without an exchange returns nothing at all — no ticks, no error. This is why a live book returned **no model greeks at all for any option position** while a surface request on the same underlying worked perfectly: those contracts had come from `reqContractDetails` already qualified. Every position is requalified before its data is requested.

**`impliedVol` and `undPrice` arrive in separate ticks.** Taking the greeks the moment a volatility appears leaves the underlying unset a good fraction of the time — a sizeable minority of positions on a measured live run — and without a forward there is nothing to reprice against, so the position silently drops out of the curve. Both fields are waited for, and a missing forward is backfilled from another position on the same underlying before it is fetched again.

**A refusal is an answer; don't wait it out.** IB replies to a market data request in about a second or not at all, and when it is "not at all" it usually says so at once with error 354 or 10091. Those are watched for and end the wait immediately. Without that, an unentitled book pays the full timeout on every contract — the difference between a check taking 30 seconds and taking minutes.

**Where IB won't publish greeks, ask it to calculate instead.** `reqCalcImpliedVolatility` takes the option and underlying prices as *inputs*, so it is a model call rather than a subscription, and it uses IB's own American-exercise model. It succeeds on contracts whose streaming greeks were refused. Those rows are stamped with their `source` and never passed off as IB's streaming output; if even that fails, the volatility is implied locally and stamped differently again.

**A bare root is ambiguous.** `ES` is the E-mini S&P 500 future **and** Eversource Energy on NYSE. With no `sec_type` the stock wins, and you get a plausible-looking volatility surface with 75-dollar strikes. Every resolution reports what else the symbol matched, and `get_vol_surface` returns the contract it actually used.

**IB requires `transmit=True` on a what-if order.** With `transmit=False` — the intuitive choice for something meant not to trade — TWS rejects it with error 321 and, because it rejects rather than answers, the call never returns at all. `whatIf=True` is what keeps the order off the market; `transmit` has nothing to do with it. Every what-if here is also bounded by a timeout, because IB not answering is an ordinary outcome.

**Volatility has a term structure.** ES at 139 days can sit near 15% at the money while the front month prints 12%. Using one ATM number for every tenor understates a long-dated position badly. The surface interpolates per tenor, in total variance rather than in volatility, which is also what keeps it free of calendar arbitrage.

**Bonds are quoted as a percentage of nominal.** 50,000 nominal at 97.85 is worth 48,925, not 4,892,500 — `quantity × price` is wrong by 100×. Position values come from IB's own `marketValue`, and the reconciliation below is what proves it worked.

**Futures have no market value to add.** Variation margin settles daily, so a future contributes its unrealised P&L to NetLiquidation, not its notional. Adding the notional instead moves the total by a quarter of a typical account.

**Two accounts must never be added together.** `ib.portfolio("")` and `ib.accountValues("")` do not mean "the default account", they mean *all of them*. On a multi-account login with no `IBKR_ACCOUNT` set, that would combine two portfolios and reconcile the total against one account's NetLiquidation — a wrong answer wearing the shape of a right one. Every tool that reads positions or account values refuses until an account is chosen, and names the candidates. Related: IB permits only **one** `reqAccountUpdates` subscription at a time, so asking about a second account silently cancels the first; only the resolved account is ever subscribed.

**Reconciliation.** `stress_portfolio` rebuilds the portfolio at zero shock — cash, plus the securities' market value, plus the futures' unrealised P&L — and compares it against `NetLiquidation`. A residual over 1% returns `reconciled: false` with the residual attached. Nothing derived from a portfolio that does not reconcile should be presented as fact.

**A minimum at the edge of the range is not a trough.** If the curve is still falling at −30% the engine says so, rather than reporting the boundary as the worst case. Some portfolios simply keep losing past the edge of the window.

## Known limitations

**Risk Navigator's volatility shock model is not public.** It is not exposed by any API and IB does not document it. `sticky_strike` — each strike keeps its current implied volatility — is the approximation corresponding to Risk Navigator's default blue curve. Expect the *shape* to match and the last few percent not to. Do not present a number from this server as "what Risk Navigator says"; it is what a documented model, fed IB's own volatilities, says.

**One shock, all underlyings.** Every underlying is moved by the same percentage at once, which is Risk Navigator's own default assumption. `betas` scales equity positions; there is no correlation matrix, and there is no per-underlying scenario.

**Volatility surface interpolation is local.** Under `sticky_moneyness` the smile is built from the strikes the portfolio actually holds. Fewer than three on one expiry does not define a slope, and that expiry falls back to `sticky_strike` with a warning rather than being given an invented flat smile. `fetch_skew=true` pulls neighbouring strikes from IB instead, at the cost of more market data requests.

**The local repricing is European; most of these options are American.** Black-76 has no early exercise, while equity options and CME futures options both do. The gap is negligible out of the money and real once an option is in the money. Measured against live IB data: a 75-strike put with spot at 71.94 priced at 4.51 locally against IB's 4.65, a 2.9% shortfall that is the early-exercise premium and nothing else. It is reported per position as `modelVsMarket` rather than hidden, and it means the curve slightly *understates* losses deep in the money.

**Rates are an input, not a measurement.** IB does not publish the rate behind its own model. `IBKR_RISK_FREE_RATE` moves option values by little over the horizons this server deals with, but it is not zero either, and it is part of the `modelVsMarket` residual above.

**Currency.** Everything is summed in the account's base currency as IB reports it. A portfolio with positions IB values in another currency will show up as a reconciliation residual rather than being converted.

**No exchange holiday calendar.** The AM-settlement shift moves to the next weekday. An expiry the day after a holiday would be off by one, and that shows in `settlementDate` rather than hiding.

## Tests

```
uv run pytest -q                      # unit tests, no TWS needed
uv run python scripts/smoke_test.py   # end-to-end, needs TWS
```

The unit tests run the whole repricing layer against recorded JSON fixtures in `tests/fixtures/`, with the valuation date pinned, so a pricing bug is distinguishable from a market data problem and the numbers do not drift as time passes. The fixture portfolio holds an AM-settled quarterly and a PM-settled weekly on the same morning, a future, an equity and a bond — every trap above, in one file.

`scripts/smoke_test.py` exercises the live path: connection, greeks with a count of any missing `modelGreeks`, a surface, `stress_portfolio` from −30% to +30% in 1% steps with the reconciliation check, `stress_whatif`, and a `whatif_order` on a single deeply out-of-the-money leg. Point `.env` at the paper port to try it safely.

## Layout

```
src/ibkr_risk_mcp/
  server.py       MCP tool definitions — the docstrings are the interface
  connection.py   ib_async lifecycle, one connection, four failure modes
  marketdata.py   greeks, vol surface, rate limiting, portfolio snapshot
  pricing.py      Black-76 / Black-Scholes, skew and surface interpolation
  stress.py       the stress and what-if engine
  margin.py       whatif_order and the segmented margin summary
  contracts.py    expiry and underlying normalisation
scripts/smoke_test.py
tests/
```

`pricing.py` and `contracts.py` have no IB dependency at all, which is what makes them testable.
