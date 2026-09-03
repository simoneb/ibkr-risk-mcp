# ibkr-risk-mcp

MCP server exposing **Interactive Brokers' portfolio risk**: IB's model greeks, IB's implied volatility surface, IB's what-if margin, and a local stress engine that rebuilds the P&L-versus-underlying curve and finds its trough.

It deliberately does **not** duplicate the official IBKR connector. Positions, balances, orders, trades, performance, allocation, spot and historical prices, option chains, watchlists and alerts all come from there. This server exists to fill the one gap that connector leaves — risk analysis — and nothing else.

The questions it is built to answer:

- where is the trough of the portfolio's P&L curve across underlying shocks, at constant volatility?
- and how much of that answer is the constant-volatility assumption itself?
- if I add N puts at strike K expiring E, where does that trough move to?
- how much margin does this hypothetical structure need, now and under stress?
- what does IB's volatility surface look like for this underlying?

Risk Navigator's risk model is not exposed by any API, so the strategy is to pull the *inputs* from IB — per-contract implied volatility, model greeks, what-if margin — and rebuild the curves locally, rather than trying to read Risk Navigator itself.

## Prerequisites

1. **TWS or IB Gateway running and logged in.** The server talks to its socket directly; it never reaches IBKR over the internet. Usually that is the same machine — for a Gateway on a host the server reaches over the network instead, see [docs/remote.md](docs/remote.md).
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

### Over HTTP

The default transport is stdio — a server the client launches as a subprocess — and everything above assumes it. `IBKR_MCP_TRANSPORT=http` serves streamable HTTP instead, for a server running somewhere the client cannot launch it: beside an IB Gateway on a host that stays up, behind a reverse proxy, reached as a custom connector.

That mode adds a bearer-token boundary (OAuth 2.1 resource server, tokens issued by an identity provider you choose), a service unit, and a set of timeouts measured rather than guessed. All of it is in **[docs/remote.md](docs/remote.md)**; none of it changes the stdio path, which remains the default and is unaffected.

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
| `IBKR_CALIBRATION_FILE` | `~/.ibkr-risk-mcp/vol_coord.json` | Where `calibrate_vol_coord` stores the fitted `vol_coord_decay`. The only file this server writes |

Copy `.env.example` to `.env` for local runs.

The HTTP transport and its authentication add `IBKR_MCP_*` variables — transport, listen address, path, and the bearer-token settings. They are documented in `.env.example` and in [docs/remote.md](docs/remote.md); none of them affect a stdio run.

## Tools

| Tool | |
|---|---|
| `check_connection` | Is TWS reachable, and if not, which of the four failure modes it is |
| `get_margin_summary` | NetLiq, margin requirements, available funds and excess liquidity — **per segment** |
| `get_position_greeks` | IB's model greeks for every option position, with both expiry dates |
| `get_vol_surface` | IB's implied volatility grid for an underlying, by expiry and strike |
| `stress_portfolio` | The P&L curve across underlying shocks, and its trough |
| `stress_curve` | The same curve under several volatility regimes and valuation dates at once — risk-graph data, with the vol assumption as a visible parameter |
| `stress_whatif` | The same curve with hypothetical legs added: base, with-legs, and the difference |
| `calibrate_vol_coord` | Refit the volatility-coordinated model against your own Risk Navigator, and keep the fit |
| `whatif_order` | IB's margin impact of a structure, per leg and cumulatively. Needs the gate below |

Each tool carries the protocol's annotations, so a client can group them by permission. Seven are marked read-only. `whatif_order` is not, because it puts something on IB's order channel even though nothing is routable; `calibrate_vol_coord` is not either, because it writes the fit to disk — neither of them changes anything in the account.

## Reading the curve: by symbol, by expiry, at another date

Every stress result carries the P&L broken down per shock. By default that breakdown is keyed on the **symbol**, which is the right unit for a book of many underlyings and the wrong one for a book running one underlying across many expiries: nine ES expiries all land under a single `ES` key, and the operative question — *which expiry is holding the trough down, and which short do I buy back* — has to be reconstructed by hand from the position list.

`breakdown` changes the key. `expiry` groups on the option's **settlement** date (`ES 2026-10-30`), so a quarterly and a weekly that settle the same morning are one row rather than two names for one expiry. Positions with no expiry get a key naming their class — `ES (future)`, `AAPL (equity)` — so the breakdown still sums to the point's total and can be checked against it rather than trusted. `both` returns symbol and expiry, `none` neither, and `symbol` remains the default: the responses are already large, and a second dictionary at every one of twenty-six shocks is not free.

With an expiry breakdown the result also carries `troughByExpiry`, two columns per expiry that answer two different questions:

- `pnl` — that expiry's own worst point along the curve, which is what it can cost.
- `pnlAtPortfolioTrough` — what it contributes at the shock where the *account's* floor actually sits, which is what says whether closing it would move that floor.

They come apart, and the gap is the useful part. An expiry whose own minimum sits at −35% while the book troughs at −22% is not the one to buy back, and reading only the first column would nominate it.

**Valuation dates.** `date_offset_days` rolls the clock forward; `valuation_date` takes the ISO date instead, so "the curve at 30 September" does not have to be counted out by hand over a weekend. There is no calendar adjustment — the date is the date — and both forms mean the same thing: the P&L is still measured *from today*, at today's spot and today's implied volatilities, with time advanced. That is decay and the change in convexity that comes with it, not a forecast of where the market will be.

`stress_curve` takes a **family** of them, `date_offsets: [0, 3]` or `valuation_dates: [...]`, and crosses them with the volatility scenarios. That is not a convenience: comparing today against Monday used to take two calls, and the book and the market moved between them, so part of the difference between the two curves was not the three days at all. One call, one loading of positions and prices, and time is the only thing that changed. Each entry under `curves` carries its own `valuationDate` and `dateOffsetDays`; `name` stays the scenario's and `label` distinguishes the pair.

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

**Where IB won't publish greeks, the price is already in the portfolio.** Error 10091 — "requires additional subscription **for API**" — refuses the option's greeks *and* a quote on its underlying, so the obvious fallback of implying a volatility locally dies for want of a spot price. But a book holding GOOGL stock next to a GOOGL option already knows what GOOGL is worth: every position carries its own mark on the portfolio update, which arrives over the **account** channel and is gated by no market data entitlement at all. Underlying prices are taken from the account's own positions before a market data line is ever spent. Measured on a live account, this is the difference between two long puts priced and the same two held flat across every shock — worth 2,826 at a 15% fall and 4,414 at 20%, on a 138k account. A long option contributes a multiple of its premium under shock, so judging it by its market value understates it badly.

**`reqCalcImpliedVolatility` can take the connection down, so it is opt-in.** Asking IB to run its own American-exercise model on prices you supply is the better answer than implying a European volatility locally — when it works. Measured against live TWS: ib_async's request is answered with **error 320**, *"Error reading request. Please use 'Key=Value' format for Misc Options"*, and TWS then **closes the API connection**. A protocol error halfway through a portfolio load costs the whole load, which is far worse than the one contract it was trying to rescue. It is behind `IBKR_USE_IB_IMPLIED_VOL`, off by default, and disables itself for the rest of the process after a single failure. The local implication is stamped with its `source` and never passed off as IB's own either way.

**A bare root is ambiguous.** `ES` is the E-mini S&P 500 future **and** Eversource Energy on NYSE. With no `sec_type` the stock wins, and you get a plausible-looking volatility surface with 75-dollar strikes. Every resolution reports what else the symbol matched, and `get_vol_surface` returns the contract it actually used.

**IB requires `transmit=True` on a what-if order.** With `transmit=False` — the intuitive choice for something meant not to trade — TWS rejects it with error 321 and, because it rejects rather than answers, the call never returns at all. `whatIf=True` is what keeps the order off the market; `transmit` has nothing to do with it. Every what-if here is also bounded by a timeout, because IB not answering is an ordinary outcome.

**The volatility level does not move with the shock unless you make it.** Both vol modes decide *which* volatility a strike gets, not how high the surface sits: `sticky_strike` pins each strike to the volatility it holds today, `sticky_moneyness` slides a strike along the smile the portfolio already has. Neither raises the level, and `vol_bump` cannot either — it is flat along the shock axis by construction. So the default curve prices the move in the underlying and none of the move in volatility that comes with it, which for a net short option book is the optimistic half of the answer and can be the larger half. `vol_slope_down` puts it in: 1.0 adds one volatility point per 1% fall, so a −20% shock reprices at +20 points. It is applied as a parallel shift across every tenor — a real surface also steepens in a sell-off, and a 120-day volatility moves less than the front month — so it is an input to be chosen and stated, in the same class as `bond_duration_years`. Expect it to change the *shape* of the curve and not only its depth: on the measured fixture it deepens the −10% and −20% points while lifting the −30% tail, because the long wings pick up vega faster than the short body does.

**Volatility has a term structure.** ES at 139 days can sit near 15% at the money while the front month prints 12%. Using one ATM number for every tenor understates a long-dated position badly. The surface interpolates per tenor, in total variance rather than in volatility, which is also what keeps it free of calendar arbitrage.

**Bonds are quoted as a percentage of nominal.** 50,000 nominal at 97.85 is worth 48,925, not 4,892,500 — `quantity × price` is wrong by 100×. Position values come from IB's own `marketValue`, and the reconciliation below is what proves it worked.

**Futures have no market value to add.** Variation margin settles daily, so a future contributes its unrealised P&L to NetLiquidation, not its notional. Adding the notional instead moves the total by a quarter of a typical account.

**Two accounts must never be added together.** `ib.portfolio("")` and `ib.accountValues("")` do not mean "the default account", they mean *all of them*. On a multi-account login with no `IBKR_ACCOUNT` set, that would combine two portfolios and reconcile the total against one account's NetLiquidation — a wrong answer wearing the shape of a right one. Every tool that reads positions or account values refuses until an account is chosen, and names the candidates. Related: IB permits only **one** `reqAccountUpdates` subscription at a time, so asking about a second account silently cancels the first; only the resolved account is ever subscribed.

**Reconciliation.** `stress_portfolio` rebuilds the portfolio at zero shock — cash, plus the securities' market value, plus the futures' unrealised P&L — and compares it against `NetLiquidation`. A residual over 1% returns `reconciled: false` with the residual attached. Nothing derived from a portfolio that does not reconcile should be presented as fact.

**A minimum at the edge of the range is not a trough.** If the curve is still falling at −30% the engine says so, rather than reporting the boundary as the worst case. Some portfolios simply keep losing past the edge of the window.

## Known limitations

**Risk Navigator's volatility shock model is not public.** It is not exposed by any API and IB does not document it. `sticky_strike` — each strike keeps its current implied volatility — is the approximation corresponding to Risk Navigator's default blue curve. Expect the *shape* to match and the last few percent not to. Do not present a number from this server as "what Risk Navigator says"; it is what a documented model, fed IB's own volatilities, says.

**One shock, all underlyings — and only the equity ones.** Every underlying in scope is moved by the same percentage at once, which is Risk Navigator's own default assumption. There is no correlation matrix and no per-underlying scenario.

Because of that, `scope` defaults to `equity`: only equity underlyings are on the axis, and FX, rates and the rest are excluded outright and listed under `excluded` with their market value. This is not a refinement, it is what makes the number mean anything — one percentage applied to every underlying at once is nonsense off the equity axis, where a 20% shock on a currency future prices an exchange rate that has never traded there. On a live account a single CAD strangle was contributing −21,716 at −20% **and** −7,183 at +10%, against −29,027 and +2,408 for an entire ES campaign: it dominated both tails of a curve that was supposed to be about equities. TWS Risk Navigator draws the same line in its Equity tab, and once it is drawn here too the two curves agree to **34 dollars on 29,000** at a 15% fall. `scope='all'` restores the old behaviour.

The classification is a table plus one heuristic — a three-letter currency code on a FUT or FOP is IB's own naming for a currency future — not a deduction, because IB publishes no reliable asset class for futures and a bond or gold ETF quoted as `STK` lands in `equity` with no field to say otherwise. So the group is reported on every position and `risk_groups` overrides it per symbol.

`betas` scales that shock per symbol, and it reaches every class that responds to one — an option is repriced at its own beta-scaled move of its underlying, with strike, smile and convexity all measured at the forward it would actually reach, rather than having its P&L scaled after the fact. Keys are tried most specific first: local symbol (`ESZ6 P5800`), then root (`ES`), then underlying (`ESZ6`).

This is what lets a foreign underlying be stood down off an equity axis — a short EUR strangle is not a 20%-down position when the S&P falls 20%, and a single shock across every underlying says it is. But read what a beta does and does not do. It scales the underlying's move only: vega and theta are untouched, so a position at beta 0 still contributes P&L the moment `vol_bump` or `date_offset_days` is set, and `pnl_by_symbol` is the only clean exclusion. More importantly, standing a position down is not measuring it — an attenuated strangle carries its whole gap risk and none of that risk is anywhere on the curve. Every run that applies a beta other than 1 says so in `warnings`.

**Volatility surface interpolation is local.** Under `sticky_moneyness` the surface is built from the strikes the portfolio actually holds — one skew per expiry that has at least three of them, assembled into a surface and interpolated across tenors in **total variance**. An expiry too thin to define its own slope borrows its shape from the tenors that do; an underlying where no expiry defines one falls back to `sticky_strike` rather than being handed an invented flat smile. Either way it is said in `warnings`. `fetch_skew=true` pulls neighbouring strikes from IB instead, at the cost of more market data requests.

The surface supplies the *change* in volatility as a strike slides to new moneyness, not the level. The level stays IB's own per-contract implied volatility, which comes out of a model that prices American exercise and is a better number than any fit through it. Reading the level off the surface would also break the curve's zero: a strike whose shape was borrowed from another tenor would not get its own volatility back at zero shock. Every result carries `volSurfaceUsed`, the quotes the repricing actually read — **if it is empty under `sticky_moneyness`, no smile was built, every option silently fell back to `sticky_strike`, and the result is not the model you asked for.**

**The volatility response is IB's own model, and both curves are validated against Risk Navigator.** Risk Navigator draws two lines: a constant-volatility curve, and one it labels `Vol.Coord.` where volatility moves as a deterministic function of the price shock. IB documents that second model's shape — the nominal shock is `-X` on a rise and `-10X` on a fall, applied **relatively** rather than in points, then damped across tenors by a response function `VR(t)` that is 1 at zero and decreasing. `vol_coord` implements it. Measured against a live index ratio book, on a −30% to 0% axis:

| | RMS against Risk Navigator, as a fraction of trough depth |
|---|---|
| `const` vs its blue curve | **~2%** |
| `vol_coord` vs its Vol.Coord. curve | **~3.5%** |

with the residual at every shock inside the error of reading the targets off a chart by eye. `VR(t)` itself is not published: `vol_coord_decay` is fitted here, `exp(-4.736 t)`, on one book from nine points read off a chart by eye. Two things follow, and the engine says both out loud rather than leaving them in the docs.

Every `vol_coord` curve running on the shipped decay **says so in `warnings`**. And the fit was constrained only out to **0.345 years**, because that is all the book it came from held; past there an exponential does not merely lose accuracy, it decays to nothing. At one year `VR` is 0.009, so this model would reprice a LEAPS as though a 20% crash barely touched its volatility. Any position beyond `vol_coord_calibrated_to_years` is priced anyway and **named in `warnings`**, because a silent extrapolation that understates long-dated vega is exactly the failure this server exists not to have. A floor on `VR` would tidy the symptom away and hide it, so there isn't one.

To replace the number rather than trust it, call `calibrate_vol_coord` with four or more readings off Risk Navigator's own Vol.Coord. curve on its Equity tab — `{shock: -0.20, pnl: -28000}`, shocks as fractions. The same fit is available from a shell:

```
uv run python scripts/calibrate_vol_coord.py -- -0.05=-8000 -0.10=-22000 -0.15=-31500 -0.20=-28000 -0.25=-12000
```

Either route refits the decay and returns the residual at every point, the tenor range your positions actually constrain, and the most extreme volatility the fit produces — a decay that reproduces the curve by pricing a wing at 150% has fitted the chart rather than the market, and it tells you so.

**The fit is kept.** It goes to `~/.ibkr-risk-mcp/vol_coord.json` (`IBKR_CALIBRATION_FILE` moves it) and becomes the default `vol_coord_decay` for every later `stress_curve` on that machine, no restart and nothing to carry by hand — which was the actual reason the shipped number kept being the one in use. Stored beside it is what it was fitted against: the targets, the residuals, the account, the date. A calibrated run then reports that provenance in `assumptions.volCoordDecaySource` and in `warnings` instead of the "factory decay" caveat, and a decay you pass explicitly is described as neither — this server did not fit it and does not vouch for it.

A fit taken against a portfolio that does **not** reconcile is returned but never stored. The asymmetry is the point: a curve missing a position announces itself through `reconciled`, while a decay that absorbed the same gap would go on deforming every later run with nothing to give it away.

Being **relative** is the part that matters, and it is why the additive slopes were removed from the defaults. Multiplying every volatility by the same factor puts more *points* on a wing already quoted at 41% than on a 31% at-the-money, so the surface steepens by itself. A parallel shift in points cannot do that at any slope, and on a ratio book the difference is not a matter of degree: the additive model made the curve monotonically worse through the region where Risk Navigator turns it back up, and put its crossover around −35% where Risk Navigator puts it near −18%. Measured there, `vol_coord` troughs at roughly **60% of the depth** of the constant-volatility curve and at a **much shallower shock** — so a rising-volatility regime came out as the *better* one, the opposite of what a naive short-vega reading predicts. That is the reason the regimes are returned as separate curves rather than as a band.

**The additive slope alternative is your input, not a measurement.** Neither vol mode moves it: `sticky_strike` pins volatility to the strike and `sticky_moneyness` slides a strike along today's smile, so with both slopes at zero the curve prices the move in the underlying and not the move in volatility that comes with it — the optimistic half of the answer for a net short option book. `vol_slope_down` puts it in, at volatility points per 1% fall. It is applied as a **parallel** shift, flat across tenors, and a real surface does neither: it steepens in a sell-off, which understates a long out-of-the-money put, and the front month moves more than a 120-day tenor. A steepening term was tried and removed — the values that reproduced the observed shape priced the long wings above 100% implied volatility, which is curve fitting rather than modelling.

`stress_curve` exists because of that. Rather than burying one regime in one result, it returns a curve per regime over a single loading of the portfolio and a single surface, so the curves differ by assumption alone — by default the same two Risk Navigator draws. **Read the slope-0 curve first**: it is the constant-volatility case and the only one with an external check against Risk Navigator's blue line — and it passes. Measured against a live index ratio book, `stress_curve` at `vol_mode='sticky_strike'`, slope 0, tracks Risk Navigator's blue curve to within **1-3% at every shock from 0 to −30%**. That check only works in `sticky_strike`, which is why it is the default here; `sticky_moneyness` is a different model and on the same book put the trough 1.8x deeper. And do not assume the steepest slope is the worst case everywhere — on a book holding long wings the ordering reverses in the far tail, where the least-deep long puts carry the most vega and a rising volatility starts helping.

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

`scripts/smoke_test.py` exercises the live path: connection, greeks with a count of any missing `modelGreeks`, a surface, `stress_portfolio` from −30% to +30% in 1% steps with the reconciliation check, `stress_curve` over its three default regimes — checking both that the constant-volatility curve starts at zero and that `volSurfaceUsed` is not empty — `stress_whatif`, and a `whatif_order` on a single deeply out-of-the-money leg. Point `.env` at the paper port to try it safely.

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
  calibration.py  where the fitted vol_coord decay is stored between sessions
  auth.py         bearer token verification, for the HTTP transport only
scripts/smoke_test.py
scripts/calibrate_vol_coord.py
scripts/http_smoke.py       drives the HTTP transport as a real client would
docs/remote.md              running it over HTTP: auth, timeouts, the service unit
tests/
```

`pricing.py` and `contracts.py` have no IB dependency at all, which is what makes them testable.
