# Running this server over HTTP

The default transport is stdio and nothing here changes that. This document is
for the other deployment: the server running somewhere the client cannot launch
it — typically beside an IB Gateway on a host that stays up — reached over
streamable HTTP through a reverse proxy.

```bash
IBKR_MCP_TRANSPORT=http IBKR_MCP_PORT=8765 ibkr-risk-mcp
```

See `.env.example` for `IBKR_MCP_HOST`, `IBKR_MCP_PORT` and `IBKR_MCP_PATH`.
Note that `IBKR_HOST` and `IBKR_PORT` are a different machine's — TWS's — and
the two pairs are deliberately not interchangeable.

## Run the entry point, not the uvicorn CLI

`ibkr-risk-mcp` with `IBKR_MCP_TRANSPORT=http` awaits `uvicorn.Server.serve()`
on the asyncio loop it is already running, rather than calling `uvicorn.run()`.
That is not an implementation detail worth ignoring: `uvicorn.run()` selects
`loop="auto"`, which is uvloop wherever uvloop is installed, and ib_async drives
its own socket transport and its market-data callbacks on whatever loop it
finds. Keeping the process on plain asyncio keeps ib_async on the loop it is
tested against.

If a deployment ever prefers `uvicorn` on the command line against the ASGI
app, it needs `--loop asyncio` to stay equivalent.

## The service unit

```ini
[Unit]
Description=ibkr-risk-mcp
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=ibkr-risk
ExecStart=/opt/ibkr-risk-mcp/.venv/bin/ibkr-risk-mcp
EnvironmentFile=/etc/ibkr-risk-mcp/env
Restart=on-failure
RestartSec=5

# The calibration lives here. StateDirectory creates /var/lib/ibkr-risk-mcp
# owned by this user and leaves it alone across reinstalls, which the service
# user's home directory does not reliably do.
StateDirectory=ibkr-risk-mcp
Environment=IBKR_CALIBRATION_FILE=/var/lib/ibkr-risk-mcp/vol_coord.json

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

`ExecStart` runs the console script, not `uvicorn` — see the section above for
why that is load-bearing rather than stylistic.

### The calibration file is the one piece of state worth protecting

`calibrate_vol_coord` fits the `vol_coord` decay against your own Risk
Navigator and writes it to a small JSON file. It is the only thing this server
writes, and it defaults to `~/.ibkr-risk-mcp/vol_coord.json` — which on a VPS
means whatever `$HOME` the service user happens to have on the day of a
redeploy.

**Losing it is silent.** `calibration.load()` falls back to the shipped decay
rather than raising, deliberately, because a stale file should not take down a
stress run. The cost of that design is that a *missing* file is equally quiet:
every curve afterwards is fitted to somebody else's book and nothing says so.

Two things make it visible and durable:

- `StateDirectory` above, plus `IBKR_CALIBRATION_FILE` pointing into it. Back
  that directory up; it is a few hundred bytes.
- `check_connection` now returns a `calibration` field — the provenance line,
  naming the date, the account and the number of Risk Navigator points the fit
  used, or `null` if the factory decay is in use. Check it after any redeploy.

```
"calibration": "calibrated on 2026-08-28T17:51:59 against account U1234567,
                from 5 Risk Navigator point(s), stored at /var/lib/ibkr-risk-mcp/vol_coord.json"
```

### Gateway settings that change on a VPS

- `IBKR_PORT=4001` for a live Gateway, `4002` for paper — not 7496, which is
  desktop TWS.
- `IBKR_CLIENT_ID` must not collide with anything else pointed at that Gateway,
  IBC included. Pick one deliberately and write it down. A collision is
  reported legibly as `client_id_in_use` and names what to change, so this is
  bookkeeping rather than risk.
- `IBKR_MARKET_DATA_TYPE`: confirm the entitlement follows the login to the
  VPS. Type 3 is a documented fallback that still carries model greeks; the
  values are about fifteen minutes old, which matters for a fast-moving book
  and not at all for the shape of a P&L curve.
- `IBKR_ENABLE_WHATIF` stays in the environment rather than anywhere a model
  can reach, which is the whole reason it is an environment variable.

## How long a call takes

Measured over HTTP against live TWS, on a 62-position book (29 of them options)
with `IBKR_MAX_MKT_DATA_LINES=12` — a third of the default, so these are
pessimistic for line contention:

| call | cold | warm |
|---|---|---|
| `check_connection` (includes the IB connect) | 0.6s | — |
| `get_margin_summary` | <0.1s | — |
| `get_position_greeks` | 5.1s | 2.7s |
| `stress_portfolio`, 61 shocks | 3.8s | 4.0s |
| `stress_curve`, 61 shocks x 2 volatility scenarios | 5.2s | — |
| `get_vol_surface`, 1 expiry, both rights | 8.7s | — |

Under two genuinely overlapping callers, a `stress_portfolio` that runs in 4.5s
alone takes 6.2s, and the second caller is no slower than solo. The slowest
single call observed anywhere, including under contention, is a `get_vol_surface`
at 17.6s.

**The bound to reason with**, when the book or the request is larger than the
one above, is market data rather than compute — the repricing itself is numpy
over a grid and is not where the time goes:

```
ceil(subscriptions / IBKR_MAX_MKT_DATA_LINES) * IBKR_GREEKS_TIMEOUT
```

With the defaults (40 lines, 4.0s) a six-expiry surface of 25 strikes on both
rights — 300 subscriptions, the widest thing this server will be asked for —
bounds at about 32 seconds, and in practice comes in far below it because most
contracts answer in well under a second and the ones that never answer are the
only ones that pay the full timeout.

## The three places a call can be cut off

Set these deliberately. They are checked here rather than assumed:

**1. The ASGI server — not a limiter.** uvicorn has no request-duration
timeout. `timeout_keep_alive` defaults to 5s and applies to idle connections
between requests, not to a request in flight, and an open SSE response is a
request in flight. So nothing needs configuring here, which is convenient:
`FastMCP`'s `uvicorn.Config` is not parameterised, and setting anything would
mean building the Config by hand.

**2. The reverse proxy — the one you own, and the only one worth choosing.**
Caddy's `http` transport documents `read_timeout` and `write_timeout` as
defaulting to *no timeout*, so out of the box the proxy imposes nothing:

```caddyfile
risk.example.com {
	reverse_proxy 127.0.0.1:8765 {
		transport http {
			read_timeout 120s
			write_timeout 120s
		}
	}
}
```

120s is a deliberate middle and not a requirement. It sits an order of
magnitude above the slowest call ever measured here and well below the client's
300s, so a genuinely wedged call fails at the proxy with a clean error instead
of hanging until the client gives up — while nothing real ever reaches it.
Leaving both unset is also defensible; what is not defensible is setting them
to a number nobody compared against the table above.

Response buffering needs no configuration: Caddy flushes immediately when the
response is `Content-Type: text/event-stream` or has an unknown
`Content-Length`, which covers streamable HTTP's streaming replies. Adding
`flush_interval -1` forces low-latency mode for everything and does no harm,
but it is belt-and-braces rather than the fix it is often quoted as.

**3. The client — not yours to set.** The MCP Python SDK's default client uses
`Timeout(connect=30, read=300, write=30, pool=30)`. The one that binds is
`read`: it is the gap allowed *between bytes*, so a tool computing in silence
is held to 300 seconds however fast the link is. Every measurement above is
under 6% of that.

**The ceiling is therefore 300 seconds, imposed by the client, and nothing
measured comes within an order of magnitude of it.** That is the reason there
is no progress reporting and no job-plus-poll pair of tools here: both are real
answers to a real problem, and this server does not have that problem. If a
future book or a wider surface starts producing calls in the tens of seconds,
revisit it then — and revisit it by measuring, not by raising a number.

## Authentication

Off by default. With `IBKR_MCP_AUTH=true` this server becomes an OAuth 2.1
**resource server**: it verifies bearer tokens that an external identity
provider issued, and never sees a credential itself. There is no login screen
here, no client secret, no user database, and no token issuance.

```bash
IBKR_MCP_AUTH=true
IBKR_MCP_AUTH_ISSUER=https://your-tenant.example.com/
IBKR_MCP_RESOURCE_URL=https://risk.example.com/mcp
IBKR_MCP_ALLOWED_SUBJECTS=auth0|1234567890
```

The SDK does the protocol half once those are set: it mounts
`/.well-known/oauth-protected-resource/mcp` (RFC 9728 — note that the
well-known segment goes *before* the resource path) and answers an
unauthenticated request with `401` and a
`WWW-Authenticate: Bearer resource_metadata=...` challenge pointing at it.
Authorization-server metadata (RFC 8414) and client registration (RFC 7591) are
the provider's endpoints, not this server's.

### Why the allowlist is the whole authorisation model

Worth being explicit, because the usual mental model does not fit. When you
connect Claude to Gmail, the OAuth token does two jobs at once: it proves who
you are *and* it selects whose mailbox to read. That is why Google has to be the
provider.

This server is not shaped like that. One IB Gateway, logged into one account,
with credentials that live on the host — the data is chosen before any request
arrives, and every caller would get the same book. So a token cannot route a
request here; it can only admit one. Which puts the entire security boundary in
two checks:

- **`aud`** — providers issue tokens for many applications. Without checking who
  a token was minted *for*, a token your own provider issued for an unrelated
  app of yours opens this one.
- **`sub`** — a valid token from the right provider naming the wrong person is a
  stranger holding a complete position book.

An empty allowlist therefore never means "everyone". With auth on and no
subjects configured, the server refuses to start.

The scope is granted by this server to allowlisted subjects, not read from the
token. A provider misconfigured to hand out scopes freely cannot widen access to
this account.

### 401 and 403 mean different things here

A token that fails cryptographically or on its claims gets `401`, and a client
reads that as "go and authenticate". A token that is entirely valid but names
somebody not on the list gets `403 insufficient_scope`. The distinction is not
pedantry: answering `401` to the second case tells a client its credentials were
bad and invites it to fetch the very same token again, forever.

### Four things that fail at connection time if you get them wrong

Claude's connector documentation is specific about these, and three of them
produce a failure with nothing useful pointing at the cause. Two are now
startup errors here rather than mysteries later.

1. **The `resource` field must match the URL you type into Claude exactly**,
   path and all. `IBKR_MCP_RESOURCE_URL` is therefore checked at startup
   against `IBKR_MCP_PATH`: a resource of `https://risk.example.com` on a
   server serving `/mcp` refuses to boot rather than publishing a document that
   is internally consistent and useless.
2. **Every auth URL must be https.** Also enforced at startup, loopback
   excepted for the local test. The JWKS URL is the one that matters and the
   one that looks harmless: whoever can answer it hands this server a signing
   key of their own, and every token check after that is theatre.
3. **Your provider must know the scope this server advertises.** Claude
   requests whatever is in `scopes_supported`, which here is
   `IBKR_MCP_AUTH_SCOPE` — `risk:read` by default. Define it as a permission on
   the API you registered, or the authorization request may be rejected before
   a token is ever minted. Note this is separate from the scope *check*: the
   server grants `risk:read` itself to allowlisted subjects and never reads the
   token's scope claim.
4. **Your identity provider has to be reachable from Anthropic's network too.**
   Discovery requests come from the same egress range as requests to the MCP
   server, `160.79.104.0/21`. A WAF or conditional-access rule in front of the
   provider breaks the flow even when the MCP server itself is reachable.

That egress range is also worth using in the other direction: the MCP port only
ever needs to answer Anthropic, so firewalling it to `160.79.104.0/21` costs
nothing and removes the whole internet from the set of things that can reach a
`401`.

### Choosing a provider

Any OIDC provider works — the server only ever needs an issuer, a JWKS URL, an
audience and a subject. That makes the choice low-stakes and reversible: moving
providers is four environment variables.

Claude's custom connectors will register themselves automatically with a
provider that supports dynamic client registration (RFC 7591), which is the
smoother path; a provider without it is still usable, because Claude's advanced
settings accept a pre-registered OAuth Client ID and Secret. The provider must
support PKCE with `S256` and advertise it — Claude sends a `code_challenge` on
every authorization request regardless of how the client was registered.

For a deployment with one user, a hosted provider (Auth0, WorkOS, Stytch) is
less ongoing work than self-hosting Keycloak on the same box — there is no
second service to patch, back up and keep alive.

Claude does support a fixed API key or bearer token instead of OAuth, through
request headers. It is not used here, deliberately: a static credential does
not expire, cannot be revoked without a redeploy, and sits in a connector
configuration for as long as nobody thinks to rotate it. Against a server that
returns a live account's positions, token expiry and revocation are worth the
extra moving parts.

### Finding your subject

The allowlist wants a token subject, and you need a token to learn what yours
is. Rather than decoding one by hand: set `IBKR_MCP_ALLOWED_SUBJECTS` to any
placeholder, connect once, and read the log —

```
refusing subject 'auth0|68f3c1...': a valid token from https://your-tenant/, but not on the allowlist
```

Paste it in and restart. The first attempt returns 403 rather than 401, so the
client will not sit in a re-authentication loop while you do it.

### Verifying it

`uv run python scripts/http_smoke.py --auth` starts the server behind a stub
provider and checks the boundary end to end: the discovery challenge, the
allowlisted subject, the valid-token-wrong-subject 403, and 401 for expired,
wrong-audience and malformed tokens. It needs no TWS, because all of those are
decided before a tool runs, and it runs on CI.

**Until auth is turned on, an HTTP deployment is only as private as whatever is
in front of it** — which is why `IBKR_MCP_HOST` defaults to loopback. Do not
bind a public interface and rely on the URL being unguessable.
