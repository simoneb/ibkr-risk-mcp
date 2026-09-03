"""Bearer token verification for the HTTP transport.

This server is an OAuth 2.1 *resource server* and nothing more. It issues no
tokens, runs no login screen, stores no passwords and holds no user database.
An external identity provider does all of that; the job here is to decide
whether the token in front of it was signed by that provider, is still valid,
was minted for this server, and names somebody allowed in.

**Why the authorisation model is a list of names.** There is one IB Gateway
behind this process, logged into one account, and the tools read that account
whoever asks. A token cannot select whose data to return, because there is only
one book — so unlike a mailbox or a drive, identity here does not route a
request, it only admits one. That makes the allowlist the entire security
boundary, and it makes the two checks either side of it matter more than they
would elsewhere:

- **Audience.** Providers issue tokens for many applications. Without checking
  who a token was minted *for*, a token your provider issued for some unrelated
  app of yours would open this one.
- **Subject.** A valid token from the right provider naming the wrong person is
  a stranger holding a complete position book.

**401 and 403 are different answers and both are used.** A token that fails
cryptographically or on its claims returns ``None`` here, which the SDK turns
into a 401 and a client turns into "go and authenticate". A token that is
entirely valid but names a subject who is not on the list comes back as a real
token carrying no scopes, which the SDK turns into a 403. The distinction is
not pedantry: answering 401 to the second case tells a client its credentials
were bad and invites it to fetch the same token again, forever.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
import jwt
from jwt import PyJWK

from mcp.server.auth.provider import AccessToken

from .config import Settings

log = logging.getLogger(__name__)

#: Signature algorithms this server will consider.
#:
#: Asymmetric only, and enumerated rather than left to the token. A JWT names
#: its own algorithm in a header nobody has verified yet, so a verifier that
#: honours that name can be handed ``alg: HS256`` and a signature computed with
#: the provider's *public* key — which is public. Listing only algorithms that
#: verify with a public key makes that substitution impossible rather than
#: merely unlikely. ``none`` is excluded for the obvious reason.
ALLOWED_ALGORITHMS = ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512")

#: How long a fetched key set is trusted before it is refetched.
JWKS_TTL_SECONDS = 3600.0

#: Tolerance on `exp` and `iat`, for clocks that disagree.
#:
#: Small and deliberate. Zero rejects a token that is valid everywhere else
#: because a VPS drifted a few seconds, which reads as an auth bug and gets
#: "fixed" by someone disabling a check. Large starts extending the life of
#: tokens the provider has already retired. Thirty seconds covers ordinary NTP
#: drift and nothing more.
CLOCK_SKEW_SECONDS = 30.0

#: The floor between refetches triggered by an unrecognised key id.
#:
#: Key rotation has to be picked up without a restart, so an unknown ``kid``
#: has to be able to force a refresh. But ``kid`` is attacker-controlled and
#: unauthenticated at that point, so without a floor anyone could turn a stream
#: of junk tokens into a stream of outbound requests to the provider.
JWKS_MIN_REFETCH_SECONDS = 60.0


class JwksCache:
    """The provider's signing keys, fetched on demand and kept for a while."""

    def __init__(
        self,
        url: str,
        *,
        ttl: float = JWKS_TTL_SECONDS,
        min_refetch: float = JWKS_MIN_REFETCH_SECONDS,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = url
        self._ttl = ttl
        self._min_refetch = min_refetch
        self._client = client
        self._keys: dict[str, PyJWK] = {}
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    async def _fetch(self) -> None:
        client = self._client or httpx.AsyncClient(timeout=10.0)
        try:
            response = await client.get(self._url)
            response.raise_for_status()
            document = response.json()
        finally:
            if self._client is None:
                await client.aclose()

        keys: dict[str, PyJWK] = {}
        for entry in document.get("keys") or []:
            kid = entry.get("kid")
            if not kid:
                continue
            try:
                keys[kid] = PyJWK(entry)
            except Exception as exc:  # a provider may publish keys for other uses
                log.debug("skipping unusable JWK %s: %s", kid, exc)
        # Only replace a populated cache with another populated one. A provider
        # that answers 200 with an empty set — a deploy blip, a proxy page —
        # would otherwise lock everybody out until the TTL expired.
        if keys or not self._keys:
            self._keys = keys
        self._fetched_at = time.monotonic()
        log.info("fetched %d signing key(s) from %s", len(keys), self._url)

    async def key_for(self, kid: str) -> PyJWK | None:
        now = time.monotonic()
        age = now - self._fetched_at
        known = self._keys.get(kid)
        if known is not None and age < self._ttl:
            return known
        if known is None and age < self._min_refetch and self._fetched_at:
            # Recently refreshed and still no such key: this token is not going
            # to become valid by asking the provider again.
            return None
        async with self._lock:
            # Another request may have refreshed while this one waited.
            if time.monotonic() - self._fetched_at < self._min_refetch and self._fetched_at:
                return self._keys.get(kid)
            try:
                await self._fetch()
            except Exception as exc:
                log.warning("could not fetch signing keys from %s: %s", self._url, exc)
                # Serve a stale key rather than failing closed on a blip. The
                # key is still the provider's and the token's own expiry is
                # still checked; refusing here would turn the provider being
                # briefly unreachable into this server being down.
                return self._keys.get(kid)
            return self._keys.get(kid)


class JwtTokenVerifier:
    """Verifies provider-issued JWTs and admits the people on the list."""

    def __init__(self, settings: Settings, *, jwks: JwksCache | None = None):
        if not (settings.mcp_auth_issuer and settings.mcp_auth_audience):
            raise ValueError("auth is not configured; load_settings refuses this combination")
        self._issuer = settings.mcp_auth_issuer
        self._audience = settings.mcp_auth_audience
        self._subjects = frozenset(settings.mcp_allowed_subjects)
        self._scope = settings.mcp_auth_scope
        self._resource = settings.mcp_resource_url
        self._jwks = jwks or JwksCache(settings.mcp_auth_jwks_url or "")

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            log.warning("rejecting token: unreadable header (%s)", exc)
            return None

        algorithm = header.get("alg")
        if algorithm not in ALLOWED_ALGORITHMS:
            log.warning("rejecting token: algorithm %r is not accepted here", algorithm)
            return None

        kid = header.get("kid")
        if not kid:
            log.warning("rejecting token: no key id in the header")
            return None

        key = await self._jwks.key_for(kid)
        if key is None:
            log.warning("rejecting token: no signing key published for kid %r", kid)
            return None

        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key.key,
                algorithms=list(ALLOWED_ALGORITHMS),
                audience=self._audience,
                issuer=self._issuer,
                leeway=CLOCK_SKEW_SECONDS,
                # Absent is not the same as wrong, and both are refusals. A
                # token with no `aud` at all must not slip past the audience
                # check by having nothing to compare.
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as exc:
            # The class of failure is worth logging; the token is not.
            log.warning("rejecting token: %s (%s)", type(exc).__name__, exc)
            return None

        subject = str(claims.get("sub") or "")
        allowed = subject in self._subjects
        if not allowed:
            log.warning(
                "refusing subject %r: a valid token from %s, but not on the allowlist",
                subject,
                self._issuer,
            )

        return AccessToken(
            token=token,
            # Who presented it, for the log. `azp` is the party the token was
            # issued to where a provider sets it; falling back to the subject
            # keeps this populated rather than accurate-or-empty.
            client_id=str(claims.get("azp") or claims.get("client_id") or subject or "unknown"),
            # The scope is granted here, not read from the token. There is
            # exactly one permission in this server and the provider does not
            # decide who holds it — which also means a provider misconfigured
            # to hand out scopes freely cannot widen access to this account.
            scopes=[self._scope] if allowed else [],
            expires_at=int(claims["exp"]),
            resource=self._resource,
            subject=subject,
            claims=claims,
        )


def verifier_for(settings: Settings) -> JwtTokenVerifier | None:
    """The verifier this configuration calls for, or ``None`` for no auth."""
    if not settings.mcp_auth:
        return None
    return JwtTokenVerifier(settings)
