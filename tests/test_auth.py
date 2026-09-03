"""Bearer token verification.

This is the security boundary, and unlike the rest of the remote work it can be
tested properly without TWS or a provider: tokens are signed here with a key
generated here, and the key set is served by a stub transport. Every case below
is a token somebody could actually present.

The two that matter most are the ones that are *nearly* right. A token signed
by the correct provider but minted for a different application of yours, and a
token that is valid in every respect but names somebody else — neither is
malformed, neither fails a signature check, and both would open an account's
complete position book.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from ibkr_risk_mcp.auth import JwksCache, JwtTokenVerifier
from ibkr_risk_mcp.config import load_settings

ISSUER = "https://idp.example.com/"
AUDIENCE = "https://risk.example.com/mcp"
RESOURCE = "https://risk.example.com/mcp"
JWKS_URL = "https://idp.example.com/.well-known/jwks.json"
ME = "auth0|the-one-account-holder"
SOMEONE_ELSE = "auth0|a-stranger"


def make_key(kid: str):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return private, jwk


@pytest.fixture(scope="module")
def keypair():
    """One 2048-bit key for the module: generating them is slow enough that a
    per-test fixture doubles the suite's runtime for no extra coverage."""
    return make_key("key-1")


@pytest.fixture
def jwks_server(keypair):
    """A stub for the provider's key endpoint that counts what it is asked.

    Counting matters: an unknown key id has to be able to force a refetch so
    that rotation works without a restart, and must not be able to force one
    per request, because the key id is attacker-controlled.
    """
    _, jwk = keypair
    state = {"calls": 0, "keys": [jwk], "status": 200}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["status"] != 200:
            return httpx.Response(state["status"])
        return httpx.Response(200, json={"keys": state["keys"]})

    state["client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return state


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("IBKR_MCP_AUTH", "true")
    monkeypatch.setenv("IBKR_MCP_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("IBKR_MCP_RESOURCE_URL", RESOURCE)
    monkeypatch.setenv("IBKR_MCP_AUTH_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("IBKR_MCP_ALLOWED_SUBJECTS", f" {ME} , ")
    return load_settings()


@pytest.fixture
def verifier(settings, jwks_server):
    cache = JwksCache(JWKS_URL, client=jwks_server["client"], min_refetch=0.0)
    return JwtTokenVerifier(settings, jwks=cache)


def token(
    keypair,
    *,
    sub: str = ME,
    aud: str = AUDIENCE,
    iss: str = ISSUER,
    kid: str = "key-1",
    exp_delta: int = 3600,
    algorithm: str = "RS256",
    key=None,
    drop: tuple[str, ...] = (),
    **extra,
) -> str:
    private, _ = keypair
    now = int(time.time())
    claims = {"sub": sub, "aud": aud, "iss": iss, "iat": now, "exp": now + exp_delta, **extra}
    for field in drop:
        claims.pop(field, None)
    return jwt.encode(claims, key or private, algorithm=algorithm, headers={"kid": kid})


# --------------------------------------------------------------------------
# the two answers this verifier gives
# --------------------------------------------------------------------------


async def test_the_account_holder_is_admitted(verifier, keypair, settings):
    result = await verifier.verify_token(token(keypair))
    assert result is not None
    assert result.subject == ME
    assert result.scopes == [settings.mcp_auth_scope]


async def test_a_valid_token_for_someone_else_is_a_403_not_a_401(verifier, keypair):
    """The distinction is the point of returning a token with no scopes.

    Answering `None` here would become a 401, which tells the client its
    credentials were bad and invites it to fetch the very same token again.
    A real token carrying no scopes becomes a 403, which is both true and
    terminal.
    """
    result = await verifier.verify_token(token(keypair, sub=SOMEONE_ELSE))
    assert result is not None, "a validly signed token is not an authentication failure"
    assert result.scopes == [], "but it grants nothing"
    assert result.subject == SOMEONE_ELSE


# --------------------------------------------------------------------------
# tokens that are nearly right
# --------------------------------------------------------------------------


async def test_a_token_for_another_application_is_refused(verifier, keypair):
    """Same provider, same signing key, different audience.

    This is the failure that has nothing to do with cryptography: everything
    about the token is genuine except who it was minted for.
    """
    assert await verifier.verify_token(token(keypair, aud="https://something-else.example.com")) is None


async def test_a_token_from_another_issuer_is_refused(verifier, keypair):
    assert await verifier.verify_token(token(keypair, iss="https://not-your-idp.example.com/")) is None


async def test_an_expired_token_is_refused(verifier, keypair):
    assert await verifier.verify_token(token(keypair, exp_delta=-30)) is None


@pytest.mark.parametrize("field", ["aud", "sub", "exp", "iss", "iat"])
async def test_a_missing_claim_is_refused(verifier, keypair, field):
    """Absent is not the same as wrong, and both have to be refusals — a token
    with no `aud` at all must not pass the audience check by having nothing to
    compare against."""
    assert await verifier.verify_token(token(keypair, drop=(field,))) is None


async def test_a_tampered_payload_is_refused(verifier, keypair):
    header, payload, signature = token(keypair).split(".")
    other, _, _ = token(keypair, sub=SOMEONE_ELSE).split(".")
    assert await verifier.verify_token(f"{header}.{payload[:-4]}AAAA.{signature}") is None


# --------------------------------------------------------------------------
# algorithm substitution
# --------------------------------------------------------------------------


async def test_the_none_algorithm_is_refused(verifier, keypair):
    unsigned = jwt.encode(
        {"sub": ME, "aud": AUDIENCE, "iss": ISSUER, "exp": int(time.time()) + 60},
        key=None,
        algorithm="none",
        headers={"kid": "key-1"},
    )
    assert await verifier.verify_token(unsigned) is None


async def test_an_hmac_token_signed_with_the_public_key_is_refused(verifier, keypair):
    """The classic substitution, and the reason the algorithm list is fixed.

    The provider's public key is public. A verifier that trusts the `alg`
    header can be handed a token that says HS256 and is signed with that public
    key as if it were a shared secret — and it verifies, because the "secret"
    was never secret. Refusing every symmetric algorithm outright is what makes
    this impossible rather than merely hard.

    Assembled by hand, because PyJWT refuses to sign *or* verify with a PEM key
    as an HMAC secret. That refusal is a second guard and a welcome one, but it
    is the library's and not this server's: it would disappear with a change of
    JWT library, and the algorithm allowlist would still hold. This test is
    about the guard we own.
    """
    private, _ = keypair
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    now = int(time.time())

    def segment(data: dict) -> bytes:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=")

    signing_input = b".".join(
        (
            segment({"alg": "HS256", "typ": "JWT", "kid": "key-1"}),
            segment({"sub": ME, "aud": AUDIENCE, "iss": ISSUER, "iat": now, "exp": now + 60}),
        )
    )
    signature = base64.urlsafe_b64encode(
        hmac.new(public_pem, signing_input, hashlib.sha256).digest()
    ).rstrip(b"=")
    forged = (signing_input + b"." + signature).decode()

    # Self-consistent: the signature really does check out against the public
    # key treated as a shared secret. That is the whole attack, and it is what
    # a verifier trusting the `alg` header would find.
    assert hmac.compare_digest(
        signature,
        base64.urlsafe_b64encode(
            hmac.new(public_pem, signing_input, hashlib.sha256).digest()
        ).rstrip(b"="),
    )
    assert jwt.get_unverified_header(forged)["alg"] == "HS256"
    assert await verifier.verify_token(forged) is None


async def test_a_token_with_no_key_id_is_refused(verifier, keypair):
    private, _ = keypair
    now = int(time.time())
    anonymous = jwt.encode(
        {"sub": ME, "aud": AUDIENCE, "iss": ISSUER, "iat": now, "exp": now + 60},
        private,
        algorithm="RS256",
    )
    assert await verifier.verify_token(anonymous) is None


# --------------------------------------------------------------------------
# the key set
# --------------------------------------------------------------------------


async def test_an_unknown_key_id_is_refused(verifier, keypair):
    assert await verifier.verify_token(token(keypair, kid="not-a-key")) is None


async def test_key_rotation_is_picked_up_without_a_restart(settings, jwks_server, keypair):
    cache = JwksCache(JWKS_URL, client=jwks_server["client"], min_refetch=0.0)
    verifier = JwtTokenVerifier(settings, jwks=cache)
    assert await verifier.verify_token(token(keypair)) is not None

    rotated_private, rotated_jwk = make_key("key-2")
    jwks_server["keys"] = [rotated_jwk]
    rotated = jwt.encode(
        {
            "sub": ME,
            "aud": AUDIENCE,
            "iss": ISSUER,
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        },
        rotated_private,
        algorithm="RS256",
        headers={"kid": "key-2"},
    )
    assert await verifier.verify_token(rotated) is not None


async def test_junk_key_ids_do_not_become_a_request_per_token(settings, jwks_server, keypair):
    """`kid` is unauthenticated and attacker-controlled at the point it is read.

    Without a floor between refetches, a stream of tokens carrying random key
    ids turns into a stream of outbound requests to the provider — this server
    doing someone else's traffic for them.
    """
    cache = JwksCache(JWKS_URL, client=jwks_server["client"], min_refetch=300.0)
    verifier = JwtTokenVerifier(settings, jwks=cache)
    for i in range(25):
        assert await verifier.verify_token(token(keypair, kid=f"junk-{i}")) is None
    assert jwks_server["calls"] == 1, "one fetch for the first miss, none for the other 24"


async def test_a_provider_outage_does_not_lock_out_a_valid_token(settings, jwks_server, keypair):
    """A cached key is still the provider's key, and the token's own expiry is
    still checked. Failing closed here would turn the provider being briefly
    unreachable into this server being down."""
    cache = JwksCache(JWKS_URL, client=jwks_server["client"], ttl=0.0, min_refetch=0.0)
    verifier = JwtTokenVerifier(settings, jwks=cache)
    assert await verifier.verify_token(token(keypair)) is not None

    jwks_server["status"] = 503
    assert await verifier.verify_token(token(keypair)) is not None
    assert jwks_server["calls"] > 1, "it did try"


async def test_an_empty_key_set_does_not_wipe_a_working_cache(settings, jwks_server, keypair):
    """A 200 carrying no keys — a deploy blip, a proxy interstitial — would
    otherwise lock everybody out until the TTL expired."""
    cache = JwksCache(JWKS_URL, client=jwks_server["client"], ttl=0.0, min_refetch=0.0)
    verifier = JwtTokenVerifier(settings, jwks=cache)
    assert await verifier.verify_token(token(keypair)) is not None

    jwks_server["keys"] = []
    assert await verifier.verify_token(token(keypair)) is not None


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_auth_on_with_no_allowlist_refuses_to_start(monkeypatch):
    """The failure this prevents is not a server that rejects everyone. It is
    one that looks configured and is not."""
    monkeypatch.setenv("IBKR_MCP_AUTH", "true")
    monkeypatch.setenv("IBKR_MCP_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("IBKR_MCP_RESOURCE_URL", RESOURCE)
    monkeypatch.delenv("IBKR_MCP_ALLOWED_SUBJECTS", raising=False)
    with pytest.raises(ValueError, match="IBKR_MCP_ALLOWED_SUBJECTS"):
        load_settings()


@pytest.mark.parametrize("missing", ["IBKR_MCP_AUTH_ISSUER", "IBKR_MCP_RESOURCE_URL"])
def test_auth_on_with_a_missing_essential_refuses_to_start(monkeypatch, missing):
    monkeypatch.setenv("IBKR_MCP_AUTH", "true")
    monkeypatch.setenv("IBKR_MCP_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("IBKR_MCP_RESOURCE_URL", RESOURCE)
    monkeypatch.setenv("IBKR_MCP_ALLOWED_SUBJECTS", ME)
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(ValueError, match=missing):
        load_settings()


def test_the_audience_defaults_to_the_resource_url(monkeypatch):
    monkeypatch.setenv("IBKR_MCP_AUTH", "true")
    monkeypatch.setenv("IBKR_MCP_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("IBKR_MCP_RESOURCE_URL", RESOURCE)
    monkeypatch.setenv("IBKR_MCP_ALLOWED_SUBJECTS", ME)
    monkeypatch.delenv("IBKR_MCP_AUTH_AUDIENCE", raising=False)
    monkeypatch.delenv("IBKR_MCP_AUTH_JWKS_URL", raising=False)
    cfg = load_settings()
    assert cfg.mcp_auth_audience == RESOURCE
    assert cfg.mcp_auth_jwks_url == "https://idp.example.com/.well-known/jwks.json"


def test_auth_off_needs_no_configuration_at_all(monkeypatch):
    for name in (
        "IBKR_MCP_AUTH",
        "IBKR_MCP_AUTH_ISSUER",
        "IBKR_MCP_RESOURCE_URL",
        "IBKR_MCP_ALLOWED_SUBJECTS",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = load_settings()
    assert cfg.mcp_auth is False
    assert cfg.mcp_allowed_subjects == ()


# --------------------------------------------------------------------------
# configuration that would break at connection time rather than at startup
# --------------------------------------------------------------------------


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("IBKR_MCP_AUTH", "true")
    monkeypatch.setenv("IBKR_MCP_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("IBKR_MCP_RESOURCE_URL", RESOURCE)
    monkeypatch.setenv("IBKR_MCP_ALLOWED_SUBJECTS", ME)
    monkeypatch.delenv("IBKR_MCP_AUTH_JWKS_URL", raising=False)
    monkeypatch.delenv("IBKR_MCP_AUTH_AUDIENCE", raising=False)
    monkeypatch.delenv("IBKR_MCP_PATH", raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "name,value",
    [
        ("IBKR_MCP_AUTH_ISSUER", "http://idp.example.com/"),
        ("IBKR_MCP_AUTH_JWKS_URL", "http://idp.example.com/keys"),
        ("IBKR_MCP_RESOURCE_URL", "http://risk.example.com/mcp"),
    ],
)
def test_a_plaintext_auth_url_refuses_to_start(auth_env, name, value):
    """The JWKS URL is the one that matters and the one that looks harmless:
    anyone who can answer it hands this server a signing key of its own."""
    auth_env.setenv(name, value)
    with pytest.raises(ValueError, match="must be https"):
        load_settings()


def test_loopback_may_be_plaintext(auth_env):
    """Only so the end-to-end auth test can stand a stub provider up locally."""
    auth_env.setenv("IBKR_MCP_AUTH_ISSUER", "http://127.0.0.1:9999/")
    auth_env.setenv("IBKR_MCP_AUTH_JWKS_URL", "http://127.0.0.1:9999/keys")
    auth_env.setenv("IBKR_MCP_RESOURCE_URL", "http://127.0.0.1:8765/mcp")
    assert load_settings().mcp_auth is True


def test_a_resource_url_that_misses_the_endpoint_refuses_to_start(auth_env):
    """Claude compares the published `resource` against the URL typed into the
    connector dialog, character for character. A resource of
    https://risk.example.com on a server serving /mcp is internally consistent,
    passes every other check here, and fails at connection time with nothing
    pointing at why."""
    auth_env.setenv("IBKR_MCP_RESOURCE_URL", "https://risk.example.com")
    with pytest.raises(ValueError, match="MCP endpoint is served at"):
        load_settings()


def test_a_custom_endpoint_path_must_be_matched_by_the_resource(auth_env):
    auth_env.setenv("IBKR_MCP_PATH", "/risk")
    auth_env.setenv("IBKR_MCP_RESOURCE_URL", "https://risk.example.com/mcp")
    with pytest.raises(ValueError, match="MCP endpoint is served at"):
        load_settings()
    auth_env.setenv("IBKR_MCP_RESOURCE_URL", "https://risk.example.com/risk")
    assert load_settings().mcp_resource_url == "https://risk.example.com/risk"


async def test_a_clock_a_few_seconds_out_does_not_lock_you_out(verifier, keypair):
    """Zero tolerance rejects a token that is valid everywhere else because a
    VPS drifted, which reads as an auth bug and gets 'fixed' by disabling a
    check. A token well past the tolerance is still refused."""
    assert await verifier.verify_token(token(keypair, exp_delta=-5)) is not None
    assert await verifier.verify_token(token(keypair, exp_delta=-120)) is None
