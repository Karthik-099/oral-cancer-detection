"""
Security primitives for the OralAID API: password hashing, JWTs, API keys,
signed file URLs, rate limiting and response hardening headers.
"""
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from jose import JWTError, jwt
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("oralaid.security")

SECRET_KEY = os.environ.get("ORALAID_SECRET_KEY") or secrets.token_urlsafe(48)
if not os.environ.get("ORALAID_SECRET_KEY"):
    log.warning("ORALAID_SECRET_KEY not set; using an ephemeral key (sessions reset on restart)")

ALGORITHM = "HS256"
ISSUER = "oralaid-api"
TOKEN_TTL_HOURS = int(os.environ.get("ORALAID_TOKEN_TTL_HOURS", "12"))
FILE_URL_TTL_S = int(os.environ.get("ORALAID_FILE_URL_TTL", "3600"))
API_KEY_PREFIX = "oak_"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── passwords ──────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(password, hashed)
    except ValueError:
        return False


def password_problems(password: str, email: str = "") -> list:
    problems = []
    if len(password) < 10:
        problems.append("at least 10 characters")
    if not any(c.isdigit() for c in password) or not any(c.isalpha() for c in password):
        problems.append("letters and numbers")
    if email and email.split("@")[0].lower() in password.lower():
        problems.append("not contain your email name")
    return problems


# ── tokens ─────────────────────────────────────────────────────────────────

def create_token(user_id: int, token_version: int = 0) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "ver": token_version,
        "iss": ISSUER,
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_TTL_HOURS),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], issuer=ISSUER)
    except JWTError:
        raise HTTPException(401, "Session expired. Please sign in again.")


# ── API keys ───────────────────────────────────────────────────────────────

def new_api_key():
    """Returns (plaintext, prefix, sha256). Only the hash is stored."""
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, raw[:12], hashlib.sha256(raw.encode()).hexdigest()


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ── signed file URLs ───────────────────────────────────────────────────────
# Clinical images are PHI. They are only reachable through short-lived HMAC
# signatures, so a leaked URL stops working and file names cannot be guessed.

def _file_sig(name: str, exp: int) -> str:
    return hmac.new(SECRET_KEY.encode(), f"{name}:{exp}".encode(), hashlib.sha256).hexdigest()[:32]


def sign_file(name: str, ttl: int = FILE_URL_TTL_S) -> str:
    exp = int(time.time()) + ttl
    return f"/files/{name}?exp={exp}&sig={_file_sig(name, exp)}"


def verify_file(name: str, exp: int, sig: str) -> bool:
    if exp < time.time():
        return False
    return hmac.compare_digest(_file_sig(name, exp), sig or "")


# ── rate limiting ──────────────────────────────────────────────────────────
# In-memory sliding window. Good for a single instance; put Redis behind this
# interface when running several replicas.

_hits = defaultdict(deque)
# Raise on staging for load tests (e.g. 100); never set below 1 in production.
RATE_LIMIT_SCALE = max(1.0, float(os.environ.get("ORALAID_RATE_LIMIT_SCALE", "1")))


def client_ip(request: Request) -> str:
    # Netlify / most proxies set X-Forwarded-For; trust only the first hop.
    fwd = request.headers.get("x-forwarded-for")
    if fwd and os.environ.get("ORALAID_TRUST_PROXY", "1") == "1":
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, limit: int, window_s: int, bucket: str, key: str = None):
    limit = int(limit * RATE_LIMIT_SCALE)
    k = f"{bucket}:{key or client_ip(request)}"
    now = time.time()
    q = _hits[k]
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        retry = int(window_s - (now - q[0])) + 1
        raise HTTPException(429, "Too many requests. Please slow down.", headers={"Retry-After": str(retry)})
    q.append(now)


def reset_rate_limits():
    _hits.clear()


# ── image upload validation ────────────────────────────────────────────────

ALLOWED_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",
}


def sniff_image(raw: bytes):
    """Trust the bytes, not the filename or Content-Type."""
    for magic, kind in ALLOWED_MAGIC.items():
        if raw.startswith(magic):
            if kind == "webp" and raw[8:12] != b"WEBP":
                continue
            return kind
    if raw[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return "heic"
    return None


# ── response hardening ─────────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        h = response.headers
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        h.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        h.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if request.url.scheme == "https":
            h.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
        if request.url.path.startswith("/api/"):
            h.setdefault("Cache-Control", "no-store")
        return response
