"""
OralAID API: FastAPI backend for the OralAID screening platform.

Run locally:
    cd oralaid_app && uvicorn app_fastapi:app --reload --port 8000

Configuration (environment variables):
    ORALAID_ENV              "production" enables strict checks
    ORALAID_SECRET_KEY       JWT / URL-signing key (required in production)
    DATABASE_URL             SQLAlchemy URL; defaults to SQLite in ORALAID_DATA_DIR
    ORALAID_DATA_DIR         where uploads and result images are written
    ORALAID_CORS_ORIGINS     comma-separated origins, e.g. https://oralaid.ai
    ORALAID_MODEL_DIR        directory holding the .pth checkpoints
    ORALAID_ADMIN_EMAIL / ORALAID_ADMIN_PASSWORD   bootstrap admin account
"""
import csv
import io
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

import security as sec
from inference.engine import HIGH_RISK_THRESHOLD, RISK_TIERS, SUSPICION_THRESHOLD, get_engine, load_image
from inference.registry import MODELS
from models.validator import is_oral_image

logging.basicConfig(level=os.environ.get("ORALAID_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("oralaid.api")

# ── configuration ──────────────────────────────────────────────────────────

APP_DIR = Path(__file__).resolve().parent
IS_PROD = os.environ.get("ORALAID_ENV") == "production"
DATA_DIR = Path(os.environ.get("ORALAID_DATA_DIR", APP_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES_PER_CASE = 6
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DEMO_RETENTION_S = 3600
CORS_ORIGINS = [o.strip() for o in os.environ.get("ORALAID_CORS_ORIGINS", "*").split(",") if o.strip()]

if IS_PROD:
    if not os.environ.get("ORALAID_SECRET_KEY"):
        raise RuntimeError("ORALAID_SECRET_KEY must be set when ORALAID_ENV=production")
    if CORS_ORIGINS == ["*"]:
        raise RuntimeError("Set ORALAID_CORS_ORIGINS to your site origin(s) in production")

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'oralaid.db'}")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine_db = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_db)
Base = declarative_base()

# ── database models ────────────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    organization = Column(String)
    specialty = Column(String)
    role = Column(String, default="clinician")  # clinician | admin
    plan = Column(String, default="starter")
    hashed_password = Column(String, nullable=False)
    token_version = Column(Integer, default=0)  # bump to revoke every session
    failed_logins = Column(Integer, default=0)
    locked_until = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    name = Column(String)
    prefix = Column(String)
    key_hash = Column(String, unique=True, index=True)
    last_used_at = Column(DateTime)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    reference = Column(String, index=True)  # clinic's own MRN / reference
    name = Column(String)
    age = Column(Integer)
    sex = Column(String)
    risk_factors = Column(Text)  # JSON list
    created_at = Column(DateTime, default=datetime.utcnow)


class Case(Base):
    __tablename__ = "cases"
    id = Column(String, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True)
    site = Column(String)
    notes = Column(Text)
    status = Column(String, default="open")  # open | referred | closed
    probability = Column(Float)
    tier = Column(String, index=True)
    result = Column(Text)  # JSON list of per-image results
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    patient = relationship("Patient")
    feedback = relationship("Feedback", uselist=False, back_populates="case", cascade="all, delete-orphan")


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), unique=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    verdict = Column(String)  # agree | disagree | unsure
    clinical_impression = Column(String)  # normal | benign | opmd | malignant
    histopathology = Column(String)
    explanation_rating = Column(Integer)  # 1-5
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    case = relationship("Case", back_populates="feedback")


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    action = Column(String, index=True)
    target = Column(String)
    ip = Column(String)
    detail = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    kind = Column(String, index=True)
    name = Column(String)
    email = Column(String)
    organization = Column(String)
    message = Column(Text)
    meta = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine_db)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def audit(db: Session, request: Request, user_id, action, target=None, **detail):
    db.add(AuditEvent(user_id=user_id, action=action, target=target, ip=sec.client_ip(request),
                      detail=json.dumps(detail) if detail else None))


# ── auth dependencies ──────────────────────────────────────────────────────

bearer = HTTPBearer(auto_error=False)


def current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    x_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Accepts a session JWT (web app) or an API key (integrations)."""
    raw_key = x_api_key or (creds.credentials if creds and creds.credentials.startswith(sec.API_KEY_PREFIX) else None)
    if raw_key:
        key = None
        candidates = db.query(ApiKey).filter(ApiKey.revoked.is_(False)).all()
        for candidate in candidates:
            if sec.verify_api_key(raw_key, candidate.key_hash):
                key = candidate
                break
        if not key:
            raise HTTPException(401, "Invalid API key")
        key.last_used_at = datetime.utcnow()
        db.commit()
        sec.rate_limit(request, 120, 60, "apikey", key=str(key.id))
        return db.get(User, key.user_id)
    if not creds:
        raise HTTPException(401, "Not authenticated")
    payload = sec.decode_token(creds.credentials)
    user = db.get(User, int(payload["sub"]))
    if not user or payload.get("ver", 0) != (user.token_version or 0):
        raise HTTPException(401, "Session expired. Please sign in again.")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admin access required")
    return user


def user_json(u: User):
    return {
        "id": u.id,
        "email": u.email,
        "full_name": u.full_name,
        "organization": u.organization,
        "specialty": u.specialty,
        "role": u.role,
        "plan": u.plan,
        "created_at": u.created_at.isoformat(),
    }


# ── app ────────────────────────────────────────────────────────────────────

HIDE_DOCS = IS_PROD and os.environ.get("ORALAID_PUBLIC_DOCS") != "1"


@asynccontextmanager
async def lifespan(_app):
    startup()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="OralAID API",
    version="3.0.0",
    description="Explainable AI screening for oral potentially malignant disorders and oral cancer.",
    docs_url=None if HIDE_DOCS else "/docs",
    redoc_url=None if HIDE_DOCS else "/redoc",
    openapi_url=None if HIDE_DOCS else "/openapi.json",
)
app.add_middleware(sec.SecurityHeadersMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,  # bearer tokens, not cookies
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Retry-After"],
    max_age=600,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES * MAX_IMAGES_PER_CASE + 1_000_000:
        return JSONResponse({"detail": "Request too large"}, status_code=413, headers={"X-Request-ID": rid})
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
        response = JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    log.info("%s %s %s %dms rid=%s", request.method, request.url.path, response.status_code,
             int((time.time() - t0) * 1000), rid)
    return response


def _warm_engine():
    try:
        get_engine()
    except Exception:
        log.exception("model engine failed to load")


def _cleanup_demo_files():
    while True:
        cutoff = time.time() - DEMO_RETENTION_S
        for d in (UPLOAD_DIR, RESULT_DIR):
            for p in d.glob("demo_*"):
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
        time.sleep(600)


def startup():
    db = SessionLocal()
    admin_email = os.environ.get("ORALAID_ADMIN_EMAIL")
    admin_password = os.environ.get("ORALAID_ADMIN_PASSWORD")
    if admin_email and admin_password and not db.query(User).filter(User.email == admin_email.lower()).first():
        db.add(User(email=admin_email.lower(), full_name="Administrator", role="admin", plan="enterprise",
                    hashed_password=sec.hash_password(admin_password)))
        db.commit()
        log.info("admin account created for %s", admin_email)
    db.close()
    if os.environ.get("ORALAID_SKIP_MODEL") != "1":
        threading.Thread(target=_warm_engine, daemon=True).start()
    threading.Thread(target=_cleanup_demo_files, daemon=True).start()


def require_engine():
    eng = get_engine() if get_engine.cache_info().currsize else None
    if not eng or not eng.ready:
        raise HTTPException(503, "The AI model is still loading. Try again in a few seconds.", headers={"Retry-After": "10"})
    return eng


# ── serialisation ──────────────────────────────────────────────────────────


def _with_urls(img):
    out = dict(img)
    out["files"] = {k: sec.sign_file(v) for k, v in img["files"].items()}
    return out


def case_json(c: Case, detail=False):
    images = json.loads(c.result or "[]")
    data = {
        "id": c.id,
        "created_at": c.created_at.isoformat(),
        "site": c.site,
        "status": c.status,
        "probability": c.probability,
        "tier": c.tier,
        "threshold": SUSPICION_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        **{k: RISK_TIERS[c.tier][k] for k in ("label", "finding", "color")},
        "thumbnail": sec.sign_file(images[0]["files"]["original"]) if images else None,
        "image_count": len(images),
        "patient": {
            "id": c.patient.id,
            "name": c.patient.name,
            "reference": c.patient.reference,
            "age": c.patient.age,
            "sex": c.patient.sex,
            "risk_factors": json.loads(c.patient.risk_factors) if c.patient.risk_factors else [],
        } if c.patient else None,
        "feedback": {
            "verdict": c.feedback.verdict,
            "clinical_impression": c.feedback.clinical_impression,
            "histopathology": c.feedback.histopathology,
            "explanation_rating": c.feedback.explanation_rating,
            "comment": c.feedback.comment,
            "created_at": c.feedback.created_at.isoformat(),
        } if c.feedback else None,
    }
    if detail:
        data["notes"] = c.notes
        data["recommendation"] = RISK_TIERS[c.tier]["recommendation"]
        data["images"] = [_with_urls(i) for i in images]
    return data


# ── image handling ─────────────────────────────────────────────────────────


async def _read_images(files: List[UploadFile]):
    if not files:
        raise HTTPException(400, "Upload at least one intraoral image.")
    if len(files) > MAX_IMAGES_PER_CASE:
        raise HTTPException(400, f"Up to {MAX_IMAGES_PER_CASE} images per case.")
    images = []
    for f in files:
        raw = await f.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Each image must be 15 MB or smaller.")
        kind = sec.sniff_image(raw)
        if kind is None:
            raise HTTPException(415, "Only JPEG, PNG, WebP or HEIC images are accepted.")
        try:
            image = load_image(io.BytesIO(raw))
        except Exception:
            raise HTTPException(400, "The image could not be decoded.")
        if image.width * image.height > 60_000_000:
            raise HTTPException(413, "Image dimensions are too large.")
        ok, reason = is_oral_image(image, "cpu")
        if not ok:
            raise HTTPException(422, reason)
        images.append((kind, raw, image))
    return images


def _run_case(eng, images, case_id, with_lime=False):
    results = []
    for idx, (kind, raw, image) in enumerate(images):
        key = f"{case_id}_{idx}"
        (UPLOAD_DIR / f"{key}.{kind}").write_bytes(raw)
        r = eng.analyze(image, str(RESULT_DIR), key, with_lime=with_lime)
        r["index"] = idx
        results.append(r)
    # Screening logic: a case is as concerning as its most concerning image.
    worst = max(results, key=lambda r: r["probability"])
    return results, worst


# ── public endpoints ───────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    eng = get_engine() if get_engine.cache_info().currsize else None
    return {"status": "ok", "version": app.version, "model_ready": bool(eng and eng.ready)}


@app.get("/api/models")
def models_info():
    status = get_engine().status if get_engine.cache_info().currsize else {}
    return {
        "threshold": SUSPICION_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "models": [{**m, "runtime": status.get(m["id"])} for m in MODELS.values()],
    }


class LeadIn(BaseModel):
    kind: str = Field(pattern="^(demo|contact|deployment|waitlist-desktop|waitlist-hardware|newsletter)$")
    name: str | None = Field(default=None, max_length=120)
    email: EmailStr
    organization: str | None = Field(default=None, max_length=160)
    message: str | None = Field(default=None, max_length=4000)
    meta: dict | None = None
    website: str | None = None  # honeypot: real users never fill this


@app.post("/api/leads", status_code=201)
def create_lead(body: LeadIn, request: Request, db: Session = Depends(get_db)):
    sec.rate_limit(request, 10, 3600, "leads")
    if body.website:
        return {"ok": True}
    meta = json.dumps(body.meta or {})[:2000]
    db.add(Lead(kind=body.kind, name=body.name, email=body.email.lower(), organization=body.organization,
                message=body.message, meta=meta))
    db.commit()
    return {"ok": True}


@app.post("/api/demo/analyze")
async def demo_analyze(request: Request, file: UploadFile = File(...), eng=Depends(require_engine)):
    """Try-it-now endpoint for the website. No account; files are purged after an hour."""
    sec.rate_limit(request, 8, 3600, "demo")
    images = await _read_images([file])
    case_id = f"demo_{uuid.uuid4().hex}"
    _, worst = _run_case(eng, images, case_id)
    return {"id": case_id, **_with_urls(worst)}


# ── auth ───────────────────────────────────────────────────────────────────


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(min_length=1, max_length=120)
    organization: str | None = Field(default=None, max_length=160)
    specialty: str | None = Field(default=None, max_length=80)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=128)


@app.post("/api/auth/register", status_code=201)
def register(body: RegisterIn, request: Request, db: Session = Depends(get_db)):
    sec.rate_limit(request, 10, 3600, "register")
    problems = sec.password_problems(body.password, body.email)
    if problems:
        raise HTTPException(422, "Password must have " + ", ".join(problems) + ".")
    if db.query(User).filter(func.lower(User.email) == body.email.lower()).first():
        raise HTTPException(409, "An account with this email already exists.")
    user = User(email=body.email.lower(), full_name=body.full_name, organization=body.organization,
                specialty=body.specialty, hashed_password=sec.hash_password(body.password))
    db.add(user)
    db.flush()
    audit(db, request, user.id, "auth.register")
    db.commit()
    db.refresh(user)
    return {"token": sec.create_token(user.id, user.token_version or 0), "user": user_json(user)}


@app.post("/api/auth/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    sec.rate_limit(request, 20, 600, "login")
    sec.rate_limit(request, 10, 600, "login-email", key=body.email.lower())
    user = db.query(User).filter(func.lower(User.email) == body.email.lower()).first()
    if user and user.locked_until and user.locked_until > datetime.utcnow():
        raise HTTPException(423, "Too many failed attempts. Try again in a few minutes.")
    if not user or not sec.verify_password(body.password, user.hashed_password):
        if user:
            user.failed_logins = (user.failed_logins or 0) + 1
            if user.failed_logins >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=15)
                user.failed_logins = 0
            audit(db, request, user.id, "auth.login_failed")
            db.commit()
        raise HTTPException(401, "Incorrect email or password.")
    user.failed_logins = 0
    user.locked_until = None
    audit(db, request, user.id, "auth.login")
    db.commit()
    return {"token": sec.create_token(user.id, user.token_version or 0), "user": user_json(user)}


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return user_json(user)


class ProfileIn(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=120)
    organization: str | None = Field(default=None, max_length=160)
    specialty: str | None = Field(default=None, max_length=80)


@app.patch("/api/auth/me")
def update_me(body: ProfileIn, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(user, k, v)
    audit(db, request, user.id, "account.update")
    db.commit()
    return user_json(user)


class PasswordIn(BaseModel):
    current_password: str = Field(max_length=128)
    new_password: str = Field(min_length=10, max_length=128)


@app.post("/api/auth/password")
def change_password(body: PasswordIn, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not sec.verify_password(body.current_password, user.hashed_password):
        raise HTTPException(400, "Current password is incorrect.")
    problems = sec.password_problems(body.new_password, user.email)
    if problems:
        raise HTTPException(422, "Password must have " + ", ".join(problems) + ".")
    user.hashed_password = sec.hash_password(body.new_password)
    user.token_version = (user.token_version or 0) + 1
    audit(db, request, user.id, "auth.password_changed")
    db.commit()
    return {"token": sec.create_token(user.id, user.token_version), "user": user_json(user)}


@app.post("/api/auth/logout-all")
def logout_all(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.token_version = (user.token_version or 0) + 1
    audit(db, request, user.id, "auth.logout_all")
    db.commit()
    return {"ok": True}


# ── API keys ───────────────────────────────────────────────────────────────


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)


def key_json(k: ApiKey):
    return {"id": k.id, "name": k.name, "prefix": k.prefix, "revoked": k.revoked,
            "created_at": k.created_at.isoformat(),
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None}


@app.get("/api/keys")
def list_keys(user: User = Depends(current_user), db: Session = Depends(get_db)):
    keys = db.query(ApiKey).filter(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc()).all()
    return {"keys": [key_json(k) for k in keys]}


@app.post("/api/keys", status_code=201)
def create_key(body: ApiKeyIn, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if db.query(ApiKey).filter(ApiKey.user_id == user.id, ApiKey.revoked.is_(False)).count() >= 10:
        raise HTTPException(400, "Revoke an existing key before creating another (limit 10).")
    raw, prefix, digest = sec.new_api_key()
    k = ApiKey(user_id=user.id, name=body.name, prefix=prefix, key_hash=digest)
    db.add(k)
    db.flush()
    audit(db, request, user.id, "apikey.create", target=str(k.id))
    db.commit()
    return {**key_json(k), "key": raw}  # plaintext is shown exactly once


@app.delete("/api/keys/{key_id}")
def revoke_key(key_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    k = db.get(ApiKey, key_id)
    if not k or k.user_id != user.id:
        raise HTTPException(404, "Key not found")
    k.revoked = True
    audit(db, request, user.id, "apikey.revoke", target=str(k.id))
    db.commit()
    return {"ok": True}


# ── cases ──────────────────────────────────────────────────────────────────

ALLOWED_SITES = {
    "Buccal mucosa", "Tongue (lateral)", "Tongue (dorsal)", "Tongue (ventral)", "Floor of mouth",
    "Hard palate", "Soft palate", "Gingiva", "Lip", "Retromolar trigone", "Other",
}
# Labels a clinician can draw on a photo: every site, a few finer tongue areas, and "Lesion".
ANNOTATION_LABELS = ALLOWED_SITES | {"Tongue (tip)", "Lesion", "Teeth"}
MAX_SHAPES_PER_IMAGE = 50
MAX_POINTS_PER_SHAPE = 200


def parse_annotations(raw: str | None, image_count: int) -> dict:
    """
    Validate clinician-drawn regions. Input is JSON:
      [{"index": 0, "shapes": [{"type": "rect"|"polygon", "label": "Lip", "points": [[x, y], ...]}]}]
    Coordinates are fractions of the image (0..1) so they survive any resizing.
    Returns {image_index: [shape, ...]}.
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        assert isinstance(data, list)
    except (ValueError, AssertionError):
        raise HTTPException(422, "annotations must be a JSON list.")
    out: dict[int, list] = {}
    for entry in data[:image_count]:
        try:
            idx = int(entry["index"])
            shapes = entry.get("shapes", [])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(422, "Each annotation entry needs an index and shapes.")
        if not 0 <= idx < image_count:
            raise HTTPException(422, "Annotation index out of range.")
        clean = []
        for sh in shapes[:MAX_SHAPES_PER_IMAGE]:
            kind, label, pts = sh.get("type"), sh.get("label"), sh.get("points")
            if kind not in ("rect", "polygon") or label not in ANNOTATION_LABELS or not isinstance(pts, list):
                raise HTTPException(422, "Invalid annotation shape or label.")
            try:
                pts = [[round(min(max(float(x), 0.0), 1.0), 4), round(min(max(float(y), 0.0), 1.0), 4)]
                       for x, y in pts[:MAX_POINTS_PER_SHAPE]]
            except (TypeError, ValueError):
                raise HTTPException(422, "Annotation points must be [x, y] pairs.")
            if (kind == "rect" and len(pts) != 2) or (kind == "polygon" and len(pts) < 3):
                raise HTTPException(422, "Rectangles need 2 points, polygons at least 3.")
            clean.append({"type": kind, "label": label, "points": pts})
        out[idx] = clean
    return out


@app.post("/api/cases", status_code=201)
async def create_case(
    request: Request,
    files: List[UploadFile] = File(...),
    patient_name: str | None = Form(None, max_length=120),
    patient_reference: str | None = Form(None, max_length=64),
    patient_age: int | None = Form(None, ge=0, le=120),
    patient_sex: str | None = Form(None, pattern="^(female|male|other)$"),
    risk_factors: str | None = Form(None, max_length=500),
    site: str | None = Form(None),
    notes: str | None = Form(None, max_length=4000),
    annotations: str | None = Form(None, max_length=200_000),
    with_lime: bool = Form(False),
    user: User = Depends(current_user),
    eng=Depends(require_engine),
    db: Session = Depends(get_db),
):
    sec.rate_limit(request, 30, 60, "analyze", key=str(user.id))
    if site and site not in ALLOWED_SITES:
        raise HTTPException(422, "Unknown anatomical site.")
    factors = None
    if risk_factors:
        try:
            factors = [str(f)[:40] for f in json.loads(risk_factors)][:10]
        except (ValueError, TypeError):
            raise HTTPException(422, "risk_factors must be a JSON list of strings.")
    marks = parse_annotations(annotations, len(files))
    images = await _read_images(files)
    case_id = uuid.uuid4().hex[:16]
    results, worst = _run_case(eng, images, case_id, with_lime=with_lime)
    for idx, shapes in marks.items():
        results[idx]["annotations"] = shapes
    if not site:  # use the first labelled site, if any
        site = next((sh["label"] for shapes in marks.values() for sh in shapes if sh["label"] in ALLOWED_SITES), None)

    patient = None
    if patient_reference:
        patient = db.query(Patient).filter(Patient.owner_id == user.id, Patient.reference == patient_reference).first()
    if patient is None:
        patient = Patient(owner_id=user.id, reference=patient_reference, name=patient_name,
                          age=patient_age, sex=patient_sex, risk_factors=json.dumps(factors) if factors else None)
        db.add(patient)
        db.flush()

    case = Case(id=case_id, owner_id=user.id, patient_id=patient.id, site=site, notes=notes,
                probability=worst["probability"], tier=worst["tier"], result=json.dumps(results))
    db.add(case)
    audit(db, request, user.id, "case.create", target=case_id, tier=worst["tier"])
    db.commit()
    db.refresh(case)
    return case_json(case, detail=True)


def _case_query(db, user, tier=None, q=None, status=None, review=False):
    query = db.query(Case).filter(Case.owner_id == user.id)
    if review:  # flagged by the model but not yet reviewed by a clinician
        query = query.filter(Case.tier != "low", ~Case.feedback.has())
    if tier:
        query = query.filter(Case.tier == tier)
    if status:
        query = query.filter(Case.status == status)
    if q:
        like = f"%{q.lower()}%"
        query = query.join(Patient).filter(
            func.lower(Patient.name).like(like) | func.lower(Patient.reference).like(like) | Case.id.like(like)
        )
    return query


@app.get("/api/cases")
def list_cases(
    tier: str | None = Query(None, pattern="^(low|moderate|high)$"),
    status: str | None = Query(None, pattern="^(open|referred|closed)$"),
    q: str | None = Query(None, max_length=80),
    review: bool = False,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    query = _case_query(db, user, tier, q, status, review)
    total = query.count()
    rows = query.order_by(Case.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "cases": [case_json(c) for c in rows]}


@app.get("/api/cases/export.csv")
def export_cases(
    request: Request,
    tier: str | None = Query(None, pattern="^(low|moderate|high)$"),
    status: str | None = Query(None, pattern="^(open|referred|closed)$"),
    q: str | None = Query(None, max_length=80),
    review: bool = False,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    rows = _case_query(db, user, tier, q, status, review).order_by(Case.created_at.desc()).all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["case_id", "created_at", "patient_reference", "age", "sex", "site", "tier",
                "probability", "status", "clinician_verdict", "clinical_impression", "histopathology"])
    for c in rows:
        p, fb = c.patient, c.feedback
        w.writerow([c.id, c.created_at.isoformat(), p.reference if p else "", p.age if p else "",
                    p.sex if p else "", c.site or "", c.tier, c.probability, c.status,
                    fb.verdict if fb else "", fb.clinical_impression if fb else "", fb.histopathology if fb else ""])
    audit(db, request, user.id, "case.export", count=len(rows))
    db.commit()
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=oralaid-cases.csv"})


def _own_case(case_id, user, db) -> Case:
    c = db.get(Case, case_id)
    if not c or c.owner_id != user.id:
        raise HTTPException(404, "Case not found")
    return c


@app.get("/api/cases/{case_id}")
def get_case(case_id: str, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    c = _own_case(case_id, user, db)
    audit(db, request, user.id, "case.view", target=case_id)
    db.commit()
    return case_json(c, detail=True)


class CasePatch(BaseModel):
    status: str | None = Field(default=None, pattern="^(open|referred|closed)$")
    notes: str | None = Field(default=None, max_length=4000)


@app.patch("/api/cases/{case_id}")
def update_case(case_id: str, body: CasePatch, request: Request, user: User = Depends(current_user),
                db: Session = Depends(get_db)):
    c = _own_case(case_id, user, db)
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(c, k, v)
    audit(db, request, user.id, "case.update", target=case_id, **body.model_dump(exclude_none=True, exclude={"notes"}))
    db.commit()
    return case_json(c, detail=True)


class AnnotationsIn(BaseModel):
    annotations: list = Field(max_length=MAX_IMAGES_PER_CASE)


@app.put("/api/cases/{case_id}/annotations")
def save_annotations(case_id: str, body: AnnotationsIn, request: Request, user: User = Depends(current_user),
                     db: Session = Depends(get_db)):
    c = _own_case(case_id, user, db)
    images = json.loads(c.result or "[]")
    marks = parse_annotations(json.dumps(body.annotations), len(images))
    for img in images:
        if img["index"] in marks:
            img["annotations"] = marks[img["index"]]
    c.result = json.dumps(images)
    audit(db, request, user.id, "case.annotate", target=case_id,
          shapes=sum(len(v) for v in marks.values()))
    db.commit()
    return case_json(c, detail=True)


@app.post("/api/cases/{case_id}/lime")
def compute_lime(case_id: str, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db),
                 eng=Depends(require_engine)):
    """LIME takes 10-30 s on CPU, so the UI asks for it after the main result is on screen."""
    sec.rate_limit(request, 10, 60, "lime", key=str(user.id))
    c = _own_case(case_id, user, db)
    images = json.loads(c.result)
    for img in images:
        if "lime" in img["files"]:
            continue
        key = f"{c.id}_{img['index']}"
        source = next(UPLOAD_DIR.glob(f"{key}.*"), None)
        if not source:
            continue
        out = eng.explain_lime(load_image(source), str(RESULT_DIR), key)
        img["files"]["lime"] = out["file"]
        img["lime_top_regions"] = out["top_regions"]
    c.result = json.dumps(images)
    db.commit()
    return case_json(c, detail=True)


class FeedbackIn(BaseModel):
    verdict: str = Field(pattern="^(agree|disagree|unsure)$")
    clinical_impression: str | None = Field(default=None, pattern="^(normal|benign|opmd|malignant)$")
    histopathology: str | None = Field(default=None, max_length=200)
    explanation_rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=4000)


@app.put("/api/cases/{case_id}/feedback")
def save_feedback(case_id: str, body: FeedbackIn, request: Request, user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    c = _own_case(case_id, user, db)
    fb = c.feedback or Feedback(case_id=c.id, user_id=user.id)
    for k, v in body.model_dump().items():
        setattr(fb, k, v)
    fb.created_at = datetime.utcnow()
    db.add(fb)
    audit(db, request, user.id, "case.feedback", target=case_id, verdict=body.verdict)
    db.commit()
    db.refresh(c)
    return case_json(c, detail=True)


@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    c = _own_case(case_id, user, db)
    for p in list(RESULT_DIR.glob(f"{c.id}_*")) + list(UPLOAD_DIR.glob(f"{c.id}_*")):
        p.unlink(missing_ok=True)
    db.delete(c)
    audit(db, request, user.id, "case.delete", target=case_id)
    db.commit()
    return {"ok": True}


# ── patients ───────────────────────────────────────────────────────────────


@app.get("/api/patients")
def list_patients(q: str | None = Query(None, max_length=80), user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    query = db.query(Patient).filter(Patient.owner_id == user.id)
    if q:
        like = f"%{q.lower()}%"
        query = query.filter(func.lower(Patient.name).like(like) | func.lower(Patient.reference).like(like))
    patients = query.order_by(Patient.created_at.desc()).limit(200).all()
    counts = dict(db.query(Case.patient_id, func.count(Case.id)).filter(Case.owner_id == user.id)
                  .group_by(Case.patient_id).all())
    latest = {}
    for c in db.query(Case).filter(Case.owner_id == user.id).order_by(Case.created_at).all():
        latest[c.patient_id] = c
    return {"patients": [{
        "id": p.id, "name": p.name, "reference": p.reference, "age": p.age, "sex": p.sex,
        "risk_factors": json.loads(p.risk_factors) if p.risk_factors else [],
        "case_count": counts.get(p.id, 0),
        "latest_tier": latest[p.id].tier if p.id in latest else None,
        "latest_at": latest[p.id].created_at.isoformat() if p.id in latest else None,
    } for p in patients]}


@app.get("/api/patients/{patient_id}/timeline")
def patient_timeline(patient_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Longitudinal view: how suspicion probability changed across visits."""
    p = db.get(Patient, patient_id)
    if not p or p.owner_id != user.id:
        raise HTTPException(404, "Patient not found")
    cases = db.query(Case).filter(Case.patient_id == p.id).order_by(Case.created_at).all()
    points = [case_json(c) for c in cases]
    change = None
    if len(cases) >= 2:
        change = round(cases[-1].probability - cases[-2].probability, 4)
    return {"patient": {"id": p.id, "name": p.name, "reference": p.reference, "age": p.age, "sex": p.sex},
            "cases": points, "latest_change": change}


# ── analytics ──────────────────────────────────────────────────────────────


@app.get("/api/dashboard")
def dashboard(days: int = Query(30, ge=7, le=365), user: User = Depends(current_user), db: Session = Depends(get_db)):
    cases = db.query(Case).filter(Case.owner_id == user.id).all()
    since = datetime.utcnow() - timedelta(days=days)
    recent = [c for c in cases if c.created_at >= since]

    trend = {}
    for i in range(days):
        d = (datetime.utcnow() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        trend[d] = {"date": d, "low": 0, "moderate": 0, "high": 0}
    for c in recent:
        d = c.created_at.strftime("%Y-%m-%d")
        if d in trend:
            trend[d][c.tier] += 1

    reviewed = [c for c in cases if c.feedback]
    agreed = [c for c in reviewed if c.feedback.verdict == "agree"]
    ratings = [c.feedback.explanation_rating for c in reviewed if c.feedback.explanation_rating]
    timings, quality = [], []
    for c in recent:
        for img in json.loads(c.result or "[]"):
            timings.append(img.get("inference_ms", 0))
            quality.append(img.get("quality", {}).get("score", 0))

    sites = defaultdict(int)
    for c in cases:
        sites[c.site or "Unspecified"] += 1

    # Probability histogram helps clinics see how often the model sits near the threshold.
    buckets = [0] * 10
    for c in cases:
        buckets[min(int(c.probability * 10), 9)] += 1

    return {
        "totals": {
            "cases": len(cases),
            "patients": db.query(Patient).filter(Patient.owner_id == user.id).count(),
            "high": sum(c.tier == "high" for c in cases),
            "moderate": sum(c.tier == "moderate" for c in cases),
            "low": sum(c.tier == "low" for c in cases),
            "last_period": len(recent),
            "awaiting_review": sum(1 for c in cases if c.tier != "low" and not c.feedback),
            "referred": sum(c.status == "referred" for c in cases),
        },
        "agreement_rate": round(len(agreed) / len(reviewed), 3) if reviewed else None,
        "reviewed": len(reviewed),
        "explanation_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "avg_inference_ms": int(sum(timings) / len(timings)) if timings else None,
        "avg_quality": int(sum(quality) / len(quality)) if quality else None,
        "threshold": SUSPICION_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "trend": list(trend.values()),
        "histogram": [{"bin": f"{i / 10:.1f}", "count": n} for i, n in enumerate(buckets)],
        "sites": [{"site": k, "count": v} for k, v in sorted(sites.items(), key=lambda kv: -kv[1])][:8],
        "recent": [case_json(c) for c in sorted(cases, key=lambda c: c.created_at, reverse=True)[:6]],
        "queue": [case_json(c) for c in sorted(
            (c for c in cases if c.tier != "low" and not c.feedback), key=lambda c: -c.probability)[:6]],
    }


@app.get("/api/audit")
def my_audit(limit: int = Query(50, ge=1, le=200), user: User = Depends(current_user), db: Session = Depends(get_db)):
    events = (db.query(AuditEvent).filter(AuditEvent.user_id == user.id)
              .order_by(AuditEvent.created_at.desc()).limit(limit).all())
    return {"events": [{"action": e.action, "target": e.target, "ip": e.ip,
                        "created_at": e.created_at.isoformat()} for e in events]}


@app.get("/api/admin/overview")
def admin_overview(_: User = Depends(admin_user), db: Session = Depends(get_db)):
    leads = db.query(Lead).order_by(Lead.created_at.desc()).limit(100).all()
    return {
        "users": db.query(User).count(),
        "cases": db.query(Case).count(),
        "feedback": db.query(Feedback).count(),
        "leads": [{"kind": lead.kind, "name": lead.name, "email": lead.email, "organization": lead.organization,
                   "message": lead.message, "created_at": lead.created_at.isoformat()} for lead in leads],
    }


# ── files & frontend ───────────────────────────────────────────────────────


@app.get("/files/{name}")
def get_file(name: str, exp: int = 0, sig: str = ""):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Bad file name")
    if not sec.verify_file(name, exp, sig):
        raise HTTPException(403, "Link expired")
    path = RESULT_DIR / name
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


FRONTEND_DIST = Path(os.environ.get("ORALAID_FRONTEND_DIST", APP_DIR.parent / "frontend" / "dist"))
if (FRONTEND_DIST / "index.html").exists() and os.environ.get("ORALAID_SERVE_FRONTEND", "1") == "1":
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(("api/", "files/")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=  # nosec B104
                int(os.environ.get("PORT", "8000")))
