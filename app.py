import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64
import json
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import secrets as pysecrets

import logging
from logging.handlers import RotatingFileHandler
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Dict, Any, List

# ============================================================
# LOGGING
# ============================================================
class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict) and ctx:
            keys = sorted(ctx.keys())
            kv = " ".join([f"[{k}={ctx[k]}]" for k in keys if ctx[k] is not None and ctx[k] != ""])
            if kv:
                return f"{base} {kv}"
        return base


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("panphy")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = KVFormatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        log_path = os.environ.get("PANPHY_LOG_FILE", "panphy_app.log")
        fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass

    logger.propagate = False
    logger.info("Logging configured", extra={"ctx": {"component": "startup"}})
    return logger


LOGGER = setup_logging()

# =========================
# --- PAGE CONFIG ---
# =========================
st.set_page_config(
    page_title="PanPhy Skill Builder",
    page_icon="⚛️",
    layout="wide"
)

# =========================
# --- CONSTANTS ---
# =========================
MODEL_NAME = "gpt-5-mini"
CANVAS_BG_HEX = "#f8f9fa"
CANVAS_BG_RGB = (248, 249, 250)
MAX_IMAGE_WIDTH = 1024

STORAGE_BUCKET = "physics-bank"

# Rate limiting
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW_SECONDS = 60 * 60  # 1 hour

# Image limits
MAX_DIM_PX = 4000
QUESTION_MAX_MB = 5.0
MARKSCHEME_MAX_MB = 5.0
CANVAS_MAX_MB = 2.0

# AI question generation
AQA_GCSE_HIGHER_TOPICS = [
    "Energy stores and transfers",
    "Work done and power",
    "Efficiency",
    "Kinetic energy and momentum",
    "Heating and thermal energy",
    "Temperature vs energy",
    "Specific heat capacity",
    "Specific latent heat",
    "Gas pressure and temperature",
    "Density",
    "Changes of state",
    "Electric current, potential difference, resistance",
    "Series and parallel circuits",
    "I-V characteristics",
    "Resistors (including thermistors and LDRs)",
    "Power in circuits",
    "Domestic electricity and safety",
    "Static electricity",
    "Particle model and internal energy",
    "Forces and motion (Newton's laws)",
    "Resultant force and acceleration",
    "Stopping distance",
    "Moments, levers and gears",
    "Pressure in fluids",
    "Waves (properties and equations)",
    "Reflection and refraction",
    "Lenses and ray diagrams",
    "The electromagnetic spectrum",
    "Radioactivity basics",
    "Half-life and decay",
    "Nuclear fission and fusion",
    "Magnetism and electromagnets",
    "Motor effect",
    "Generators and transformers",
]
QUESTION_TYPES = ["Calculation", "Explanation", "Practical/Methods", "Graph/Analysis", "Mixed"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

# ============================================================
# DATABASE DDLs
#   IMPORTANT: avoid $$ PL/pgSQL blocks inside app DDL to prevent split/execution issues.
# ============================================================
RATE_LIMITS_DDL = """
create table if not exists public.rate_limits (
  student_id text primary key,
  submission_count int not null default 0,
  window_start_time timestamptz not null default now()
);
create index if not exists idx_rate_limits_window_start_time
  on public.rate_limits (window_start_time);
""".strip()

# NOTE: No trigger/function here. Keeping DDL simple avoids "unterminated dollar-quoted string" errors.
QUESTION_BANK_DDL = """
create table if not exists public.question_bank_v1 (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  source text not null check (source in ('teacher','ai_generated')),
  created_by text,

  assignment_name text not null,
  question_label text not null,
  max_marks int not null check (max_marks > 0),
  tags jsonb,

  question_text text,
  question_image_path text,

  markscheme_text text,
  markscheme_image_path text,

  is_active boolean not null default true
);

create unique index if not exists uq_question_bank_source_assignment_label
  on public.question_bank_v1 (source, assignment_name, question_label);

create index if not exists idx_question_bank_assignment
  on public.question_bank_v1 (assignment_name);

create index if not exists idx_question_bank_source
  on public.question_bank_v1 (source);

create index if not exists idx_question_bank_active
  on public.question_bank_v1 (is_active);
""".strip()

# =========================
# --- OPENAI CLIENT (CACHED) ---
# =========================
@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


try:
    client = get_client()
    AI_READY = True
except Exception as e:
    st.error("⚠️ OpenAI API Key missing or invalid in Streamlit Secrets!")
    AI_READY = False
    LOGGER.error("OpenAI client init failed", extra={"ctx": {"component": "openai", "error": type(e).__name__}})

# =========================
# --- SUPABASE STORAGE CLIENT (CACHED) ---
# =========================
@st.cache_resource
def get_supabase_client():
    url = (st.secrets.get("SUPABASE_URL", "") or "").strip()
    key = (st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        LOGGER.error("Supabase client init failed", extra={"ctx": {"component": "supabase", "error": type(e).__name__}})
        return None


def supabase_ready() -> bool:
    return get_supabase_client() is not None

# =========================
# --- SESSION STATE ---
# =========================
def _ss_init(k: str, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_init("canvas_key", 0)
_ss_init("feedback", None)
_ss_init("anon_id", pysecrets.token_hex(4))
_ss_init("db_last_error", "")
_ss_init("db_table_ready", False)
_ss_init("bank_table_ready", False)
_ss_init("is_teacher", False)

# Canvas robustness cache
_ss_init("last_canvas_image_data", None)

# Question selection cache
_ss_init("selected_qid", None)
_ss_init("cached_q_row", None)
_ss_init("cached_question_img", None)
_ss_init("cached_q_path", None)
_ss_init("cached_ms_path", None)

# AI generator draft cache (teacher-only)
_ss_init("ai_draft", None)

# ============================================================
#  ROBUST DATABASE LAYER
# ============================================================
def get_db_driver_type():
    try:
        import psycopg  # noqa: F401
        return "psycopg"
    except ImportError:
        try:
            import psycopg2  # noqa: F401
            return "psycopg2"
        except ImportError:
            return None


def _normalize_db_url(db_url: str) -> str:
    u = (db_url or "").strip()
    if not u:
        return ""
    if u.startswith("postgres://"):
        u = u.replace("postgres://", "postgresql://", 1)

    driver = get_db_driver_type()
    if driver == "psycopg":
        if u.startswith("postgresql://") and "psycopg" not in u:
            u = u.replace("postgresql://", "postgresql+psycopg://", 1)
    elif driver == "psycopg2":
        if u.startswith("postgresql://") and "psycopg2" not in u:
            u = u.replace("postgresql://", "postgresql+psycopg2://", 1)
    return u


@st.cache_resource
def _cached_engine(url: str):
    return create_engine(url, pool_pre_ping=True)


def get_db_engine():
    raw_url = st.secrets.get("DATABASE_URL", "")
    url = _normalize_db_url(raw_url)
    if not url:
        return None
    if not get_db_driver_type():
        return None
    try:
        return _cached_engine(url)
    except Exception as e:
        st.session_state["db_last_error"] = f"DB Engine Error: {type(e).__name__}: {e}"
        LOGGER.error("DB engine creation failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})
        return None


def db_ready() -> bool:
    return get_db_engine() is not None


def _split_sql_statements(sql_blob: str) -> List[str]:
    """
    Split SQL blob into statements at semicolons, but ignore semicolons in:
    - single-quoted strings
    - double-quoted identifiers
    This is enough for our simple DDL blobs (no $$ blocks in-app).
    """
    s = sql_blob or ""
    out: List[str] = []
    buf: List[str] = []
    in_sq = False
    in_dq = False
    esc = False

    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue

        if ch == "\\":
            buf.append(ch)
            if in_sq:
                esc = True
            continue

        if ch == "'" and not in_dq:
            in_sq = not in_sq
            buf.append(ch)
            continue

        if ch == '"' and not in_sq:
            in_dq = not in_dq
            buf.append(ch)
            continue

        if ch == ";" and (not in_sq) and (not in_dq):
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _exec_sql_many(conn, sql_blob: str):
    for stmt in _split_sql_statements(sql_blob):
        conn.execute(text(stmt))


def ensure_attempts_table():
    if st.session_state.get("db_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return

    ddl_create = """
    create table if not exists public.physics_attempts_v1 (
      id bigserial primary key,
      created_at timestamptz not null default now(),
      student_id text not null,
      question_key text not null,
      mode text not null,
      marks_awarded int not null,
      max_marks int not null,
      summary text,
      feedback_points jsonb,
      next_steps jsonb
    );
    """

    ddl_alter = """
    alter table public.physics_attempts_v1
      add column if not exists readback_type text;
    alter table public.physics_attempts_v1
      add column if not exists readback_markdown text;
    alter table public.physics_attempts_v1
      add column if not exists readback_warnings jsonb;
    """

    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, ddl_create)
            _exec_sql_many(conn, ddl_alter)
        st.session_state["db_last_error"] = ""
        st.session_state["db_table_ready"] = True
        LOGGER.info("Attempts table ready", extra={"ctx": {"component": "db", "table": "physics_attempts_v1"}})
    except Exception as e:
        st.session_state["db_last_error"] = f"Table Creation Error: {type(e).__name__}: {e}"
        st.session_state["db_table_ready"] = False
        LOGGER.error("Attempts table ensure failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})


def ensure_rate_limits_table():
    eng = get_db_engine()
    if eng is None:
        return
    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, RATE_LIMITS_DDL)
        LOGGER.info("Rate limits table ready", extra={"ctx": {"component": "db", "table": "rate_limits"}})
    except Exception as e:
        st.session_state["db_last_error"] = f"Rate Limits Table Error: {type(e).__name__}: {e}"
        LOGGER.error("Rate limits table ensure failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})


def ensure_question_bank_table():
    if st.session_state.get("bank_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return
    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, QUESTION_BANK_DDL)
        st.session_state["bank_table_ready"] = True
        st.session_state["db_last_error"] = ""
        LOGGER.info("Question bank table ready", extra={"ctx": {"component": "db", "table": "question_bank_v1"}})
    except Exception as e:
        st.session_state["db_last_error"] = f"Question Bank Table Error: {type(e).__name__}: {e}"
        st.session_state["bank_table_ready"] = False
        LOGGER.error("Question bank table ensure failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})

# ============================================================
# RATE LIMITING (Per Student, stored in Postgres)
# ============================================================
def _effective_student_id(student_id: str) -> str:
    sid = (student_id or "").strip()
    if sid:
        return sid
    return f"anon_{st.session_state['anon_id']}"


def _format_reset_time(dt_utc: datetime) -> str:
    try:
        tz = ZoneInfo("Europe/London")
        local = dt_utc.astimezone(tz)
        return local.strftime("%H:%M on %d %b %Y")
    except Exception:
        return dt_utc.strftime("%H:%M UTC on %d %b %Y")


def _check_rate_limit_db(student_id: str) -> Tuple[bool, int, str]:
    if st.session_state.get("is_teacher", False):
        return True, RATE_LIMIT_MAX, ""

    eng = get_db_engine()
    if eng is None:
        return True, RATE_LIMIT_MAX, ""

    ensure_rate_limits_table()

    sid = (student_id or "").strip() or f"anon_{st.session_state['anon_id']}"
    now_utc = datetime.now(timezone.utc)

    with eng.begin() as conn:
        row = conn.execute(
            text("""
                select student_id, submission_count, window_start_time
                from public.rate_limits
                where student_id = :sid
                for update
            """),
            {"sid": sid},
        ).mappings().first()

        if not row:
            conn.execute(
                text("""
                    insert into public.rate_limits (student_id, submission_count, window_start_time)
                    values (:sid, 0, now())
                """),
                {"sid": sid},
            )
            submission_count = 0
            window_start = now_utc
        else:
            submission_count = int(row["submission_count"] or 0)
            window_start = row["window_start_time"]
            if isinstance(window_start, datetime):
                if window_start.tzinfo is None:
                    window_start = window_start.replace(tzinfo=timezone.utc)
                else:
                    window_start = window_start.astimezone(timezone.utc)
            else:
                window_start = now_utc

        elapsed = (now_utc - window_start).total_seconds()
        if elapsed >= RATE_LIMIT_WINDOW_SECONDS:
            conn.execute(
                text("""
                    update public.rate_limits
                    set submission_count = 0,
                        window_start_time = now()
                    where student_id = :sid
                """),
                {"sid": sid},
            )
            submission_count = 0
            window_start = now_utc

        remaining = max(0, RATE_LIMIT_MAX - submission_count)
        reset_time_utc = window_start + timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)
        allowed = submission_count < RATE_LIMIT_MAX
        reset_str = _format_reset_time(reset_time_utc)
        return allowed, remaining, reset_str


@st.cache_data(ttl=15)
def check_rate_limit_cached(student_id: str, _fp: str) -> Tuple[bool, int, str]:
    # Cached "display" check only (not used for enforcement).
    try:
        return _check_rate_limit_db(student_id)
    except Exception:
        return True, RATE_LIMIT_MAX, ""


def check_rate_limit(student_id: str) -> Tuple[bool, int, str]:
    fp = (st.secrets.get("DATABASE_URL", "") or "")[:40]
    return check_rate_limit_cached(student_id, fp)


def increment_rate_limit(student_id: str):
    if st.session_state.get("is_teacher", False):
        return

    eng = get_db_engine()
    if eng is None:
        return

    ensure_rate_limits_table()
    sid = (student_id or "").strip() or f"anon_{st.session_state['anon_id']}"

    try:
        with eng.begin() as conn:
            conn.execute(
                text("""
                    insert into public.rate_limits (student_id, submission_count, window_start_time)
                    values (:sid, 0, now())
                    on conflict (student_id) do nothing
                """),
                {"sid": sid},
            )
            conn.execute(
                text("""
                    update public.rate_limits
                    set submission_count = submission_count + 1
                    where student_id = :sid
                """),
                {"sid": sid},
            )
    except Exception:
        pass

    try:
        check_rate_limit_cached.clear()
    except Exception:
        pass

# ============================================================
# STORAGE HELPERS
# ============================================================
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "untitled"


def _clean_storage_path(path: str) -> str:
    if not isinstance(path, str):
        return ""
    p = path.strip().lstrip("/")
    p = p.replace("\\", "/")
    p = re.sub(r"/{2,}", "/", p)
    return p


def upload_to_storage(path: str, file_bytes: bytes, content_type: str) -> bool:
    sb = get_supabase_client()
    if sb is None:
        st.session_state["db_last_error"] = "Supabase Storage not configured."
        return False

    p = _clean_storage_path(path)
    if not p:
        st.session_state["db_last_error"] = "Storage Upload Error: empty path."
        return False

    # Header values must be strings (avoid booleans).
    file_options = {
        "contentType": str(content_type),
        "content-type": str(content_type),
        "cacheControl": "3600",
        "cache-control": "3600",
        "upsert": "true",
        "x-upsert": "true",
    }

    try:
        try:
            res = sb.storage.from_(STORAGE_BUCKET).upload(p, file_bytes, file_options)
        except TypeError:
            res = sb.storage.from_(STORAGE_BUCKET).upload(path=p, file=file_bytes, file_options=file_options)

        err = None
        if hasattr(res, "error"):
            err = getattr(res, "error")
        elif isinstance(res, dict):
            err = res.get("error")
        if err:
            raise RuntimeError(str(err))
        return True
    except Exception as e:
        st.session_state["db_last_error"] = f"Storage Upload Error: {type(e).__name__}: {e}"
        LOGGER.error(
            "Storage upload failed",
            extra={"ctx": {"component": "storage", "op": "upload", "path": p, "error": type(e).__name__}},
        )
        return False


def download_from_storage(path: str) -> bytes:
    sb = get_supabase_client()
    if sb is None:
        return b""

    p = _clean_storage_path(path)
    if not p:
        return b""

    try:
        res = sb.storage.from_(STORAGE_BUCKET).download(p)

        if isinstance(res, (bytes, bytearray)):
            return bytes(res)

        if hasattr(res, "data") and res.data is not None:
            if isinstance(res.data, (bytes, bytearray)):
                return bytes(res.data)

        if hasattr(res, "content") and res.content is not None:
            if isinstance(res.content, (bytes, bytearray)):
                return bytes(res.content)

        if hasattr(res, "read"):
            try:
                out = res.read()
                if isinstance(out, (bytes, bytearray)):
                    return bytes(out)
            except Exception:
                pass

        return b""
    except Exception as e:
        st.session_state["db_last_error"] = f"Storage Download Error: {type(e).__name__}: {e}"
        LOGGER.error(
            "Storage download failed",
            extra={"ctx": {"component": "storage", "op": "download", "path": p, "error": type(e).__name__}},
        )
        return b""


@st.cache_data(ttl=300)
def cached_download_from_storage(path: str, _fp: str = "") -> bytes:
    return download_from_storage(path)


def bytes_to_pil(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    img.load()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def safe_bytes_to_pil(img_bytes: bytes) -> Optional[Image.Image]:
    if not img_bytes:
        return None
    try:
        return bytes_to_pil(img_bytes)
    except Exception as e:
        LOGGER.error(
            "Failed to decode image bytes",
            extra={"ctx": {"component": "image", "error": type(e).__name__}},
        )
        return None

# ============================================================
# FILE SIZE VALIDATION + COMPRESSION
# ============================================================
def _human_mb(num_bytes: int) -> str:
    return f"{(num_bytes / (1024 * 1024)):.1f}MB"


def validate_image_file(file_bytes: bytes, max_mb: float, _purpose: str) -> Tuple[bool, str]:
    if not file_bytes:
        return False, "No image data received."

    size_bytes = len(file_bytes)
    max_bytes = int(max_mb * 1024 * 1024)

    try:
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
    except Exception:
        return False, "Invalid image file. Please upload a valid PNG/JPG."

    if w > MAX_DIM_PX or h > MAX_DIM_PX:
        return False, f"Image dimensions too large ({w}x{h}). Max allowed is {MAX_DIM_PX}x{MAX_DIM_PX}px."

    if size_bytes <= max_bytes:
        return True, ""

    return False, f"Image too large ({_human_mb(size_bytes)}). Please use an image under {max_mb:.0f}MB."


def _encode_image_bytes(img: Image.Image, fmt: str, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _compress_bytes_to_limit(
    file_bytes: bytes,
    max_mb: float,
    _purpose: str,
    prefer_fmt: Optional[str] = None,
) -> Tuple[bool, bytes, str, str]:
    max_bytes = int(max_mb * 1024 * 1024)
    size_bytes = len(file_bytes)

    if size_bytes > int(max_bytes * 1.30):
        return False, b"", "", f"Image too large ({_human_mb(size_bytes)}). Please use an image under {max_mb:.0f}MB."

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except Exception:
        return False, b"", "", "Invalid image file. Please upload a valid PNG/JPG."

    w, h = img.size
    if w > MAX_DIM_PX or h > MAX_DIM_PX:
        return False, b"", "", f"Image dimensions too large ({w}x{h}). Max allowed is {MAX_DIM_PX}x{MAX_DIM_PX}px."

    in_fmt = (img.format or "").upper()
    target_fmt = (prefer_fmt or "JPEG").upper()
    if in_fmt in ("JPG", "JPEG"):
        target_fmt = "JPEG"

    best_bytes = None
    best_quality = None
    for q in [85, 80, 75, 70, 65, 60, 55, 50]:
        out = _encode_image_bytes(img, target_fmt, quality=q)
        if len(out) <= max_bytes:
            best_bytes = out
            best_quality = q
            break

    if best_bytes is None:
        w, h = img.size
        scale = 0.9
        for _ in range(4):
            img2 = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            for q in [75, 70, 65, 60, 55, 50]:
                out = _encode_image_bytes(img2, target_fmt, quality=q)
                if len(out) <= max_bytes:
                    best_bytes = out
                    best_quality = q
                    img = img2
                    break
            if best_bytes is not None:
                break
            scale *= 0.9

    if best_bytes is None:
        return False, b"", "", f"Image too large ({_human_mb(size_bytes)}) and could not be compressed under {max_mb:.0f}MB."

    ct = "image/jpeg" if target_fmt == "JPEG" else "image/png"
    LOGGER.warning(
        "Image compression applied",
        extra={"ctx": {"component": "image", "from": _human_mb(size_bytes), "to": _human_mb(len(best_bytes)), "quality": best_quality}},
    )
    return True, best_bytes, ct, ""

# ============================================================
# CANVAS HELPERS
# ============================================================
def canvas_has_ink(image_data: np.ndarray) -> bool:
    """
    More reliable detection (especially for light pencil strokes / iPad).
    Uses per-channel max difference against background and counts pixels.
    """
    if image_data is None:
        return False
    try:
        arr = np.asarray(image_data)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    except Exception:
        return False

    if arr.ndim != 3 or arr.shape[2] < 3:
        return False

    rgb = arr[:, :, :3]
    bg = np.array(CANVAS_BG_RGB, dtype=np.uint8)

    diff = np.max(np.abs(rgb.astype(np.int16) - bg.astype(np.int16)), axis=2)
    ink_pixels = int(np.count_nonzero(diff > 10))
    return ink_pixels >= 25


def preprocess_canvas_image(image_data: np.ndarray) -> Image.Image:
    raw_img = Image.fromarray(np.asarray(image_data).astype("uint8"))
    if raw_img.mode == "RGBA":
        white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
        white_bg.paste(raw_img, mask=raw_img.split()[3])
        img = white_bg
    else:
        img = raw_img.convert("RGB")
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / img.width
        img = img.resize((MAX_IMAGE_WIDTH, max(1, int(img.height * ratio))))
    return img

# ============================================================
# JSON HELPERS
# ============================================================
def encode_image(image_pil: Image.Image) -> str:
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def safe_parse_json(text_str: str):
    try:
        return json.loads(text_str)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text_str, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def clamp_int(value, lo, hi, default=0):
    try:
        v = int(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))

# ============================================================
# MARKDOWN RENDER HELPERS
# ============================================================
def render_md_box(title: str, md_text: str, caption: str = "", empty_text: str = ""):
    st.markdown(f"**{title}**")
    with st.container(border=True):
        txt = (md_text or "").strip()
        if txt:
            st.markdown(txt)
        else:
            st.caption(empty_text or "No content.")
    if caption:
        st.caption(caption)

# ============================================================
# PROGRESS INDICATORS
# ============================================================
def _run_ai_with_progress(task_fn, ctx: dict, typical_range: str, est_seconds: float) -> dict:
    with st.status(f"Processing… (typically {typical_range})", expanded=True) as status:
        progress = st.progress(0)
        start = time.monotonic()

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(task_fn)
            while not fut.done():
                elapsed = time.monotonic() - start
                frac = min(0.95, max(0.02, elapsed / max(1e-6, est_seconds)))
                progress.progress(int(frac * 100))
                time.sleep(0.12)

            report = fut.result()

        progress.progress(100)
        status.update(label="✓ Done", state="complete", expanded=False)

    return report

# ============================================================
# DB OPERATIONS (attempts + question bank)
# ============================================================
def insert_attempt(student_id: str, question_key: str, report: dict, mode: str):
    eng = get_db_engine()
    if eng is None:
        return
    ensure_attempts_table()

    sid = (student_id or "").strip() or f"anon_{st.session_state['anon_id']}"

    m_awarded = int(report.get("marks_awarded", 0))
    m_max = int(report.get("max_marks", 1))
    summ = str(report.get("summary", ""))[:1000]
    fb_json = json.dumps(report.get("feedback_points", [])[:6])
    ns_json = json.dumps(report.get("next_steps", [])[:6])

    rb_type = str(report.get("readback_type", "") or "")[:40]
    rb_md = str(report.get("readback_markdown", "") or "")[:8000]
    rb_warn = json.dumps(report.get("readback_warnings", [])[:6])

    query = """
        insert into public.physics_attempts_v1
        (student_id, question_key, mode, marks_awarded, max_marks, summary, feedback_points, next_steps,
         readback_type, readback_markdown, readback_warnings)
        values
        (:student_id, :question_key, :mode, :marks_awarded, :max_marks, :summary,
         CAST(:feedback_points AS jsonb), CAST(:next_steps AS jsonb),
         :readback_type, :readback_markdown, CAST(:readback_warnings AS jsonb))
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(query), {
                "student_id": sid,
                "question_key": question_key,
                "mode": mode,
                "marks_awarded": m_awarded,
                "max_marks": m_max,
                "summary": summ,
                "feedback_points": fb_json,
                "next_steps": ns_json,
                "readback_type": rb_type if rb_type else None,
                "readback_markdown": rb_md if rb_md else None,
                "readback_warnings": rb_warn
            })
        st.session_state["db_last_error"] = ""
    except Exception as e:
        st.session_state["db_last_error"] = f"Insert Error: {type(e).__name__}: {e}"


@st.cache_data(ttl=20)
def load_attempts_df_cached(_fp: str, limit: int = 5000) -> pd.DataFrame:
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_attempts_table()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("""
                select created_at, student_id, question_key, mode, marks_awarded, max_marks, readback_type
                from public.physics_attempts_v1
                order by created_at desc
                limit :limit
            """),
            conn,
            params={"limit": int(limit)},
        )
    if not df.empty:
        df["marks_awarded"] = pd.to_numeric(df["marks_awarded"], errors="coerce").fillna(0).astype(int)
        df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce").fillna(0).astype(int)
    return df


def load_attempts_df(limit: int = 5000) -> pd.DataFrame:
    fp = (st.secrets.get("DATABASE_URL", "") or "")[:40]
    try:
        return load_attempts_df_cached(fp, limit=limit)
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Error: {type(e).__name__}: {e}"
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_question_bank_df_cached(_fp: str, limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
    """
    NOTE: This returns a *summary* table for listing/filtering, so we include
    light metadata + tags/question_text (for search). Full row is loaded by id.
    """
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_question_bank_table()
    where = "" if include_inactive else "where is_active = true"
    with eng.connect() as conn:
        df = pd.read_sql(
            text(f"""
                select
                  id, created_at, updated_at,
                  source, assignment_name, question_label,
                  max_marks, tags, question_text,
                  is_active
                from public.question_bank_v1
                {where}
                order by created_at desc
                limit :limit
            """),
            conn,
            params={"limit": int(limit)},
        )
    return df


def load_question_bank_df(limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
    fp = (st.secrets.get("DATABASE_URL", "") or "")[:40]
    try:
        return load_question_bank_df_cached(fp, limit=limit, include_inactive=include_inactive)
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Question Bank Error: {type(e).__name__}: {e}"
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_question_by_id_cached(_fp: str, qid: int) -> Dict[str, Any]:
    eng = get_db_engine()
    if eng is None:
        return {}
    ensure_question_bank_table()
    with eng.connect() as conn:
        row = conn.execute(
            text("select * from public.question_bank_v1 where id = :id limit 1"),
            {"id": int(qid)}
        ).mappings().first()
    return dict(row) if row else {}


def load_question_by_id(qid: int) -> Dict[str, Any]:
    fp = (st.secrets.get("DATABASE_URL", "") or "")[:40]
    try:
        return load_question_by_id_cached(fp, int(qid))
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Question Error: {type(e).__name__}: {e}"
        return {}


def insert_question_bank_row(
    source: str,
    created_by: str,
    assignment_name: str,
    question_label: str,
    max_marks: int,
    tags: List[str],
    question_text: str = "",
    markscheme_text: str = "",
    question_image_path: Optional[str] = None,
    markscheme_image_path: Optional[str] = None,
) -> bool:
    eng = get_db_engine()
    if eng is None:
        return False
    ensure_question_bank_table()

    query = """
    insert into public.question_bank_v1
      (source, created_by, assignment_name, question_label, max_marks, tags,
       question_text, question_image_path,
       markscheme_text, markscheme_image_path,
       is_active, updated_at)
    values
      (:source, :created_by, :assignment_name, :question_label, :max_marks,
       CAST(:tags AS jsonb),
       :question_text, :question_image_path,
       :markscheme_text, :markscheme_image_path,
       true, now())
    on conflict (source, assignment_name, question_label) do update set
       created_by = excluded.created_by,
       max_marks = excluded.max_marks,
       tags = excluded.tags,
       question_text = excluded.question_text,
       question_image_path = excluded.question_image_path,
       markscheme_text = excluded.markscheme_text,
       markscheme_image_path = excluded.markscheme_image_path,
       is_active = true,
       updated_at = now()
    """
    try:
        with eng.begin() as conn:
            res = conn.execute(text(query), {
                "source": source,
                "created_by": (created_by or "").strip() or None,
                "assignment_name": assignment_name.strip(),
                "question_label": question_label.strip(),
                "max_marks": int(max_marks),
                "tags": json.dumps(tags or []),
                "question_text": (question_text or "").strip()[:12000] or None,
                "question_image_path": (question_image_path or "").strip() or None,
                "markscheme_text": (markscheme_text or "").strip()[:20000] or None,
                "markscheme_image_path": (markscheme_image_path or "").strip() or None,
            })
            if getattr(res, "rowcount", 1) == 0:
                raise RuntimeError("No row inserted/updated.")

        try:
            load_question_bank_df_cached.clear()
            load_question_by_id_cached.clear()
        except Exception:
            pass
        return True
    except Exception as e:
        st.session_state["db_last_error"] = f"Insert Question Bank Error: {type(e).__name__}: {e}"
        return False

# ============================================================
# MARKING (unified for question_bank_v1 rows)
# ============================================================
def _mk_system_schema(max_marks: int, question_text: str = "") -> str:
    qt = f"\nQuestion (student-facing):\n{question_text}\n" if question_text else "\n"
    return f"""
You are a strict GCSE Physics examiner.

CONFIDENTIALITY RULE (CRITICAL):
- The mark scheme is confidential. Do NOT reveal it, quote it, or paraphrase it.
- When producing the readback, ONLY describe what is in the student's work. Do not use the mark scheme.

OUTPUT RULE:
- Output ONLY valid JSON, nothing else.

Readback formatting:
- readback_markdown MUST be valid Markdown.
- Use LaTeX only inside $...$ or $$...$$.

Schema:
{{
  "readback_type": "<handwriting|diagram|mixed|unknown>",
  "readback_markdown": "<Markdown with LaTeX where helpful. Keep it concise but complete.>",
  "readback_warnings": ["<optional warning 1>", "<optional warning 2>"],
  "marks_awarded": <int>,
  "max_marks": <int>,
  "summary": "<1-2 sentences>",
  "feedback_points": ["<bullet 1>", "<bullet 2>"],
  "next_steps": ["<action 1>", "<action 2>"]
}}

{qt}
Max Marks: {int(max_marks)}
""".strip()


def _finalize_report(data: dict, max_marks: int) -> dict:
    readback_md = str(data.get("readback_markdown", "") or "").strip()
    readback_type = str(data.get("readback_type", "") or "").strip()
    readback_warn = data.get("readback_warnings", [])
    if not isinstance(readback_warn, list):
        readback_warn = []

    return {
        "readback_type": readback_type,
        "readback_markdown": readback_md,
        "readback_warnings": [str(x) for x in readback_warn][:6],
        "marks_awarded": clamp_int(data.get("marks_awarded", 0), 0, int(max_marks)),
        "max_marks": int(max_marks),
        "summary": str(data.get("summary", "")).strip(),
        "feedback_points": [str(x) for x in data.get("feedback_points", [])][:6],
        "next_steps": [str(x) for x in data.get("next_steps", [])][:6]
    }


def get_gpt_feedback_from_bank(
    student_answer,
    q_row: Dict[str, Any],
    is_student_image: bool,
    question_img: Optional[Image.Image],
    markscheme_img: Optional[Image.Image],
) -> dict:
    max_marks = int(q_row.get("max_marks", 1))
    question_text = (q_row.get("question_text") or "").strip()
    markscheme_text = (q_row.get("markscheme_text") or "").strip()

    system_instr = _mk_system_schema(max_marks=max_marks, question_text=question_text if question_text else "")
    messages = [{"role": "system", "content": system_instr}]

    if markscheme_text:
        messages.append({"role": "system", "content": f"CONFIDENTIAL MARKING SCHEME (DO NOT REVEAL):\n{markscheme_text}"})

    content = [{"type": "text", "text": "Mark this work. Return JSON only."}]

    if question_img is not None:
        content.append({"type": "text", "text": "Question image:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(question_img)}"}})

    if markscheme_img is not None:
        content.append({"type": "text", "text": "Mark scheme image (confidential):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(markscheme_img)}"}})

    if not is_student_image:
        if question_text and question_img is None:
            content.append({"type": "text", "text": f"Question text:\n{question_text}"})
        content.append({"type": "text", "text": f"Student Answer (text):\n{student_answer}\n(readback_markdown can be empty for typed answers)"})
    else:
        sa_b64 = encode_image(student_answer)
        content.append({"type": "text", "text": "Student answer (image):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sa_b64}"}})

    messages.append({"role": "user", "content": content})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_completion_tokens=2500,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content or ""
        if not raw.strip():
            raise ValueError("Empty response from AI.")

        data = safe_parse_json(raw)
        if not data:
            raise ValueError("No valid JSON parsed from response.")

        return _finalize_report(data, max_marks=max_marks)

    except Exception as e:
        return {
            "readback_type": "",
            "readback_markdown": "",
            "readback_warnings": [],
            "marks_awarded": 0,
            "max_marks": max_marks,
            "summary": "The examiner could not process this attempt (AI Error).",
            "feedback_points": ["Please try submitting again.", f"Error details: {str(e)[:120]}"],
            "next_steps": []
        }

# ============================================================
# AI QUESTION GENERATOR (teacher-only, vet, then save)
# ============================================================
GCSE_ONLY_GUARDRAILS = """
GCSE-ONLY CONTENT GUARDRAILS (CRITICAL):
- This app generates questions for AQA GCSE Physics (Higher). The question MUST be GCSE-level.
- Do NOT include A-level/IB/degree content, including (non-exhaustive):
  * permeability/permittivity constants: \\mu_0, \\epsilon_0
  * solenoid field equations like B = \\mu_0 n I
  * magnetic flux density calculations, magnetic flux, Faraday's law in equation form
  * calculus, differentiation/integration, vector cross products, field theory
  * "n (turns per metre)" style quantities, "flux linkage", "inductance"
- If the chosen topic is "Magnetism and electromagnets", focus on GCSE outcomes:
  * magnetic fields around magnets/wires, compasses, plotting fields
  * electromagnets (coil + iron core), factors affecting strength (current, turns, core)
  * uses and safety, qualitative reasoning, simple circuit context
  * avoid any magnetic field strength formula or B calculations.
""".strip()

MARKDOWN_LATEX_RULES = """
FORMATTING RULES:
- question_text MUST be valid Markdown and must render well in Streamlit's st.markdown.
- Use LaTeX only inside $...$ or $$...$$.
- Use (a), (b), (c) subparts as plain text, and put any equations in LaTeX.
- Use SI units and clear formatting.
""".strip()


def generate_practice_question_with_ai(
    topic_text: str,
    difficulty: str,
    qtype: str,
    marks: int,
    extra_instructions: str = "",
) -> Dict[str, Any]:
    def _extract_total_from_marksheme(ms: str) -> Optional[int]:
        m = re.search(r"\btotal\b\s*[:=]\s*(\d+)\b", ms or "", flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _has_part_marking(ms: str) -> bool:
        s = ms or ""
        return bool(re.search(r"(\([a-z]\)|\b[a-z]\))\s.*\[\s*\d+\s*\]", s, flags=re.IGNORECASE | re.DOTALL))

    def _forbidden_found(q: str, ms: str) -> List[str]:
        t = (q or "") + "\n" + (ms or "")
        bad = []
        patterns = [
            (r"\\mu_0|\bmu0\b|\bμ0\b", "Uses μ0 (not GCSE)"),
            (r"\\epsilon_0|\bepsilon0\b|\bε0\b", "Uses ε0 (not GCSE)"),
            (r"\bB\s*=\s*\\mu_0\s*n\s*I\b|\bB\s*=\s*μ0\s*n\s*I\b", "Uses solenoid field equation B=μ0 n I (not GCSE)"),
            (r"\bflux\b|\bflux linkage\b|\binductance\b", "Uses flux/inductance language (not GCSE here)"),
            (r"\bFaraday\b|\bLenz\b", "Uses Faraday/Lenz law (not GCSE equation form here)"),
            (r"\bcalculus\b|\bdifferentiat|\bintegrat", "Uses calculus (not GCSE)"),
        ]
        for pat, label in patterns:
            if re.search(pat, t, flags=re.IGNORECASE):
                bad.append(label)
        return bad

    def _validate(d: Dict[str, Any]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        qtxt = str(d.get("question_text", "") or "").strip()
        mstxt = str(d.get("markscheme_text", "") or "").strip()

        if not qtxt:
            reasons.append("Missing question_text.")
        if not mstxt:
            reasons.append("Missing markscheme_text.")

        mm = d.get("max_marks", None)
        try:
            mm_int = int(mm)
        except Exception:
            mm_int = None
        if mm_int != int(marks):
            reasons.append(f"max_marks must equal {int(marks)}.")

        total = _extract_total_from_marksheme(mstxt)
        if total != int(marks):
            reasons.append(f"Mark scheme TOTAL must equal {int(marks)}.")

        if not _has_part_marking(mstxt):
            reasons.append("Mark scheme must include part-by-part marks like '(a) ... [2]'.")

        bad = _forbidden_found(qtxt, mstxt)
        reasons.extend(bad)

        if "$" in qtxt and "\\(" in qtxt:
            reasons.append("Use $...$ for LaTeX, avoid \\(...\\).")

        return (len(reasons) == 0), reasons

    def _call_model(repair: bool, reasons: Optional[List[str]] = None) -> Dict[str, Any]:
        system = f"""
You are an expert AQA GCSE Physics (Higher) question writer and examiner.

Hard rules:
1) Create an ORIGINAL practice question. Do not reproduce copyrighted exam questions.
2) Must match AQA GCSE Physics Higher style and standard.
3) Stay strictly GCSE-level. Do not drift into A-level content.
4) Marks must be allocated per part, and the total must equal max_marks EXACTLY.
5) The mark scheme must include marking points for EVERY part.
6) Calculation marking must be GCSE-appropriate:
   - If a calculation requires more than one step, allocate at least 2 marks:
     one method mark (setup/substitution/rearrangement) and one accuracy mark (answer with unit).
7) Return ONLY valid JSON, nothing else.

{GCSE_ONLY_GUARDRAILS}

{MARKDOWN_LATEX_RULES}

Schema:
{{
  "question_text": "string",
  "markscheme_text": "string",
  "max_marks": integer,
  "tags": ["string", "string", ...]
}}

Mark scheme formatting requirements inside markscheme_text:
- Use a part-by-part breakdown with explicit mark allocation, for example:
  (a) ... [2]
  (b) ... [3]
- End with: TOTAL = <max_marks>
""".strip()

        base_user = f"""
Topic: {topic_text.strip()}
Difficulty: {difficulty}
Question type: {qtype}
max_marks: {int(marks)}

Additional teacher instructions (optional):
{extra_instructions.strip() if extra_instructions else "(none)"}

Constraints:
- Keep the question clearly GCSE. Avoid any forbidden content in the guardrails.
- Ensure the question is in Markdown and uses LaTeX only inside $...$ or $$...$$.
- End markscheme_text with EXACTLY: TOTAL = {int(marks)}.
""".strip()

        if not repair:
            user = base_user
        else:
            bullet_reasons = "\n".join([f"- {r}" for r in (reasons or [])]) or "- (unspecified)"
            user = f"""
You previously generated a draft that failed validation. Fix it and return corrected JSON only.

Validation failures:
{bullet_reasons}

You MUST:
- Keep topic, difficulty, type and max_marks unchanged.
- Remove any forbidden GCSE-only content (see guardrails).
- Ensure Markdown + LaTeX formatting rules are followed.
- Make the TOTAL line match max_marks exactly: TOTAL = {int(marks)}.
""".strip() + "\n\n" + base_user

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=2500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = safe_parse_json(raw) or {}
        return data

    data = _call_model(repair=False)
    ok, reasons = _validate(data)

    if not ok:
        data2 = _call_model(repair=True, reasons=reasons)
        ok2, reasons2 = _validate(data2)
        if ok2:
            data = data2
        else:
            data = data2 if isinstance(data2, dict) and data2 else data
            data["warnings"] = reasons2[:10]

    out = {
        "question_text": str(data.get("question_text", "") or "").strip(),
        "markscheme_text": str(data.get("markscheme_text", "") or "").strip(),
        "max_marks": int(marks),
        "tags": data.get("tags", []),
        "warnings": data.get("warnings", []),
    }
    if not isinstance(out["tags"], list):
        out["tags"] = []
    out["tags"] = [str(t).strip() for t in out["tags"] if str(t).strip()][:12]
    if not isinstance(out["warnings"], list):
        out["warnings"] = []
    out["warnings"] = [str(w) for w in out["warnings"]][:10]
    return out

# ============================================================
# REPORT RENDERER
# ============================================================
def render_report(report: dict):
    readback_md = (report.get("readback_markdown") or "").strip()
    if readback_md:
        st.markdown("**AI readback (what it thinks you wrote/drew):**")
        with st.container(border=True):
            st.markdown(readback_md)

        rb_warn = report.get("readback_warnings", [])
        if rb_warn:
            st.caption("Readback notes:")
            for w in rb_warn[:6]:
                st.write(f"- {w}")
        st.divider()

    st.markdown(f"**Marks:** {report.get('marks_awarded', 0)} / {report.get('max_marks', 0)}")
    if report.get("summary"):
        st.markdown(f"**Summary:** {report.get('summary')}")
    if report.get("feedback_points"):
        st.markdown("**Feedback:**")
        for p in report["feedback_points"]:
            st.write(f"- {p}")
    if report.get("next_steps"):
        st.markdown("**Next steps:**")
        for n in report["next_steps"]:
            st.write(f"- {n}")

# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.title("⚛️ PanPhy")
nav = st.sidebar.radio(
    "Navigate",
    ["🧑‍🎓 Student", "🔒 Teacher Dashboard", "📚 Question Bank"],
    index=0,
    key="nav_page",
)

header_left, header_mid, header_right = st.columns([3, 2, 1])
with header_left:
    st.title("⚛️ PanPhy Skill Builder")
    st.caption(f"Model: {MODEL_NAME}")
with header_right:
    issues = []
    if not AI_READY:
        issues.append("AI model not connected.")
    if not db_ready():
        issues.append("Database not connected.")
    if issues:
        st.caption("⚠️ System status")
        for msg in issues:
            st.caption(msg)

# ============================================================
# STUDENT PAGE
# ============================================================
if nav == "🧑‍🎓 Student":
    st.divider()

    source_options = ["AI Practice", "Teacher Uploads", "All"]
    expand_by_default = st.session_state.get("selected_qid") is None

    with st.expander("Question selection", expanded=expand_by_default):
        sel1, sel2 = st.columns([2, 2])
        with sel1:
            source = st.selectbox("Source", source_options, key="student_source")
        with sel2:
            st.text_input(
                "Student ID (optional)",
                placeholder="e.g. 10A_23",
                help="Leave blank to submit anonymously.",
                key="student_id",
            )

        if not db_ready():
            st.error("Database not ready. Configure DATABASE_URL first.")
        else:
            dfb = load_question_bank_df(limit=5000)
            if dfb.empty:
                st.info("No questions in the database yet. Ask your teacher to generate or upload questions in the Question Bank page.")
            else:
                if source == "AI Practice":
                    df_src = dfb[dfb["source"] == "ai_generated"].copy()
                elif source == "Teacher Uploads":
                    df_src = dfb[dfb["source"] == "teacher"].copy()
                else:
                    df_src = dfb.copy()

                if df_src.empty:
                    st.info("No questions available for this source yet.")
                else:
                    assignments = ["All"] + sorted(df_src["assignment_name"].dropna().unique().tolist())
                    if st.session_state.get("student_assignment_filter") not in assignments:
                        st.session_state["student_assignment_filter"] = "All"
                    assignment_filter = st.selectbox("Assignment", assignments, key="student_assignment_filter")

                    if assignment_filter != "All":
                        df2 = df_src[df_src["assignment_name"] == assignment_filter].copy()
                    else:
                        df2 = df_src.copy()

                    if df2.empty:
                        st.info("No questions available for this assignment.")
                    else:
                        df2 = df2.sort_values(["assignment_name", "question_label", "id"], kind="mergesort")
                        df2["label"] = df2.apply(
                            lambda r: f"{r['assignment_name']} | {r['question_label']} ({int(r['max_marks'])} marks) [id {int(r['id'])}]",
                            axis=1
                        )
                        choices = df2["label"].tolist()
                        labels_map = dict(zip(df2["label"], df2["id"]))

                        choice_key = f"student_question_choice::{source}::{assignment_filter}"
                        if st.session_state.get(choice_key) not in choices:
                            st.session_state[choice_key] = choices[0]

                        choice = st.selectbox("Question", choices, key=choice_key)
                        chosen_id = int(labels_map.get(choice, 0)) if choice else 0

                        if chosen_id:
                            if st.session_state.get("selected_qid") != chosen_id:
                                st.session_state["selected_qid"] = chosen_id

                                q_row = load_question_by_id(chosen_id)
                                st.session_state["cached_q_row"] = q_row

                                st.session_state["cached_q_path"] = (q_row.get("question_image_path") or "").strip()
                                st.session_state["cached_ms_path"] = (q_row.get("markscheme_image_path") or "").strip()

                                q_path = (st.session_state.get("cached_q_path") or "").strip()
                                if q_path:
                                    fp = (st.secrets.get("SUPABASE_URL", "") or "")[:40]
                                    q_bytes = cached_download_from_storage(q_path, fp)
                                    st.session_state["cached_question_img"] = safe_bytes_to_pil(q_bytes)
                                else:
                                    st.session_state["cached_question_img"] = None

                                st.session_state["feedback"] = None
                                st.session_state["canvas_key"] += 1
                                st.session_state["last_canvas_image_data"] = None

    if st.session_state.get("cached_q_row"):
        _qr = st.session_state["cached_q_row"]
        st.caption(f"Selected: {_qr.get('assignment_name', '')} | {_qr.get('question_label', '')}")

    student_id = st.session_state.get("student_id", "") or ""
    q_row: Dict[str, Any] = st.session_state.get("cached_q_row") or {}
    question_img = st.session_state.get("cached_question_img")
    max_marks = int(q_row.get("max_marks", 1)) if q_row else None
    q_text = (q_row.get("question_text") or "").strip() if q_row else ""
    q_key = None
    if q_row and q_row.get("id") is not None:
        try:
            q_key = f"QB:{int(q_row['id'])}:{q_row.get('assignment_name','')}:{q_row.get('question_label','')}"
        except Exception:
            q_key = None

    col1, col2 = st.columns([5, 4])

    with col1:
        st.subheader("📝 The Question")

        if not q_row:
            st.info("Select a question above to begin.")
        else:
            st.markdown("**Question**")
            with st.container(border=True):
                if question_img is not None:
                    st.image(question_img, caption="Question image", use_container_width=True)
                if q_text:
                    st.markdown(q_text)
                if (question_img is None) and (not q_text):
                    st.warning("This question has no question text or image.")
            st.caption(f"Max Marks: {max_marks}")

        st.write("")
        tab_type, tab_write = st.tabs(["⌨️ Type Answer", "✍️ Write Answer"])

        with tab_type:
            answer = st.text_area("Type your working:", height=200, placeholder="Enter your answer here...", key="student_answer_text")

            if st.button("Submit Text", type="primary", disabled=not AI_READY or not db_ready(), key="submit_text_btn"):
                sid = _effective_student_id(student_id)

                if not answer.strip():
                    st.toast("Please type an answer first.", icon="⚠️")
                elif not q_row:
                    st.error("Please select a question first.")
                else:
                    try:
                        allowed_now, _, reset_str = _check_rate_limit_db(sid)
                    except Exception:
                        allowed_now, reset_str = True, ""
                    if not allowed_now:
                        st.error(f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}.")
                    else:
                        increment_rate_limit(sid)

                        def task():
                            ms_path = (st.session_state.get("cached_ms_path") or q_row.get("markscheme_image_path") or "").strip()
                            ms_img = None
                            if ms_path:
                                fp = (st.secrets.get("SUPABASE_URL", "") or "")[:40]
                                ms_bytes = cached_download_from_storage(ms_path, fp)
                                ms_img = bytes_to_pil(ms_bytes) if ms_bytes else None
                            return get_gpt_feedback_from_bank(
                                student_answer=answer,
                                q_row=q_row,
                                is_student_image=False,
                                question_img=question_img,
                                markscheme_img=ms_img
                            )

                        st.session_state["feedback"] = _run_ai_with_progress(
                            task_fn=task,
                            ctx={"student_id": sid, "question": q_key or "", "mode": "text"},
                            typical_range="5-10 seconds",
                            est_seconds=9.0
                        )

                        if db_ready() and q_key:
                            insert_attempt(student_id, q_key, st.session_state["feedback"], mode="text")

        with tab_write:
            tool_row = st.columns([2, 1])
            with tool_row[0]:
                tool = st.radio("Tool", ["Pen", "Eraser"], horizontal=True, label_visibility="collapsed", key="canvas_tool")
            clear_clicked = tool_row[1].button("🗑️ Clear", use_container_width=True, key="canvas_clear")

            if clear_clicked:
                st.session_state["feedback"] = None
                st.session_state["last_canvas_image_data"] = None
                st.session_state["canvas_key"] += 1
                st.rerun()

            stroke_width = 2 if tool == "Pen" else 30
            stroke_color = "#000000" if tool == "Pen" else CANVAS_BG_HEX

            canvas_result = st_canvas(
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=CANVAS_BG_HEX,
                height=400,
                width=600,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state['canvas_key']}",
                display_toolbar=False,
                update_streamlit=True,
            )

            if canvas_result is not None and getattr(canvas_result, "image_data", None) is not None:
                if canvas_has_ink(canvas_result.image_data):
                    st.session_state["last_canvas_image_data"] = canvas_result.image_data

            submitted_writing = st.button(
                "Submit Writing",
                type="primary",
                disabled=not AI_READY or not db_ready(),
                key="submit_writing_btn",
            )

            if submitted_writing:
                sid = _effective_student_id(student_id)

                if not q_row:
                    st.error("Please select a question first.")
                else:
                    img_data = None
                    if canvas_result is not None and getattr(canvas_result, "image_data", None) is not None:
                        img_data = canvas_result.image_data
                    if img_data is None:
                        img_data = st.session_state.get("last_canvas_image_data")

                    if img_data is None or (not canvas_has_ink(img_data)):
                        st.toast("Canvas is blank. Write your answer first, then press Submit.", icon="⚠️")
                        st.stop()

                    try:
                        allowed_now, _, reset_str = _check_rate_limit_db(sid)
                    except Exception:
                        allowed_now, reset_str = True, ""
                    if not allowed_now:
                        st.error(f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}.")
                        st.stop()

                    img_for_ai = preprocess_canvas_image(img_data)

                    canvas_bytes = _encode_image_bytes(img_for_ai, "JPEG", quality=80)
                    ok_canvas, msg_canvas = validate_image_file(canvas_bytes, CANVAS_MAX_MB, "canvas")
                    if not ok_canvas:
                        okc, outb, _outct, err = _compress_bytes_to_limit(
                            canvas_bytes, CANVAS_MAX_MB, _purpose="canvas", prefer_fmt="JPEG"
                        )
                        if not okc:
                            st.error(err or msg_canvas)
                            st.stop()
                        img_for_ai = Image.open(io.BytesIO(outb)).convert("RGB")

                    increment_rate_limit(sid)

                    def task():
                        ms_path = (st.session_state.get("cached_ms_path") or q_row.get("markscheme_image_path") or "").strip()
                        ms_img = None
                        if ms_path:
                            fp = (st.secrets.get("SUPABASE_URL", "") or "")[:40]
                            ms_bytes = cached_download_from_storage(ms_path, fp)
                            ms_img = bytes_to_pil(ms_bytes) if ms_bytes else None
                        return get_gpt_feedback_from_bank(
                            student_answer=img_for_ai,
                            q_row=q_row,
                            is_student_image=True,
                            question_img=question_img,
                            markscheme_img=ms_img
                        )

                    st.session_state["feedback"] = _run_ai_with_progress(
                        task_fn=task,
                        ctx={"student_id": sid, "question": q_key or "", "mode": "writing"},
                        typical_range="8-15 seconds",
                        est_seconds=13.0
                    )

                    if db_ready() and q_key:
                        insert_attempt(student_id, q_key, st.session_state["feedback"], mode="writing")

    with col2:
        st.subheader("👨‍🏫 Report")
        with st.container(border=True):
            if st.session_state["feedback"]:
                render_report(st.session_state["feedback"])
                st.divider()
                if st.button("Start New Attempt", use_container_width=True, key="new_attempt"):
                    st.session_state["feedback"] = None
                    st.rerun()
            else:
                st.info("Submit an answer to receive feedback.")

# ============================================================
# TEACHER DASHBOARD PAGE
# ============================================================
elif nav == "🔒 Teacher Dashboard":
    st.divider()
    st.subheader("🔒 Teacher Dashboard")

    with st.expander("Database tools"):
        if st.button("Reconnect to database", key="reconnect_db_teacher"):
            _cached_engine.clear()
            try:
                load_attempts_df_cached.clear()
                load_question_bank_df_cached.clear()
                load_question_by_id_cached.clear()
            except Exception:
                pass
            st.session_state["db_table_ready"] = False
            st.session_state["bank_table_ready"] = False
            st.session_state["cached_q_row"] = None
            st.session_state["selected_qid"] = None
            st.rerun()
        if st.session_state.get("db_last_error"):
            st.write("Last DB error:")
            st.code(st.session_state["db_last_error"])

    if not (st.secrets.get("DATABASE_URL", "") or "").strip():
        st.info("Database not configured in secrets.")
    elif not db_ready():
        st.error("Database Connection Failed. Check drivers and URL.")
        if not get_db_driver_type():
            st.caption("No Postgres driver found. Add 'psycopg[binary]' (or psycopg2-binary) to requirements.txt")
        if st.session_state.get("db_last_error"):
            st.caption(st.session_state["db_last_error"])
    else:
        teacher_pw = st.text_input("Teacher password", type="password", key="pw_teacher_dash")
        if teacher_pw and teacher_pw == st.secrets.get("TEACHER_PASSWORD", ""):
            st.session_state["is_teacher"] = True
            ensure_attempts_table()

            df = load_attempts_df(limit=5000)

            if st.session_state.get("db_last_error"):
                st.error(f"Database Error: {st.session_state['db_last_error']}")

            if df.empty:
                st.info("No attempts logged yet.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total attempts", int(len(df)))
                c2.metric("Unique students", int(df["student_id"].nunique()))
                c3.metric("Topics attempted", int(df["question_key"].nunique()))

                st.write("### By student (overall %)")
                by_student = (
                    df.groupby("student_id")[["marks_awarded", "max_marks"]]
                    .sum()
                    .assign(percent=lambda x: (100 * x["marks_awarded"] / x["max_marks"].replace(0, np.nan)).round(1))
                    .sort_values("percent", ascending=False)
                )
                st.dataframe(by_student, use_container_width=True)

                st.write("### By topic (overall %)")
                by_topic = (
                    df.groupby("question_key")[["marks_awarded", "max_marks"]]
                    .sum()
                    .assign(percent=lambda x: (100 * x["marks_awarded"] / x["max_marks"].replace(0, np.nan)).round(1))
                    .sort_values("percent", ascending=False)
                )
                st.dataframe(by_topic, use_container_width=True)

                st.write("### Recent attempts")
                st.dataframe(df.head(50), use_container_width=True)
        else:
            st.caption("Enter the teacher password to view analytics.")

# ============================================================
# QUESTION BANK PAGE
# ============================================================
else:
    st.divider()
    st.subheader("📚 Question Bank")

    with st.expander("Database tools"):
        if st.button("Reconnect to database", key="reconnect_db_bank"):
            _cached_engine.clear()
            try:
                load_question_bank_df_cached.clear()
                load_question_by_id_cached.clear()
                cached_download_from_storage.clear()
            except Exception:
                pass
            st.session_state["db_table_ready"] = False
            st.session_state["bank_table_ready"] = False
            st.session_state["ai_draft"] = None
            st.rerun()
        if st.session_state.get("db_last_error"):
            st.write("Last DB error:")
            st.code(st.session_state["db_last_error"])

    if not db_ready():
        st.error("Database not ready. Configure DATABASE_URL first.")
    elif not supabase_ready():
        st.error("Supabase Storage not ready. Configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        st.caption("Also ensure the Python package 'supabase' is installed.")
    else:
        teacher_pw2 = st.text_input("Teacher password (to manage question bank)", type="password", key="pw_bank")
        if not (teacher_pw2 and teacher_pw2 == st.secrets.get("TEACHER_PASSWORD", "")):
            st.caption("Enter the teacher password to generate/upload/manage questions.")
        else:
            st.session_state["is_teacher"] = True
            ensure_question_bank_table()

            st.write("### Question Bank manager")
            st.caption("Use the tabs below to (1) browse and preview what is already in the bank, (2) generate AI practice questions, or (3) upload scanned questions. All features are unchanged, only reorganised.")

            tab_browse, tab_ai, tab_upload = st.tabs(["🔎 Browse & preview", "🤖 AI generator", "🖼️ Upload scans"])

            # -------------------------
            # Browse & preview
            # -------------------------
            with tab_browse:
                st.write("## 🔎 Browse & preview")
                df_all = load_question_bank_df(limit=5000, include_inactive=False)

                if df_all.empty:
                    st.info("No questions yet.")
                else:
                    df_all = df_all.copy()

                    sources = sorted([s for s in df_all["source"].dropna().unique().tolist() if str(s).strip()])
                    assignments = sorted([a for a in df_all["assignment_name"].dropna().unique().tolist() if str(a).strip()])

                    f1, f2, f3 = st.columns([2, 2, 2])
                    with f1:
                        src_sel = st.multiselect(
                            "Source",
                            options=sources,
                            default=sources,
                            key="bank_filter_source",
                        )
                    with f2:
                        asg_sel = st.selectbox(
                            "Assignment",
                            ["All"] + assignments,
                            index=0,
                            key="bank_filter_assignment",
                        )
                    with f3:
                        search_txt = st.text_input(
                            "Search (label, tag, keyword)",
                            value="",
                            placeholder="e.g. Q3b, circuit, energy",
                            key="bank_filter_search",
                        )

                    df_f = df_all
                    if src_sel:
                        df_f = df_f[df_f["source"].isin(src_sel)]
                    if asg_sel != "All":
                        df_f = df_f[df_f["assignment_name"] == asg_sel]
                    if (search_txt or "").strip():
                        s = search_txt.strip().lower()

                        def _row_match(r):
                            return (
                                s in str(r.get("assignment_name", "")).lower()
                                or s in str(r.get("question_label", "")).lower()
                                or s in str(r.get("question_text", "")).lower()
                                or s in str(r.get("tags", "")).lower()
                            )

                        df_f = df_f[df_f.apply(_row_match, axis=1)]

                    st.caption(f"Showing {len(df_f)} of {len(df_all)} questions.")

                    if df_f.empty:
                        st.warning("No questions match the current filters.")
                    else:
                        df_f = df_f.copy()

                        def _fmt_label(r):
                            asg = str(r.get("assignment_name") or "").strip()
                            ql = str(r.get("question_label") or "").strip()
                            src = str(r.get("source") or "").strip()
                            try:
                                mk = int(r.get("max_marks") or 0)
                            except Exception:
                                mk = 0
                            try:
                                qid = int(r.get("id"))
                            except Exception:
                                qid = -1
                            return f"{asg} | {ql} ({mk} marks) [{src}] [id {qid}]"

                        df_f["label"] = df_f.apply(_fmt_label, axis=1)
                        options = df_f["label"].tolist()

                        if "bank_preview_pick" in st.session_state and st.session_state["bank_preview_pick"] not in options:
                            st.session_state["bank_preview_pick"] = options[0]

                        pick = st.selectbox("Select a question to preview", options, key="bank_preview_pick")
                        pick_id = int(df_f.loc[df_f["label"] == pick, "id"].iloc[0])

                        row = load_question_by_id(pick_id) or {}
                        q_text = (row.get("question_text") or "").strip()
                        ms_text = (row.get("markscheme_text") or "").strip()

                        q_img = None
                        q_path = row.get("question_image_path")
                        if isinstance(q_path, str) and q_path.strip():
                            q_img = safe_bytes_to_pil(cached_download_from_storage(q_path))

                        ms_img = None
                        ms_path = row.get("markscheme_image_path")
                        if isinstance(ms_path, str) and ms_path.strip():
                            ms_img = safe_bytes_to_pil(cached_download_from_storage(ms_path))

                        meta1, meta2, meta3, meta4 = st.columns([3, 2, 2, 1])
                        with meta1:
                            st.caption(f"Assignment: {row.get('assignment_name', '')}")
                        with meta2:
                            st.caption(f"Label: {row.get('question_label', '')}")
                        with meta3:
                            st.caption(f"Source: {row.get('source', '')}")
                        with meta4:
                            st.caption(f"ID: {row.get('id', '')}")

                        pv1, pv2 = st.columns(2)

                        with pv1:
                            st.markdown("**Question (student view)**")
                            with st.container(border=True):
                                if q_img is not None:
                                    st.image(q_img, use_container_width=True)
                                if q_text:
                                    st.markdown(q_text)
                                if (q_img is None) and (not q_text):
                                    st.caption("No question text/image.")

                        with pv2:
                            st.markdown("**Mark scheme (teacher only)**")
                            with st.container(border=True):
                                if ms_img is not None:
                                    st.image(ms_img, use_container_width=True)
                                if ms_text:
                                    st.markdown(ms_text)
                                if (ms_img is None) and (not ms_text):
                                    st.caption("No mark scheme text/image (image-only teacher uploads are supported).")

                st.divider()
                st.write("### Recent question bank entries")
                df_bank = load_question_bank_df(limit=50, include_inactive=False)
                if not df_bank.empty:
                    st.dataframe(df_bank[["created_at", "source", "assignment_name", "question_label", "max_marks", "id"]], use_container_width=True)
                else:
                    st.info("No recent entries.")

            # -------------------------
            # AI generator
            # -------------------------
            with tab_ai:
                st.write("## 🤖 Generate practice question with AI (teacher vetting required)")

                gen_c1, gen_c2 = st.columns([2, 1])
                with gen_c1:
                    topic_mode = st.radio("Topic input", ["Choose from AQA list", "Describe a topic"], horizontal=True, key="topic_mode")
                    if topic_mode == "Choose from AQA list":
                        topic_choice = st.selectbox("AQA GCSE Physics Higher topic", AQA_GCSE_HIGHER_TOPICS, key="topic_choice")
                        topic_text = topic_choice
                    else:
                        topic_text = st.text_input("Describe the topic", placeholder="e.g. stopping distance with thinking vs braking distance", key="topic_text")

                    qtype = st.selectbox("Question type", QUESTION_TYPES, key="gen_qtype")
                    difficulty = st.selectbox("Difficulty", DIFFICULTIES, key="gen_difficulty")
                    marks_req = st.number_input("Max marks (target)", min_value=1, max_value=12, value=4, step=1, key="gen_marks")

                    extra_instr = st.text_area(
                        "Optional constraints for the AI",
                        height=80,
                        placeholder="e.g. Include one tricky unit conversion. Use g = 9.8 N/kg. Require a final answer with units.",
                        key="gen_extra"
                    )

                    assignment_name_ai = st.text_input("Assignment name for saving", value="AI Practice", key="gen_assignment")

                with gen_c2:
                    st.caption("Workflow: Generate draft → edit/vet → Approve & Save.")
                    gen_clicked = st.button("Generate draft", type="primary", use_container_width=True, disabled=not AI_READY, key="gen_btn")

                    if st.button("Clear draft", use_container_width=True, key="clear_draft"):
                        st.session_state["ai_draft"] = None
                        st.rerun()

                if gen_clicked:
                    if not (topic_text or "").strip():
                        st.warning("Please choose or describe a topic first.")
                    else:
                        def task_generate():
                            return generate_practice_question_with_ai(
                                topic_text=topic_text.strip(),
                                difficulty=difficulty,
                                qtype=qtype,
                                marks=int(marks_req),
                                extra_instructions=extra_instr or "",
                            )

                        draft_raw = _run_ai_with_progress(
                            task_fn=task_generate,
                            ctx={"student_id": "teacher", "question": "AI_GENERATOR", "mode": "generate"},
                            typical_range="5-12 seconds",
                            est_seconds=10.0
                        )

                        qtxt = str(draft_raw.get("question_text", "") or "").strip()
                        mstxt = str(draft_raw.get("markscheme_text", "") or "").strip()
                        mm = clamp_int(draft_raw.get("max_marks", int(marks_req)), 1, 50, default=int(marks_req))
                        tags = draft_raw.get("tags", [])
                        warnings = draft_raw.get("warnings", [])
                        if not isinstance(tags, list):
                            tags = []
                        if not isinstance(warnings, list):
                            warnings = []

                        if not qtxt or not mstxt:
                            st.error("AI did not return a valid draft. Please try again.")
                        else:
                            token = pysecrets.token_hex(3)
                            default_label = f"AI-{slugify(topic_text)[:24]}-{token}"

                            st.session_state["ai_draft"] = {
                                "assignment_name": (assignment_name_ai or "").strip() or "AI Practice",
                                "question_label": default_label,
                                "max_marks": int(mm),
                                "tags": [str(t).strip() for t in tags if str(t).strip()][:10],
                                "question_text": qtxt,
                                "markscheme_text": mstxt,
                                "warnings": warnings[:10],
                            }
                            st.success("Draft generated. Please vet and edit below, then approve to save.")

                if st.session_state.get("ai_draft"):
                    d = st.session_state["ai_draft"]

                    if d.get("warnings"):
                        st.warning("AI draft warnings (auto-check):\n\n" + "\n".join([f"- {w}" for w in d["warnings"]]))

                    st.write("### ✅ Vet and edit the draft (Markdown + LaTeX supported)")
                    ed1, ed2 = st.columns([2, 1])
                    with ed1:
                        d_assignment = st.text_input("Assignment name", value=d.get("assignment_name", "AI Practice"), key="draft_assignment")
                        d_label = st.text_input("Question label", value=d.get("question_label", ""), key="draft_label")
                        d_marks = st.number_input("Max marks", min_value=1, max_value=50, value=int(d.get("max_marks", 4)), step=1, key="draft_marks")
                        d_tags_str = st.text_input("Tags (comma separated)", value=", ".join(d.get("tags", [])), key="draft_tags")

                    with ed2:
                        st.caption("Mark scheme is confidential. Students never see it.")
                        approve_clicked = st.button("Approve & Save to bank", type="primary", use_container_width=True, key="approve_save")
                        st.caption("Tip: use Markdown and LaTeX ($...$) freely.")

                    d_qtext = st.text_area("Question text (student will see this)", value=d.get("question_text", ""), height=180, key="draft_qtext")
                    d_mstext = st.text_area("Mark scheme (teacher-only)", value=d.get("markscheme_text", ""), height=220, key="draft_mstext")

                    p1, p2 = st.columns(2)
                    with p1:
                        render_md_box("Preview: Question (student view)", d_qtext, empty_text="No question text.")
                    with p2:
                        render_md_box("Preview: Mark scheme (teacher only)", d_mstext, empty_text="No mark scheme.")

                    if approve_clicked:
                        if not d_assignment.strip() or not d_label.strip():
                            st.error("Assignment name and Question label cannot be blank.")
                        elif not d_qtext.strip() or not d_mstext.strip():
                            st.error("Question text and mark scheme cannot be blank.")
                        else:
                            combined = d_qtext + "\n" + d_mstext
                            if re.search(r"\\mu_0|\bμ0\b|\\epsilon_0|\bε0\b|B\s*=\s*\\mu_0\s*n\s*I", combined, flags=re.IGNORECASE):
                                st.error("This draft contains non-GCSE content (e.g. μ0/ε0 or B=μ0 n I). Please edit it out before saving.")
                                st.stop()

                            tags = [t.strip() for t in (d_tags_str or "").split(",") if t.strip()]
                            ok = insert_question_bank_row(
                                source="ai_generated",
                                created_by="teacher",
                                assignment_name=d_assignment.strip(),
                                question_label=d_label.strip(),
                                max_marks=int(d_marks),
                                tags=tags,
                                question_text=d_qtext.strip(),
                                markscheme_text=d_mstext.strip(),
                                question_image_path=None,
                                markscheme_image_path=None,
                            )
                            if ok:
                                st.session_state["ai_draft"] = None
                                st.success("Approved and saved. Students can now access this under AI Practice.")
                            else:
                                st.error("Failed to save to database. Check errors below.")

                st.divider()

            # -------------------------
            # Upload scans
            # -------------------------
            with tab_upload:
                st.write("## 🖼️ Upload a teacher question (images)")
                st.caption("Optional question text supports Markdown and LaTeX ($...$).")

                with st.form("upload_q_form", clear_on_submit=True):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        assignment_name = st.text_input("Assignment name", placeholder="e.g. AQA Paper 1 (Electricity)", key="up_assignment")
                        question_label = st.text_input("Question label", placeholder="e.g. Q3b", key="up_label")
                    with c2:
                        max_marks_in = st.number_input("Max marks", min_value=1, max_value=50, value=3, step=1, key="up_marks")

                    tags_str = st.text_input("Tags (comma separated)", placeholder="forces, resultant, newton", key="up_tags")
                    q_text_opt = st.text_area("Optional: question text (Markdown + LaTeX supported)", height=100, key="up_qtext")

                    q_file = st.file_uploader("Upload question screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"], key="up_qfile")
                    ms_file = st.file_uploader("Upload mark scheme screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"], key="up_msfile")

                    submitted = st.form_submit_button("Save to Question Bank", type="primary")

                if q_text_opt and q_text_opt.strip():
                    render_md_box("Preview: Optional question text", q_text_opt)

                if submitted:
                    if not assignment_name.strip() or not question_label.strip():
                        st.warning("Please fill in Assignment name and Question label.")
                    elif q_file is None or ms_file is None:
                        st.warning("Please upload both the question screenshot and the mark scheme screenshot.")
                    else:
                        assignment_slug = slugify(assignment_name)
                        qlabel_slug = slugify(question_label)
                        token = pysecrets.token_hex(6)

                        q_bytes_raw = q_file.getvalue()
                        ms_bytes_raw = ms_file.getvalue()

                        ok_q, msg_q = validate_image_file(q_bytes_raw, QUESTION_MAX_MB, "question image")
                        ok_ms, msg_ms = validate_image_file(ms_bytes_raw, MARKSCHEME_MAX_MB, "mark scheme image")

                        if not ok_q:
                            okc, q_bytes, q_ct, err = _compress_bytes_to_limit(q_bytes_raw, QUESTION_MAX_MB, _purpose="question image")
                            if not okc:
                                st.error(err or msg_q)
                                st.stop()
                        else:
                            q_bytes = q_bytes_raw
                            q_ct = "image/png" if (q_file.name or "").lower().endswith(".png") else "image/jpeg"

                        if not ok_ms:
                            okc, ms_bytes, ms_ct, err = _compress_bytes_to_limit(ms_bytes_raw, MARKSCHEME_MAX_MB, _purpose="mark scheme image")
                            if not okc:
                                st.error(err or msg_ms)
                                st.stop()
                        else:
                            ms_bytes = ms_bytes_raw
                            ms_ct = "image/png" if (ms_file.name or "").lower().endswith(".png") else "image/jpeg"

                        q_ext = ".jpg" if q_ct == "image/jpeg" else ".png"
                        ms_ext = ".jpg" if ms_ct == "image/jpeg" else ".png"

                        q_path = f"{assignment_slug}/{token}/{qlabel_slug}_question{q_ext}"
                        ms_path = f"{assignment_slug}/{token}/{qlabel_slug}_markscheme{ms_ext}"

                        ok1 = upload_to_storage(q_path, q_bytes, q_ct)
                        ok2 = upload_to_storage(ms_path, ms_bytes, ms_ct)

                        tags = [t.strip() for t in (tags_str or "").split(",") if t.strip()]

                        if ok1 and ok2:
                            ok_db = insert_question_bank_row(
                                source="teacher",
                                created_by="teacher",
                                assignment_name=assignment_name.strip(),
                                question_label=question_label.strip(),
                                max_marks=int(max_marks_in),
                                tags=tags,
                                question_text=(q_text_opt or "").strip(),
                                markscheme_text="",
                                question_image_path=q_path,
                                markscheme_image_path=ms_path
                            )
                            if ok_db:
                                st.success("Saved. This question is now available in the Student page.")
                            else:
                                st.error("Uploaded images, but failed to save metadata to DB. Check errors below.")
                        else:
                            st.error("Failed to upload one or both images to Supabase Storage. Check errors below.")

            if st.session_state.get("db_last_error"):
                st.error(f"Error: {st.session_state['db_last_error']}")
