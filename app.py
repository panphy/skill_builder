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
import textwrap

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

# Display/UX tuning
RATE_LIMIT_DISPLAY_TTL_SECONDS = 8  # avoid DB hit on every widget rerun

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

QUESTION_BANK_DDL = """
create table if not exists public.question_bank_v1 (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  -- provenance
  source text not null check (source in ('teacher','ai_generated')),
  created_by text,

  -- organization
  assignment_name text not null,
  question_label text not null,
  max_marks int not null check (max_marks > 0),
  tags jsonb,

  -- question content
  question_text text,
  question_image_path text,

  -- mark scheme content (confidential, never shown to students)
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

create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_question_bank_updated_at on public.question_bank_v1;
create trigger trg_question_bank_updated_at
before update on public.question_bank_v1
for each row execute function public.set_updated_at();
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
    """
    Uses Supabase Python client for Storage.
    Recommended: use SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets (server-side only).
    """
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
def _ss_init(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


_ss_init("canvas_key", 0)
_ss_init("feedback", None)
_ss_init("anon_id", pysecrets.token_hex(4))
_ss_init("db_last_error", "")
_ss_init("db_table_ready", False)
_ss_init("bank_table_ready", False)
_ss_init("rate_table_ready", False)
_ss_init("is_teacher", False)

# Cache versioning to cheaply bust @st.cache_data reads after writes
_ss_init("attempts_version", 0)
_ss_init("bank_version", 0)

# Canvas robustness cache
_ss_init("last_canvas_image_data", None)

# Question selection cache
_ss_init("selected_qid", None)
_ss_init("cached_q_row", None)
_ss_init("cached_question_img", None)
_ss_init("cached_q_path", None)
_ss_init("cached_ms_path", None)

# Question list cache
_ss_init("cached_bank_df", None)
_ss_init("cached_assignments", ["All"])
_ss_init("cached_labels_map", {})
_ss_init("cached_labels", [])
_ss_init("cached_labels_map_key", None)

# AI generator draft cache (teacher-only)
_ss_init("ai_draft", None)

# Rate limit display cache (to stop DB hits on every widget change)
_ss_init("rate_limit_display_cache", {"sid": None, "ts": 0.0, "remaining": RATE_LIMIT_MAX, "reset": ""})

# ============================================================
#  ROBUST DATABASE LAYER
# ============================================================
def get_db_driver_type():
    try:
        import psycopg  # noqa
        return "psycopg"
    except ImportError:
        try:
            import psycopg2  # noqa
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


def _exec_sql_many(conn, sql_blob: str):
    parts = [p.strip() for p in (sql_blob or "").split(";")]
    for p in parts:
        if p:
            conn.execute(text(p))


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
    if st.session_state.get("rate_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return
    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, RATE_LIMITS_DDL)
        st.session_state["rate_table_ready"] = True
        LOGGER.info("Rate limits table ready", extra={"ctx": {"component": "db", "table": "rate_limits"}})
    except Exception as e:
        st.session_state["db_last_error"] = f"Rate Limits Table Error: {type(e).__name__}: {e}"
        st.session_state["rate_table_ready"] = False
        LOGGER.error("Rate limits table ensure failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})


def _fix_question_bank_source_constraint(conn):
    try:
        rows = conn.execute(text("""
            select conname, pg_get_constraintdef(c.oid) as condef
            from pg_constraint c
            where c.conrelid = 'public.question_bank_v1'::regclass
              and c.contype = 'c'
        """)).mappings().all()
    except Exception:
        return

    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", (s or "").lower())

    desired_bits = ["source", "teacher", "ai_generated"]

    for r in rows:
        name = r.get("conname", "")
        condef = r.get("condef", "")
        n = _norm(condef)
        if "source" in n:
            ok = all(bit in n for bit in desired_bits)
            if not ok:
                try:
                    conn.execute(text(f'alter table public.question_bank_v1 drop constraint if exists "{name}"'))
                except Exception:
                    pass

    try:
        conn.execute(text("""
            alter table public.question_bank_v1
            add constraint question_bank_v1_source_check
            check (source in ('teacher','ai_generated'))
        """))
    except Exception:
        pass


def ensure_question_bank_table():
    if st.session_state.get("bank_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return
    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, QUESTION_BANK_DDL)
            _fix_question_bank_source_constraint(conn)

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


def check_rate_limit(student_id: str, lock: bool = False) -> Tuple[bool, int, str]:
    """
    If lock=False: best-effort display query (no row locks, no writes).
    If lock=True: enforcement query (locks row and performs reset writes as needed).
    """
    if st.session_state.get("is_teacher", False):
        return True, RATE_LIMIT_MAX, ""

    eng = get_db_engine()
    if eng is None:
        return True, RATE_LIMIT_MAX, ""

    ensure_rate_limits_table()

    sid = (student_id or "").strip() or f"anon_{st.session_state['anon_id']}"
    now_utc = datetime.now(timezone.utc)

    try:
        with eng.begin() as conn:
            if lock:
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

            else:
                row = conn.execute(
                    text("""
                        select submission_count, window_start_time
                        from public.rate_limits
                        where student_id = :sid
                        limit 1
                    """),
                    {"sid": sid},
                ).mappings().first()

                if not row:
                    submission_count = 0
                    window_start = now_utc
                else:
                    submission_count = int(row.get("submission_count") or 0)
                    window_start = row.get("window_start_time")
                    if isinstance(window_start, datetime):
                        if window_start.tzinfo is None:
                            window_start = window_start.replace(tzinfo=timezone.utc)
                        else:
                            window_start = window_start.astimezone(timezone.utc)
                    else:
                        window_start = now_utc

                elapsed = (now_utc - window_start).total_seconds()
                if elapsed >= RATE_LIMIT_WINDOW_SECONDS:
                    # Treat as reset for display purposes; actual reset is done on enforcement/increment.
                    submission_count = 0
                    window_start = now_utc

            remaining = max(0, RATE_LIMIT_MAX - submission_count)
            reset_time_utc = window_start + timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)
            allowed = submission_count < RATE_LIMIT_MAX
            reset_str = _format_reset_time(reset_time_utc)
            return allowed, remaining, reset_str

    except Exception:
        return True, RATE_LIMIT_MAX, ""


def get_rate_limit_display(student_id: str) -> Tuple[int, str]:
    """
    Cache remaining/reset for a few seconds to prevent DB hits on every widget rerun.
    """
    if st.session_state.get("is_teacher", False):
        return RATE_LIMIT_MAX, ""

    sid = _effective_student_id(student_id)
    cache = st.session_state.get("rate_limit_display_cache", {})
    now = time.monotonic()

    if cache.get("sid") == sid and (now - float(cache.get("ts", 0.0))) < RATE_LIMIT_DISPLAY_TTL_SECONDS:
        return int(cache.get("remaining", RATE_LIMIT_MAX)), str(cache.get("reset", "") or "")

    _, remaining, reset_str = check_rate_limit(sid, lock=False)
    st.session_state["rate_limit_display_cache"] = {"sid": sid, "ts": now, "remaining": int(remaining), "reset": reset_str}
    return int(remaining), str(reset_str or "")


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

# ============================================================
# STORAGE HELPERS
# ============================================================
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "untitled"


def upload_to_storage(path: str, file_bytes: bytes, content_type: str) -> bool:
    sb = get_supabase_client()
    if sb is None:
        st.session_state["db_last_error"] = "Supabase Storage not configured."
        return False
    try:
        res = sb.storage.from_(STORAGE_BUCKET).upload(
            path,
            file_bytes,
            {"content-type": content_type, "upsert": "true"}
        )
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
        return False


def download_from_storage(path: str) -> bytes:
    sb = get_supabase_client()
    if sb is None:
        return b""
    try:
        res = sb.storage.from_(STORAGE_BUCKET).download(path)
        if isinstance(res, (bytes, bytearray)):
            return bytes(res)
        if hasattr(res, "data") and res.data is not None:
            if isinstance(res.data, (bytes, bytearray)):
                return bytes(res.data)
        return b""
    except Exception as e:
        st.session_state["db_last_error"] = f"Storage Download Error: {type(e).__name__}: {e}"
        return b""


@st.cache_data(ttl=300, show_spinner=False)
def cached_download_from_storage(path: str, bank_version: int) -> bytes:
    # bank_version busts after uploads/saves that change images/paths
    if not path:
        return b""
    return download_from_storage(path)


def bytes_to_pil(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img

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
    if image_data is None:
        return False
    arr = image_data.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] if arr.shape[2] >= 4 else np.full((arr.shape[0], arr.shape[1]), 255, dtype=np.uint8)
    bg = np.array(CANVAS_BG_RGB, dtype=np.uint8)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    ink = (diff > 60) & (alpha > 30)
    return (ink.mean() > 0.001)


def preprocess_canvas_image(image_data: np.ndarray) -> Image.Image:
    raw_img = Image.fromarray(image_data.astype("uint8"))
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
# MARKDOWN + LaTeX RENDER HELPERS (end-to-end consistency)
# ============================================================
_LATEX_INLINE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_LATEX_BLOCK = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


def normalize_markdown(md_text: str) -> str:
    """
    Make Markdown + $...$ / $$...$$ render more reliably in st.markdown.
    - Normalize newlines
    - Dedent accidental indentation (prevents unwanted code blocks)
    - Convert \\( ... \\) -> $ ... $ and \\[ ... \\] -> $$ ... $$
    """
    txt = (md_text or "")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    # Remove BOM / zero-width oddities
    txt = txt.replace("\ufeff", "")

    # Avoid accidental code blocks from global indentation
    txt = textwrap.dedent(txt)

    # Convert alternate LaTeX delimiters to $...$ / $$...$$
    txt = _LATEX_BLOCK.sub(lambda m: "$$\n" + m.group(1).strip() + "\n$$", txt)
    txt = _LATEX_INLINE.sub(lambda m: "$" + m.group(1).strip() + "$", txt)

    # Trim trailing whitespace on lines
    txt = "\n".join([ln.rstrip() for ln in txt.split("\n")]).strip()
    return txt


def _find_suspicious_latex_outside_math(md_text: str) -> List[str]:
    """
    Best-effort warning detector: backslash LaTeX commands outside $...$ blocks often appear as 'weird escapes'.
    We do not auto-rewrite aggressively (risk of altering meaning), but we can warn teachers.
    """
    s = md_text or ""
    if not s.strip():
        return []

    # Remove code blocks first
    s2 = re.sub(r"```.*?```", "", s, flags=re.DOTALL)

    # Mask math blocks ($$...$$ and $...$) so we only scan outside math
    s2 = re.sub(r"\$\$.*?\$\$", " ", s2, flags=re.DOTALL)
    s2 = re.sub(r"\$(?:\\\$|[^\$])*\$", " ", s2, flags=re.DOTALL)

    hits = []
    # Common LaTeX commands that users might type outside math
    for pat, label in [
        (r"\\times\b", r"\\times outside $...$"),
        (r"\\frac\b", r"\\frac outside $...$"),
        (r"\\mu\b|\\epsilon\b|\\rho\b|\\lambda\b|\\theta\b", r"Greek command outside $...$"),
        (r"\\sqrt\b", r"\\sqrt outside $...$"),
        (r"\\mathrm\b|\\text\b", r"\\mathrm/\\text outside $...$"),
    ]:
        if re.search(pat, s2):
            hits.append(label)
    return hits[:6]


def render_markdown(md_text: str):
    st.markdown(normalize_markdown(md_text), unsafe_allow_html=False)


def render_md_box(title: str, md_text: str, caption: str = "", empty_text: str = ""):
    st.markdown(f"**{title}**")
    with st.container(border=True):
        txt = (md_text or "").strip()
        if txt:
            render_markdown(txt)
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
# DB OPERATIONS (attempts + question bank) + caching wrappers
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
        st.session_state["attempts_version"] = int(st.session_state.get("attempts_version", 0)) + 1
    except Exception as e:
        st.session_state["db_last_error"] = f"Insert Error: {type(e).__name__}: {e}"


def _load_attempts_df_uncached(limit: int = 5000) -> pd.DataFrame:
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_attempts_table()
    try:
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
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Error: {type(e).__name__}: {e}"
        return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def load_attempts_df(limit: int, attempts_version: int) -> pd.DataFrame:
    return _load_attempts_df_uncached(limit=limit)


def _load_question_bank_df_uncached(limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_question_bank_table()

    where = "" if include_inactive else "where is_active = true"
    try:
        with eng.connect() as conn:
            df = pd.read_sql(
                text(f"""
                    select id, created_at, source, assignment_name, question_label, max_marks, is_active
                    from public.question_bank_v1
                    {where}
                    order by created_at desc
                    limit :limit
                """),
                conn,
                params={"limit": int(limit)},
            )
        return df
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Question Bank Error: {type(e).__name__}: {e}"
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_question_bank_df(limit: int, include_inactive: bool, bank_version: int) -> pd.DataFrame:
    return _load_question_bank_df_uncached(limit=limit, include_inactive=include_inactive)


def _load_question_by_id_uncached(qid: int) -> Dict[str, Any]:
    eng = get_db_engine()
    if eng is None:
        return {}
    ensure_question_bank_table()
    try:
        with eng.connect() as conn:
            row = conn.execute(
                text("select * from public.question_bank_v1 where id = :id limit 1"),
                {"id": int(qid)}
            ).mappings().first()
        return dict(row) if row else {}
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Question Error: {type(e).__name__}: {e}"
        return {}


@st.cache_data(ttl=120, show_spinner=False)
def load_question_by_id(qid: int, bank_version: int) -> Dict[str, Any]:
    return _load_question_by_id_uncached(qid=qid)


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
       is_active)
    values
      (:source, :created_by, :assignment_name, :question_label, :max_marks,
       CAST(:tags AS jsonb),
       :question_text, :question_image_path,
       :markscheme_text, :markscheme_image_path,
       true)
    on conflict (source, assignment_name, question_label) do nothing
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(query), {
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
        st.session_state["bank_version"] = int(st.session_state.get("bank_version", 0)) + 1
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
        "readback_markdown": normalize_markdown(readback_md) if readback_md else "",
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
        reasons = []
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

        # Soft warning: suspicious LaTeX outside math
        susp = _find_suspicious_latex_outside_math(qtxt) + _find_suspicious_latex_outside_math(mstxt)
        if susp:
            reasons.append("Formatting: " + "; ".join(susp))

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
        "question_text": normalize_markdown(str(data.get("question_text", "") or "").strip()),
        "markscheme_text": normalize_markdown(str(data.get("