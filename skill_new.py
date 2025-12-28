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

# NEW: logging + utilities
import logging
import sys
import os
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# ============================================================
# 1) DDL for rate limiting table (PostgreSQL)
# ============================================================
RATE_LIMITS_DDL = """
create table if not exists public.rate_limits (
  student_id text primary key,
  submission_count int not null,
  window_start_time timestamptz not null
);

create index if not exists idx_rate_limits_window_start
  on public.rate_limits (window_start_time);
""".strip()

# ============================================================
# 4) LOGGING FOR DEBUGGING (configure at startup)
# ============================================================
def setup_logging() -> logging.Logger:
    """
    Logs to:
      - Console (Streamlit Cloud log output)
      - Rotating file (useful for local dev)
    Uses structured-ish messages with [key=value] tags + extra fields.
    """
    logger = logging.getLogger("panphy")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s %(levelname)s %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (best-effort; Streamlit Cloud may be read-only in some envs)
    try:
        os.makedirs("logs", exist_ok=True)
        fh = RotatingFileHandler("logs/app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        # Do not crash the app if file logging is unavailable
        pass

    logger.propagate = False
    return logger


LOGGER = setup_logging()


def _kv(**kwargs) -> str:
    """Helper to format key-value tags like [student_id=10A_23] consistently."""
    parts = []
    for k, v in kwargs.items():
        if v is None or v == "":
            continue
        safe = str(v).replace("\n", " ").replace("\r", " ")
        parts.append(f"[{k}={safe}]")
    return " ".join(parts)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
CUSTOM_QUESTION_PREFIX = "CUSTOM"

# NEW: rate limit constants
RATE_LIMIT_MAX_PER_HOUR = 10
RATE_LIMIT_WINDOW_SECONDS = 3600

# NEW: upload limits
MAX_DIM_PX = 4000
MAX_Q_IMG_MB = 5.0
MAX_MS_IMG_MB = 5.0
MAX_CANVAS_MB = 2.0

# =========================
# --- OPENAI CLIENT (CACHED) ---
# =========================
@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


try:
    client = get_client()
    AI_READY = True
except Exception:
    st.error("⚠️ OpenAI API Key missing or invalid in Streamlit Secrets!")
    AI_READY = False

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
    except Exception:
        return None


def supabase_ready() -> bool:
    return get_supabase_client() is not None

# =========================
# --- SESSION STATE ---
# =========================
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state:
    st.session_state["feedback"] = None
if "anon_id" not in st.session_state:
    st.session_state["anon_id"] = pysecrets.token_hex(4)
if "db_last_error" not in st.session_state:
    st.session_state["db_last_error"] = ""
if "db_table_ready" not in st.session_state:
    st.session_state["db_table_ready"] = False
if "custom_table_ready" not in st.session_state:
    st.session_state["custom_table_ready"] = False
if "rate_table_ready" not in st.session_state:
    st.session_state["rate_table_ready"] = False

# Cache teacher-upload selection/images so no repeated download while writing
if "selected_custom_id" not in st.session_state:
    st.session_state["selected_custom_id"] = None
if "cached_custom_row" not in st.session_state:
    st.session_state["cached_custom_row"] = None
if "cached_question_img" not in st.session_state:
    st.session_state["cached_question_img"] = None
if "cached_q_path" not in st.session_state:
    st.session_state["cached_q_path"] = None
if "cached_ms_path" not in st.session_state:
    st.session_state["cached_ms_path"] = None

# Cache custom question list to avoid DB query work during canvas usage
if "cached_dfq" not in st.session_state:
    st.session_state["cached_dfq"] = None
if "cached_assignments" not in st.session_state:
    st.session_state["cached_assignments"] = ["All"]
if "cached_labels_map" not in st.session_state:
    st.session_state["cached_labels_map"] = {}
if "cached_labels" not in st.session_state:
    st.session_state["cached_labels"] = []
if "cached_labels_map_key" not in st.session_state:
    st.session_state["cached_labels_map_key"] = None

# =========================
# --- QUESTION BANK (BUILT-IN) ---
# =========================
QUESTIONS = {
    "Q1: Forces (Resultant)": {
        "question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate the acceleration.",
        "marks": 3,
        "mark_scheme": "1. Resultant force = 20 - 4 = 16N (1 mark). 2. F = ma (1 mark). 3. a = 16 / 5 = 3.2 m/s² (1 mark).",
    },
    "Q2: Refraction (Drawing)": {
        "question": "Draw a ray diagram showing light passing from air into a glass block at an angle.",
        "marks": 2,
        "mark_scheme": "1. Ray bends towards the normal inside the glass. 2. Angles of incidence and refraction labeled correctly.",
    },
}

# =========================
#  ROBUST DATABASE LAYER
# =========================
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
    # Cache only successful engine creation by keying on the URL.
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
        LOGGER.error(f"{_kv(area='db')} DB engine creation failed: {type(e).__name__}: {e}")
        return None


def db_ready() -> bool:
    return get_db_engine() is not None


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

    # Safe migration: add columns if missing (no data loss)
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
            conn.execute(text(ddl_create))
            conn.execute(text(ddl_alter))
        st.session_state["db_last_error"] = ""
        st.session_state["db_table_ready"] = True
        LOGGER.info(f"{_kv(area='db')} attempts table ready")
    except Exception as e:
        st.session_state["db_last_error"] = f"Table Creation Error: {type(e).__name__}: {e}"
        st.session_state["db_table_ready"] = False
        LOGGER.error(f"{_kv(area='db')} attempts table creation failed: {type(e).__name__}: {e}")


# =========================
# 1) RATE LIMITING (DB TABLE + HELPERS)
# =========================
def ensure_rate_limits_table():
    if st.session_state.get("rate_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return
    try:
        with eng.begin() as conn:
            conn.execute(text(RATE_LIMITS_DDL))
        st.session_state["rate_table_ready"] = True
        LOGGER.info(f"{_kv(area='db')} rate_limits table ready")
    except Exception as e:
        st.session_state["db_last_error"] = f"Rate Table Creation Error: {type(e).__name__}: {e}"
        st.session_state["rate_table_ready"] = False
        LOGGER.error(f"{_kv(area='db')} rate_limits table creation failed: {type(e).__name__}: {e}")


def _effective_student_id(student_id: str) -> str:
    sid = (student_id or "").strip()
    if not sid:
        sid = f"anon_{st.session_state['anon_id']}"
    return sid


def check_rate_limit(student_id: str) -> tuple[bool, int, str]:
    """
    Returns (allowed, remaining, reset_time_str).
    - Resets the counter if window expired.
    - Does NOT increment. Call increment_rate_limit() after passing check.
    """
    eng = get_db_engine()
    if eng is None:
        # If DB is down, we choose to allow (avoid blocking learning),
        # but you can flip this to deny if cost control is more important.
        return True, RATE_LIMIT_MAX_PER_HOUR, ""

    ensure_rate_limits_table()

    sid = _effective_student_id(student_id)
    now = _utc_now()
    window = timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)

    try:
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
                # Initialize window
                conn.execute(
                    text("""
                        insert into public.rate_limits (student_id, submission_count, window_start_time)
                        values (:sid, 0, :wstart)
                    """),
                    {"sid": sid, "wstart": now},
                )
                reset_time = (now + window).astimezone(timezone.utc)
                return True, RATE_LIMIT_MAX_PER_HOUR, reset_time.strftime("%H:%M")

            count = int(row["submission_count"] or 0)
            wstart = row["window_start_time"]
            if wstart.tzinfo is None:
                wstart = wstart.replace(tzinfo=timezone.utc)

            # Reset if expired
            if now - wstart >= window:
                conn.execute(
                    text("""
                        update public.rate_limits
                        set submission_count = 0, window_start_time = :wstart
                        where student_id = :sid
                    """),
                    {"sid": sid, "wstart": now},
                )
                reset_time = (now + window).astimezone(timezone.utc)
                return True, RATE_LIMIT_MAX_PER_HOUR, reset_time.strftime("%H:%M")

            remaining = RATE_LIMIT_MAX_PER_HOUR - count
            reset_time = (wstart + window).astimezone(timezone.utc)

            if remaining <= 0:
                return False, 0, reset_time.strftime("%H:%M")

            return True, remaining, reset_time.strftime("%H:%M")

    except Exception as e:
        # If rate-limit DB query fails, allow but log an error.
        LOGGER.error(f"{_kv(area='rate_limit', student_id=sid)} check failed: {type(e).__name__}: {e}")
        return True, RATE_LIMIT_MAX_PER_HOUR, ""


def increment_rate_limit(student_id: str):
    """Increment submission count, resetting window if needed."""
    eng = get_db_engine()
    if eng is None:
        return

    ensure_rate_limits_table()

    sid = _effective_student_id(student_id)
    now = _utc_now()
    window = timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)

    try:
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
                        values (:sid, 1, :wstart)
                    """),
                    {"sid": sid, "wstart": now},
                )
                return

            count = int(row["submission_count"] or 0)
            wstart = row["window_start_time"]
            if wstart.tzinfo is None:
                wstart = wstart.replace(tzinfo=timezone.utc)

            if now - wstart >= window:
                # New window starts now, count resets to 1
                conn.execute(
                    text("""
                        update public.rate_limits
                        set submission_count = 1, window_start_time = :wstart
                        where student_id = :sid
                    """),
                    {"sid": sid, "wstart": now},
                )
            else:
                conn.execute(
                    text("""
                        update public.rate_limits
                        set submission_count = :cnt
                        where student_id = :sid
                    """),
                    {"sid": sid, "cnt": count + 1},
                )

    except Exception as e:
        LOGGER.error(f"{_kv(area='rate_limit', student_id=sid)} increment failed: {type(e).__name__}: {e}")


def insert_attempt(student_id: str, question_key: str, report: dict, mode: str):
    eng = get_db_engine()
    if eng is None:
        return
    ensure_attempts_table()

    sid = _effective_student_id(student_id)

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
        st.session_state["db_last_error"] = None
        LOGGER.info(f"{_kv(student_id=sid, question=question_key, mode=mode, marks=f'{m_awarded}/{m_max}')} Attempt inserted")
    except Exception as e:
        st.session_state["db_last_error"] = f"Insert Error: {type(e).__name__}: {e}"
        LOGGER.error(f"{_kv(student_id=sid, question=question_key, mode=mode)} DB insert failed: {type(e).__name__}: {e}")


def load_attempts_df(limit: int = 5000) -> pd.DataFrame:
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_attempts_table()
    try:
        with eng.connect() as conn:
            df = pd.read_sql(
                text("""
                    select created_at, student_id, question_key, mode, marks_awarded, max_marks,
                           readback_type
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
        LOGGER.error(f"{_kv(area='db')} Load attempts failed: {type(e).__name__}: {e}")
        return pd.DataFrame()

# =========================
#  CUSTOM QUESTION BANK TABLES
# =========================
def ensure_custom_questions_table():
    if st.session_state.get("custom_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return
    ddl = """
    create table if not exists public.custom_questions_v1 (
      id bigserial primary key,
      created_at timestamptz not null default now(),
      created_by text,
      assignment_name text not null,
      question_label text not null,
      max_marks int not null,
      tags jsonb,
      question_image_path text not null,
      markscheme_image_path text not null,
      question_text text,
      markscheme_text text
    );
    create index if not exists idx_custom_questions_assignment
      on public.custom_questions_v1 (assignment_name);
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(ddl))
        st.session_state["custom_table_ready"] = True
        LOGGER.info(f"{_kv(area='db')} custom_questions table ready")
    except Exception as e:
        st.session_state["db_last_error"] = f"Custom Table Creation Error: {type(e).__name__}: {e}"
        st.session_state["custom_table_ready"] = False
        LOGGER.error(f"{_kv(area='db')} custom_questions table creation failed: {type(e).__name__}: {e}")


def insert_custom_question(created_by: str,
                          assignment_name: str,
                          question_label: str,
                          max_marks: int,
                          tags: list,
                          q_path: str,
                          ms_path: str,
                          question_text: str = "",
                          markscheme_text: str = "") -> bool:
    eng = get_db_engine()
    if eng is None:
        return False
    ensure_custom_questions_table()

    query = """
    insert into public.custom_questions_v1
      (created_by, assignment_name, question_label, max_marks, tags,
       question_image_path, markscheme_image_path, question_text, markscheme_text)
    values
      (:created_by, :assignment_name, :question_label, :max_marks,
       CAST(:tags AS jsonb), :q_path, :ms_path, :question_text, :markscheme_text)
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(query), {
                "created_by": (created_by or "").strip() or None,
                "assignment_name": assignment_name.strip(),
                "question_label": question_label.strip(),
                "max_marks": int(max_marks),
                "tags": json.dumps(tags or []),
                "q_path": q_path,
                "ms_path": ms_path,
                "question_text": (question_text or "").strip()[:5000],
                "markscheme_text": (markscheme_text or "").strip()[:8000],
            })
        LOGGER.info(f"{_kv(area='bank', assignment=assignment_name, question_label=question_label)} Custom question inserted")
        return True
    except Exception as e:
        st.session_state["db_last_error"] = f"Insert Custom Question Error: {type(e).__name__}: {e}"
        LOGGER.error(f"{_kv(area='bank', assignment=assignment_name, question_label=question_label)} Insert failed: {type(e).__name__}: {e}")
        return False


def load_custom_questions_df(limit: int = 2000) -> pd.DataFrame:
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_custom_questions_table()
    try:
        with eng.connect() as conn:
            df = pd.read_sql(
                text("""
                    select id, created_at, assignment_name, question_label, max_marks
                    from public.custom_questions_v1
                    order by created_at desc
                    limit :limit
                """),
                conn,
                params={"limit": int(limit)},
            )
        return df
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Custom Questions Error: {type(e).__name__}: {e}"
        LOGGER.error(f"{_kv(area='bank')} Load custom questions failed: {type(e).__name__}: {e}")
        return pd.DataFrame()


def load_custom_question_by_id(qid: int) -> dict:
    eng = get_db_engine()
    if eng is None:
        return {}
    ensure_custom_questions_table()
    try:
        with eng.connect() as conn:
            row = conn.execute(
                text("""
                    select id, assignment_name, question_label, max_marks,
                           question_image_path, markscheme_image_path, question_text
                    from public.custom_questions_v1
                    where id = :id
                    limit 1
                """),
                {"id": int(qid)}
            ).mappings().first()
        return dict(row) if row else {}
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Custom Question Error: {type(e).__name__}: {e}"
        LOGGER.error(f"{_kv(area='bank', qid=qid)} Load question failed: {type(e).__name__}: {e}")
        return {}

# =========================
# --- STORAGE HELPERS ---
# =========================
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "untitled"


def upload_to_storage(path: str, file_bytes: bytes, content_type: str) -> bool:
    sb = get_supabase_client()
    if sb is None:
        st.session_state["db_last_error"] = "Supabase Storage not configured."
        LOGGER.error(f"{_kv(area='storage')} upload failed: supabase not configured")
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
        LOGGER.info(f"{_kv(area='storage', path=path, bytes=len(file_bytes))} Uploaded")
        return True
    except Exception as e:
        st.session_state["db_last_error"] = f"Storage Upload Error: {type(e).__name__}: {e}"
        LOGGER.error(f"{_kv(area='storage', path=path)} upload failed: {type(e).__name__}: {e}")
        return False


def download_from_storage(path: str) -> bytes:
    sb = get_supabase_client()
    if sb is None:
        return b""
    try:
        res = sb.storage.from_(STORAGE_BUCKET).download(path)
        if isinstance(res, (bytes, bytearray)):
            b = bytes(res)
        elif hasattr(res, "data") and res.data is not None and isinstance(res.data, (bytes, bytearray)):
            b = bytes(res.data)
        else:
            b = b""
        if b:
            LOGGER.info(f"{_kv(area='storage', path=path, bytes=len(b))} Downloaded")
        return b
    except Exception as e:
        st.session_state["db_last_error"] = f"Storage Download Error: {type(e).__name__}: {e}"
        LOGGER.error(f"{_kv(area='storage', path=path)} download failed: {type(e).__name__}: {e}")
        return b""


def bytes_to_pil(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img

# =========================
# 3) FILE SIZE VALIDATION + COMPRESSION
# =========================
def validate_image_file(file_bytes: bytes, max_mb: float, purpose: str) -> tuple[bool, str]:
    """
    Validate an image (bytes) BEFORE uploading.
    Checks:
      - size <= max_mb (allow slight oversize if compressible)
      - dimensions <= MAX_DIM_PX x MAX_DIM_PX
      - is a valid image
    Returns (ok, message). If not ok, message is user-friendly.
    """
    if not file_bytes:
        return False, f"{purpose}: empty file."

    size_mb = len(file_bytes) / (1024 * 1024)
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except Exception:
        return False, f"{purpose}: invalid image file."

    w, h = img.size
    if w > MAX_DIM_PX or h > MAX_DIM_PX:
        return False, f"Image dimensions too large ({w}x{h}). Max is {MAX_DIM_PX}x{MAX_DIM_PX}."

    # If within limit, ok
    if size_mb <= max_mb:
        return True, ""

    # Slightly over is allowed if we can compress (handled elsewhere)
    return False, f"Image too large ({size_mb:.1f}MB). Please use an image under {max_mb:.0f}MB."


def compress_image_if_needed(img: Image.Image, max_kb: int) -> Image.Image:
    """
    Returns a (possibly resized) PIL Image that is more likely to encode under max_kb.
    This function only changes dimensions (and mode). Actual byte size depends on encoding.
    """
    # Always work in RGB for predictable compression
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Downscale progressively until likely under max_kb when JPEG-encoded.
    # (Heuristic: keep shrinking by 10% each step, with a lower bound.)
    w, h = img.size
    min_side = 600  # avoid destroying legibility
    for _ in range(10):
        est_pixels = w * h
        # Rough heuristic: huge images are usually the problem
        if est_pixels <= 1_500_000:
            break
        w = int(w * 0.9)
        h = int(h * 0.9)
        if w < min_side or h < min_side:
            break
        img = img.resize((w, h))
    return img


def _encode_jpeg_under_kb(img: Image.Image, max_kb: int) -> tuple[bytes, str]:
    """
    Encode to JPEG, reducing quality and/or size until under max_kb.
    Returns (bytes, content_type).
    """
    img_work = compress_image_if_needed(img, max_kb=max_kb)

    # Try a few qualities
    for q in [85, 80, 75, 70, 65, 60]:
        buf = io.BytesIO()
        img_work.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        data = buf.getvalue()
        if len(data) <= max_kb * 1024:
            return data, "image/jpeg"

    # If still too large, downscale further
    w, h = img_work.size
    for _ in range(6):
        w = int(w * 0.85)
        h = int(h * 0.85)
        if w < 500 or h < 500:
            break
        img_work = img_work.resize((w, h))
        buf = io.BytesIO()
        img_work.save(buf, format="JPEG", quality=60, optimize=True, progressive=True)
        data = buf.getvalue()
        if len(data) <= max_kb * 1024:
            return data, "image/jpeg"

    # Return best-effort (may still be large)
    return data, "image/jpeg"


def _validate_and_maybe_compress_bytes(file_bytes: bytes, max_mb: float, purpose: str, allow_compress: bool = True) -> tuple[bool, str, bytes, str]:
    """
    Returns (ok, message, out_bytes, out_content_type).
    - If slightly over limit and allow_compress, compress automatically.
    """
    ok, msg = validate_image_file(file_bytes, max_mb=max_mb, purpose=purpose)
    if ok:
        # Keep original bytes, content-type guessed later
        return True, "", file_bytes, ""

    # If not ok due to size, we may compress.
    size_mb = len(file_bytes) / (1024 * 1024)
    if (not allow_compress) or (size_mb > max_mb * 1.25):
        # Too large to "slightly compress"
        return False, msg, b"", ""

    # Try compressing
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
        target_kb = int(max_mb * 1024)

        before_kb = len(file_bytes) / 1024
        out_bytes, out_ct = _encode_jpeg_under_kb(img, max_kb=target_kb)
        after_kb = len(out_bytes) / 1024

        if len(out_bytes) <= target_kb * 1024:
            LOGGER.warning(f"{_kv(area='image', purpose=purpose)} compression applied (KB {before_kb:.0f} -> {after_kb:.0f})")
            return True, f"Compressed {purpose.lower()} to fit size limit.", out_bytes, out_ct

        # Still too big
        return False, f"Image too large ({size_mb:.1f}MB). Please use an image under {max_mb:.0f}MB.", b"", ""

    except Exception as e:
        LOGGER.error(f"{_kv(area='image', purpose=purpose)} compression failed: {type(e).__name__}: {e}")
        return False, msg, b"", ""


def _guess_ct_from_name(name: str) -> str:
    name = (name or "").lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    return "application/octet-stream"


def _human_mb(n_bytes: int) -> str:
    return f"{(n_bytes / (1024 * 1024)):.1f}MB"


# =========================
# --- HELPER FUNCTIONS ---
# =========================
def encode_image(image_pil: Image.Image) -> str:
    """
    For OpenAI image input.
    Keeps PNG but optimized. Canvas preprocessing below ensures <= 2MB.
    """
    buffered = io.BytesIO()
    # Use optimize for PNG; canvas flow further ensures size
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
        img = img.resize((MAX_IMAGE_WIDTH, int(img.height * ratio)))
    return img


def _ensure_canvas_under_limit(img: Image.Image, max_mb: float) -> Image.Image:
    """
    Ensure the preprocessed canvas image encodes under max_mb (after preprocessing).
    If slightly over, downscale more.
    """
    max_bytes = int(max_mb * 1024 * 1024)

    img_work = img
    for _ in range(10):
        buf = io.BytesIO()
        img_work.save(buf, format="PNG", optimize=True)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return img_work

        # Too large, downscale further
        w, h = img_work.size
        w2 = int(w * 0.85)
        h2 = int(h * 0.85)
        if w2 < 400 or h2 < 400:
            break
        img_work = img_work.resize((w2, h2))

    # If still too large, return last attempt (caller will block with error)
    return img_work


# =========================
# --- 2) BETTER PROGRESS INDICATORS ---
# =========================
def run_with_progress(fn, *,
                      status_label: str,
                      typical_range: str,
                      still_working_after_s: float = 15.0,
                      estimated_total_s: float = 10.0):
    """
    Run a blocking function in a worker thread and update Streamlit UI:
      - status box
      - progress bar (estimate-based)
      - "Still working..." after threshold
      - brief completion message

    Returns (result, duration_s).
    """
    start = time.time()
    progress = st.progress(0.0)

    with st.status(f"{status_label} (typically {typical_range})", expanded=True) as status:
        status.write("Starting analysis...")
        showed_slow_msg = False

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)

            # Poll until complete so we can update progress UI
            while True:
                if fut.done():
                    break

                elapsed = time.time() - start
                # estimate-based fill: asymptotically approaches 95%
                p = min(0.95, elapsed / max(estimated_total_s, 1e-6))
                progress.progress(float(p))

                if (elapsed > still_working_after_s) and not showed_slow_msg:
                    status.write("Still working... (complex answer detected)")
                    showed_slow_msg = True

                time.sleep(0.12)

            # Collect result (no long blocking now)
            result = fut.result()

        duration = time.time() - start
        progress.progress(1.0)
        status.update(label="✓ Analysis complete!", state="complete", expanded=False)

    # Small post-complete message (non-blocking)
    st.toast("✓ Analysis complete!", icon="✅")
    return result, duration


# =========================
# --- MARKING + READBACK (BUILT-IN) ---
# =========================
def get_gpt_feedback(student_answer, q_data, is_image=False):
    """
    Returns a dict with marking report.
    If is_image=True, also includes an AI readback of the handwriting/drawing so the student can check.
    """
    max_marks = q_data["marks"]

    system_instr = f"""
You are a strict GCSE Physics examiner.

CONFIDENTIALITY RULE (CRITICAL):
- The mark scheme is confidential. Do NOT reveal it, quote it, or paraphrase it.
- When producing the readback, ONLY describe what is in the student's work. Do not use the mark scheme.
- Output ONLY valid JSON, nothing else.

Schema:
{{
  "readback_type": "<handwriting|diagram|mixed|unknown>",
  "readback_markdown": "<Markdown with LaTeX where helpful. Keep it concise but complete. Use $...$ for maths.>",
  "readback_warnings": ["<optional warning 1>", "<optional warning 2>"],
  "marks_awarded": <int>,
  "max_marks": <int>,
  "summary": "<1-2 sentences>",
  "feedback_points": ["<bullet 1>", "<bullet 2>"],
  "next_steps": ["<action 1>", "<action 2>"]
}}

Question: {q_data["question"]}
Max Marks: {max_marks}
""".strip()

    messages = [{"role": "system", "content": system_instr}]
    messages.append({
        "role": "system",
        "content": f"CONFIDENTIAL MARKING SCHEME (DO NOT REVEAL): {q_data['mark_scheme']}"
    })

    if is_image:
        base64_img = encode_image(student_answer)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Mark this work. Also provide a short readback of what you think the student wrote/drew, in Markdown with LaTeX if needed. Return JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Student Answer:\n{student_answer}\n\nReturn JSON only. (readback_markdown can be empty for typed answers)"
        })

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

        readback_md = str(data.get("readback_markdown", "") or "").strip()
        readback_type = str(data.get("readback_type", "") or "").strip()
        readback_warn = data.get("readback_warnings", [])
        if not isinstance(readback_warn, list):
            readback_warn = []

        return {
            "readback_type": readback_type,
            "readback_markdown": readback_md,
            "readback_warnings": [str(x) for x in readback_warn][:6],
            "marks_awarded": clamp_int(data.get("marks_awarded", 0), 0, max_marks),
            "max_marks": max_marks,
            "summary": str(data.get("summary", "")).strip(),
            "feedback_points": [str(x) for x in data.get("feedback_points", [])][:6],
            "next_steps": [str(x) for x in data.get("next_steps", [])][:6]
        }

    except Exception as e:
        LOGGER.error(f"{_kv(area='openai')} Marking error: {type(e).__name__}: {e}")
        return {
            "readback_type": "",
            "readback_markdown": "",
            "readback_warnings": [],
            "marks_awarded": 0,
            "max_marks": max_marks,
            "summary": "The examiner could not process this attempt (AI Error).",
            "feedback_points": ["Please try submitting again.", f"Error details: {str(e)[:50]}"],
            "next_steps": []
        }

# =========================
# --- MARKING + READBACK (CUSTOM QUESTION IMAGES) ---
# =========================
def get_gpt_feedback_custom(student_answer,
                            question_img: Image.Image,
                            markscheme_img: Image.Image,
                            max_marks: int,
                            is_student_image: bool = False):
    """
    Returns a dict with marking report.
    If is_student_image=True, also includes an AI readback of the student's handwriting/drawing.
    """
    system_instr = f"""
You are a strict GCSE Physics examiner.

CONFIDENTIALITY RULE (CRITICAL):
- The mark scheme is confidential. Do NOT reveal it, quote it, or paraphrase it.
- When producing the readback, ONLY describe what is in the student's work. Do not use the mark scheme.
- Output ONLY valid JSON, nothing else.

Schema:
{{
  "readback_type": "<handwriting|diagram|mixed|unknown>",
  "readback_markdown": "<Markdown with LaTeX where helpful. Keep it concise but complete. Use $...$ for maths.>",
  "readback_warnings": ["<optional warning 1>", "<optional warning 2>"],
  "marks_awarded": <int>,
  "max_marks": <int>,
  "summary": "<1-2 sentences>",
  "feedback_points": ["<bullet 1>", "<bullet 2>"],
  "next_steps": ["<action 1>", "<action 2>"]
}}

Max Marks: {int(max_marks)}
""".strip()

    q_b64 = encode_image(question_img)
    ms_b64 = encode_image(markscheme_img)

    content = [
        {"type": "text", "text": "You will be shown: (1) question image, (2) mark scheme image (confidential), and the student answer. Mark it. If the student answer is an image, also provide a readback of what the student wrote/drew (Markdown + LaTeX if needed). Return JSON only."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{q_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ms_b64}"}},
    ]

    if is_student_image:
        sa_b64 = encode_image(student_answer)
        content.append({"type": "text", "text": "Student answer (image):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sa_b64}"}})
    else:
        content.append({"type": "text", "text": f"Student Answer (text):\n{student_answer}\n(readback_markdown can be empty for typed answers)"})

    messages = [
        {"role": "system", "content": system_instr},
        {"role": "user", "content": content}
    ]

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
            "next_steps": [str(x) for x in data.get("next_steps", [])[:6])
        }

    except Exception as e:
        LOGGER.error(f"{_kv(area='openai')} Custom marking error: {type(e).__name__}: {e}")
        return {
            "readback_type": "",
            "readback_markdown": "",
            "readback_warnings": [],
            "marks_awarded": 0,
            "max_marks": int(max_marks),
            "summary": "The examiner could not process this attempt (AI Error).",
            "feedback_points": ["Please try submitting again.", f"Error details: {str(e)[:50]}"],
            "next_steps": []
        }


def render_report(report: dict):
    # Readback box first (only if present)
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
#  MAIN APP UI
# ============================================================
tab_student, tab_teacher, tab_bank = st.tabs(["🧑‍🎓 Student", "🔒 Teacher Dashboard", "📚 Question Bank"])

# -------------------------
# STUDENT TAB
# -------------------------
with tab_student:
    top_col1, top_col2, top_col3 = st.columns([3, 2, 1])

    with top_col1:
        st.title("⚛️ PanPhy Skill Builder")
        st.caption(f"Powered by {MODEL_NAME}")

    with top_col3:
        if AI_READY:
            st.success("System Online", icon="🟢")
        else:
            st.error("API Error", icon="🔴")

    with top_col2:
        source_options = ["Built-in"]
        if db_ready() and supabase_ready():
            source_options.append("Teacher Uploads")
        source = st.selectbox("Question Source:", source_options)

    st.divider()

    col1, col2 = st.columns([5, 4])

    selected_is_custom = (source == "Teacher Uploads")
    q_key = None
    q_data = None
    question_img = None
    max_marks = None
    custom_row = {}

    with col1:
        st.subheader("📝 The Question")

        student_id = st.text_input(
            "Student ID",
            placeholder="e.g. 10A_23",
            help="Optional. Leave blank to submit anonymously."
        )

        # NEW: Teacher bypass (optional, minimal UI)
        with st.expander("Teacher override (bypass rate limits)", expanded=False):
            teacher_override_pw = st.text_input("Teacher password", type="password", key="teacher_override_pw")
            is_teacher_override = bool(teacher_override_pw and teacher_override_pw == st.secrets.get("TEACHER_PASSWORD", ""))

        eff_sid = _effective_student_id(student_id)

        # NEW: rate-limit indicator
        if db_ready() and not is_teacher_override:
            allowed_now, remaining_now, reset_at = check_rate_limit(student_id)
            st.caption(f"{remaining_now}/{RATE_LIMIT_MAX_PER_HOUR} attempts remaining this hour" + (f" (resets at {reset_at})" if reset_at else ""))
        elif is_teacher_override:
            st.caption("Teacher override enabled. Rate limit bypassed.")

        if not selected_is_custom:
            q_key = st.selectbox("Select Topic:", list(QUESTIONS.keys()))
            q_data = QUESTIONS[q_key]
            st.markdown(f"**{q_data['question']}**")
            st.caption(f"Max Marks: {q_data['marks']}")
            max_marks = q_data["marks"]
        else:
            ensure_custom_questions_table()

            if st.session_state["cached_dfq"] is None:
                dfq = load_custom_questions_df(limit=2000)
                st.session_state["cached_dfq"] = dfq
                st.session_state["cached_assignments"] = ["All"] + sorted(dfq["assignment_name"].dropna().unique().tolist()) if not dfq.empty else ["All"]
            else:
                dfq = st.session_state["cached_dfq"]

            if dfq is None or dfq.empty:
                st.info("No teacher-uploaded questions yet.")
            else:
                assignment_filter = st.selectbox("Assignment:", st.session_state["cached_assignments"], key="student_assignment_filter")

                map_key = f"labels_{assignment_filter}"
                if st.session_state.get("cached_labels_map_key") != map_key:
                    if assignment_filter != "All":
                        dfq2 = dfq[dfq["assignment_name"] == assignment_filter].copy()
                    else:
                        dfq2 = dfq.copy()

                    dfq2["label"] = dfq2.apply(
                        lambda r: f"{r['assignment_name']} | {r['question_label']} ({int(r['max_marks'])} marks) [id {int(r['id'])}]",
                        axis=1
                    )
                    labels_map = {row["label"]: int(row["id"]) for _, row in dfq2.iterrows()}
                    st.session_state["cached_labels_map"] = labels_map
                    st.session_state["cached_labels"] = list(labels_map.keys())
                    st.session_state["cached_labels_map_key"] = map_key

                choices = st.session_state.get("cached_labels", [])
                if not choices:
                    st.info("No questions in this assignment filter.")
                else:
                    choice = st.selectbox("Select Question:", choices, key="student_custom_choice")
                    chosen_id = int(st.session_state["cached_labels_map"][choice])

                    if st.session_state["selected_custom_id"] != chosen_id:
                        st.session_state["selected_custom_id"] = chosen_id
                        custom_row = load_custom_question_by_id(chosen_id)
                        st.session_state["cached_custom_row"] = custom_row

                        st.session_state["cached_q_path"] = custom_row.get("question_image_path")
                        st.session_state["cached_ms_path"] = custom_row.get("markscheme_image_path")

                        q_bytes = download_from_storage(st.session_state["cached_q_path"]) if st.session_state["cached_q_path"] else b""
                        st.session_state["cached_question_img"] = bytes_to_pil(q_bytes) if q_bytes else None

                        st.session_state["feedback"] = None
                        st.session_state["canvas_key"] += 1

                    custom_row = st.session_state.get("cached_custom_row") or {}
                    question_img = st.session_state.get("cached_question_img")

                    if custom_row:
                        max_marks = int(custom_row.get("max_marks", 1))
                        q_key = f"{CUSTOM_QUESTION_PREFIX}:{int(custom_row['id'])}:{custom_row.get('assignment_name','')}:{custom_row.get('question_label','')}"
                        qtext = (custom_row.get("question_text") or "").strip()

                        if question_img is not None:
                            st.image(question_img, caption="Question (teacher upload)", use_container_width=True)
                        else:
                            st.warning("Could not load question image from storage.")

                        if qtext:
                            st.markdown("**Extracted question text (optional):**")
                            st.write(qtext)

                        st.caption(f"Max Marks: {max_marks}")

        st.write("")
        tab_type, tab_write = st.tabs(["⌨️ Type Answer", "✍️ Write Answer"])

        # -------------------------
        # Type Answer
        # -------------------------
        with tab_type:
            answer = st.text_area("Type your working:", height=200, placeholder="Enter your answer here...")

            if st.button("Submit Text", type="primary", disabled=not AI_READY):
                if not answer.strip():
                    st.toast("Please type an answer first.", icon="⚠️")
                else:
                    mode = "text"

                    # 1) Rate limit check (teachers bypass)
                    if not is_teacher_override:
                        allowed, remaining, reset_at = check_rate_limit(student_id)
                        if not allowed:
                            LOGGER.warning(f"{_kv(student_id=eff_sid, question=q_key, mode=mode)} Rate limit reached (10/10 in window)")
                            st.error(f"You’ve reached the limit of 10 submissions per hour. Please try again at {reset_at}.")
                            st.stop()

                        # increment before the API call to protect costs even if user refreshes
                        increment_rate_limit(student_id)

                    LOGGER.info(f"{_kv(student_id=eff_sid, question=q_key, mode=mode)} Submission received")

                    def _do_mark():
                        if not selected_is_custom:
                            return get_gpt_feedback(answer, q_data, is_image=False)
                        else:
                            if not custom_row or question_img is None:
                                return {
                                    "readback_type": "",
                                    "readback_markdown": "",
                                    "readback_warnings": [],
                                    "marks_awarded": 0,
                                    "max_marks": int(max_marks or 1),
                                    "summary": "Custom question not ready (missing images).",
                                    "feedback_points": ["Please inform your teacher.", "Question image could not be loaded."],
                                    "next_steps": []
                                }
                            ms_path = st.session_state.get("cached_ms_path") or custom_row.get("markscheme_image_path")
                            ms_bytes = download_from_storage(ms_path) if ms_path else b""
                            if not ms_bytes:
                                return {
                                    "readback_type": "",
                                    "readback_markdown": "",
                                    "readback_warnings": [],
                                    "marks_awarded": 0,
                                    "max_marks": int(max_marks or 1),
                                    "summary": "Mark scheme image missing.",
                                    "feedback_points": ["Please inform your teacher."],
                                    "next_steps": []
                                }
                            ms_img = bytes_to_pil(ms_bytes)
                            return get_gpt_feedback_custom(
                                student_answer=answer,
                                question_img=question_img,
                                markscheme_img=ms_img,
                                max_marks=max_marks,
                                is_student_image=False
                            )

                    report, duration = run_with_progress(
                        _do_mark,
                        status_label="Marking...",
                        typical_range="5-10 seconds",
                        still_working_after_s=15.0,
                        estimated_total_s=8.0
                    )

                    st.session_state["feedback"] = report

                    LOGGER.info(f"{_kv(student_id=eff_sid, question=q_key, mode=mode, duration=f'{duration:.1f}s')} OpenAI marking completed")
                    LOGGER.info(f"{_kv(student_id=eff_sid, question=q_key, mode=mode, marks=f'{report.get('marks_awarded',0)}/{report.get('max_marks',0)')} Feedback generated")

                    if db_ready() and q_key:
                        insert_attempt(student_id, q_key, st.session_state["feedback"], mode=mode)

        # -------------------------
        # Write Answer (Canvas)
        # -------------------------
        with tab_write:
            tool_row = st.columns([2, 1])
            with tool_row[0]:
                tool = st.radio("Tool", ["Pen", "Eraser"], horizontal=True, label_visibility="collapsed")
            clear_clicked = tool_row[1].button("🗑️ Clear", use_container_width=True)

            if clear_clicked:
                st.session_state["feedback"] = None
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

            if st.button("Submit Writing", type="primary", disabled=not AI_READY):
                mode = "writing"

                if canvas_result.image_data is None or not canvas_has_ink(canvas_result.image_data):
                    st.toast("Canvas is empty!", icon="⚠️")
                else:
                    # 3) Canvas preprocessing + size validation BEFORE any upload / AI usage
                    img_for_ai = preprocess_canvas_image(canvas_result.image_data)
                    img_for_ai = _ensure_canvas_under_limit(img_for_ai, max_mb=MAX_CANVAS_MB)

                    # Validate size after preprocessing (hard block if still too large)
                    buf_chk = io.BytesIO()
                    img_for_ai.save(buf_chk, format="PNG", optimize=True)
                    canvas_bytes = buf_chk.getvalue()
                    if len(canvas_bytes) > int(MAX_CANVAS_MB * 1024 * 1024):
                        LOGGER.warning(f"{_kv(student_id=eff_sid, question=q_key, mode=mode)} Canvas image too large after preprocessing ({_human_mb(len(canvas_bytes))})")
                        st.error(f"Image too large ({_human_mb(len(canvas_bytes))}). Please write less densely or use a smaller canvas.")
                        st.stop()

                    # 1) Rate limit check (teachers bypass)
                    if not is_teacher_override:
                        allowed, remaining, reset_at = check_rate_limit(student_id)
                        if not allowed:
                            LOGGER.warning(f"{_kv(student_id=eff_sid, question=q_key, mode=mode)} Rate limit reached (10/10 in window)")
                            st.error(f"You’ve reached the limit of 10 submissions per hour. Please try again at {reset_at}.")
                            st.stop()

                        increment_rate_limit(student_id)

                    LOGGER.info(f"{_kv(student_id=eff_sid, question=q_key, mode=mode)} Submission received")

                    def _do_mark_canvas():
                        if not selected_is_custom:
                            return get_gpt_feedback(img_for_ai, q_data, is_image=True)

                        if not custom_row or question_img is None:
                            return {
                                "readback_type": "",
                                "readback_markdown": "",
                                "readback_warnings": [],
                                "marks_awarded": 0,
                                "max_marks": int(max_marks or 1),
                                "summary": "Custom question not ready (missing images).",
                                "feedback_points": ["Please inform your teacher.", "Question image could not be loaded."],
                                "next_steps": []
                            }

                        ms_path = st.session_state.get("cached_ms_path") or custom_row.get("markscheme_image_path")
                        ms_bytes = download_from_storage(ms_path) if ms_path else b""
                        if not ms_bytes:
                            return {
                                "readback_type": "",
                                "readback_markdown": "",
                                "readback_warnings": [],
                                "marks_awarded": 0,
                                "max_marks": int(max_marks or 1),
                                "summary": "Mark scheme image missing.",
                                "feedback_points": ["Please inform your teacher."],
                                "next_steps": []
                            }

                        ms_img = bytes_to_pil(ms_bytes)
                        return get_gpt_feedback_custom(
                            student_answer=img_for_ai,
                            question_img=question_img,
                            markscheme_img=ms_img,
                            max_marks=max_marks,
                            is_student_image=True
                        )

                    report, duration = run_with_progress(
                        _do_mark_canvas,
                        status_label="Analyzing handwriting...",
                        typical_range="8-15 seconds",
                        still_working_after_s=15.0,
                        estimated_total_s=12.0
                    )

                    st.session_state["feedback"] = report

                    LOGGER.info(f"{_kv(student_id=eff_sid, question=q_key, mode=mode, duration=f'{duration:.1f}s')} OpenAI marking completed")
                    LOGGER.info(f"{_kv(student_id=eff_sid, question=q_key, mode=mode, marks=f'{report.get('marks_awarded',0)}/{report.get('max_marks',0)')} Feedback generated")

                    if db_ready() and q_key:
                        insert_attempt(student_id, q_key, st.session_state["feedback"], mode=mode)

    with col2:
        st.subheader("👨‍🏫 Report")
        with st.container(border=True):
            if st.session_state["feedback"]:
                render_report(st.session_state["feedback"])
                st.divider()
                if st.button("Start New Attempt", use_container_width=True):
                    st.session_state["feedback"] = None
                    st.rerun()
            else:
                st.info("Submit an answer to receive feedback.")

# -------------------------
# TEACHER DASHBOARD TAB
# -------------------------
with tab_teacher:
    st.subheader("🔒 Teacher Dashboard")

    with st.expander("Database tools"):
        if st.button("Reconnect to database"):
            _cached_engine.clear()
            st.session_state["db_table_ready"] = False
            st.session_state["custom_table_ready"] = False
            st.session_state["rate_table_ready"] = False
            LOGGER.info(f"{_kv(area='db')} reconnect requested")
            st.rerun()
        if st.session_state.get("db_last_error"):
            st.write("Last DB error:")
            st.code(st.session_state["db_last_error"])

    if not st.secrets.get("DATABASE_URL", "").strip():
        st.info("Database not configured in secrets.")
    elif not db_ready():
        st.error("Database Connection Failed. Check drivers and URL.")
        if not get_db_driver_type():
            st.caption("No Postgres driver found. Add 'psycopg[binary]' (or psycopg2-binary) to requirements.txt")
        if st.session_state.get("db_last_error"):
            st.caption(st.session_state["db_last_error"])
    else:
        teacher_pw = st.text_input("Teacher password", type="password")
        if teacher_pw and teacher_pw == st.secrets.get("TEACHER_PASSWORD", ""):
            with st.status("Loading class data...", expanded=False) as status:
                df = load_attempts_df(limit=5000)
                status.update(label="✓ Loaded", state="complete")

            if st.session_state.get("db_last_error"):
                st.error(f"Database Error: {st.session_state['db_last_error']}")
                if st.button("Clear Error"):
                    st.session_state["db_last_error"] = ""
                    st.rerun()

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

# -------------------------
# QUESTION BANK TAB
# -------------------------
with tab_bank:
    st.subheader("📚 Question Bank (Upload one question at a time)")

    with st.expander("Database tools"):
        if st.button("Reconnect to database", key="reconnect_db_bank"):
            _cached_engine.clear()
            st.session_state["db_table_ready"] = False
            st.session_state["custom_table_ready"] = False
            st.session_state["rate_table_ready"] = False
            LOGGER.info(f"{_kv(area='db')} reconnect requested (bank)")
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
            st.caption("Enter the teacher password to upload/manage questions.")
        else:
            ensure_custom_questions_table()

            with st.form("upload_q_form", clear_on_submit=True):
                c1, c2 = st.columns([2, 1])
                with c1:
                    assignment_name = st.text_input("Assignment name", placeholder="e.g. AQA Paper 1 (Electricity)")
                    question_label = st.text_input("Question label", placeholder="e.g. Q3b")
                with c2:
                    max_marks_in = st.number_input("Max marks", min_value=1, max_value=50, value=3, step=1)

                tags_str = st.text_input("Tags (comma separated)", placeholder="forces, resultant, newton")
                q_text_opt = st.text_area("Optional: extracted question text (teacher edit)", height=100)
                st.caption("You can leave this blank and rely on the screenshot only.")

                q_file = st.file_uploader("Upload question screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
                ms_file = st.file_uploader("Upload mark scheme screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])

                submitted = st.form_submit_button("Save to Question Bank", type="primary")

            if submitted:
                if not assignment_name.strip() or not question_label.strip():
                    st.warning("Please fill in Assignment name and Question label.")
                elif q_file is None or ms_file is None:
                    st.warning("Please upload both the question screenshot and the mark scheme screenshot.")
                else:
                    assignment_slug = slugify(assignment_name)
                    qlabel_slug = slugify(question_label)
                    token = pysecrets.token_hex(6)

                    q_bytes_in = q_file.getvalue()
                    ms_bytes_in = ms_file.getvalue()

                    # 3) Validate and (optionally) compress BEFORE uploading
                    okq, msgq, q_bytes_out, q_ct_out = _validate_and_maybe_compress_bytes(
                        q_bytes_in, max_mb=MAX_Q_IMG_MB, purpose="Question image", allow_compress=True
                    )
                    if not okq:
                        st.error(msgq)
                        LOGGER.warning(f"{_kv(area='teacher_upload', purpose='question')} validation failed: {msgq}")
                        st.stop()

                    okm, msgm, ms_bytes_out, ms_ct_out = _validate_and_maybe_compress_bytes(
                        ms_bytes_in, max_mb=MAX_MS_IMG_MB, purpose="Mark scheme image", allow_compress=True
                    )
                    if not okm:
                        st.error(msgm)
                        LOGGER.warning(f"{_kv(area='teacher_upload', purpose='markscheme')} validation failed: {msgm}")
                        st.stop()

                    # Content-type: keep original if not re-encoded; else use new
                    q_ct = q_ct_out or _guess_ct_from_name(q_file.name)
                    ms_ct = ms_ct_out or _guess_ct_from_name(ms_file.name)

                    # Choose file extension consistent with content-type
                    q_ext = ".jpg" if q_ct == "image/jpeg" else ".png"
                    ms_ext = ".jpg" if ms_ct == "image/jpeg" else ".png"

                    q_path = f"{assignment_slug}/{token}/{qlabel_slug}_question{q_ext}"
                    ms_path = f"{assignment_slug}/{token}/{qlabel_slug}_markscheme{ms_ext}"

                    # Upload
                    ok1 = upload_to_storage(q_path, q_bytes_out, q_ct)
                    ok2 = upload_to_storage(ms_path, ms_bytes_out, ms_ct)

                    tags = [t.strip() for t in (tags_str or "").split(",") if t.strip()]

                    if ok1 and ok2:
                        ok_db = insert_custom_question(
                            created_by="teacher",
                            assignment_name=assignment_name,
                            question_label=question_label,
                            max_marks=int(max_marks_in),
                            tags=tags,
                            q_path=q_path,
                            ms_path=ms_path,
                            question_text=q_text_opt or "",
                            markscheme_text=""
                        )
                        if ok_db:
                            st.session_state["cached_dfq"] = None
                            st.session_state["cached_labels_map_key"] = None
                            st.success("Saved. This question is now available under 'Teacher Uploads' in the Student tab.")
                            LOGGER.info(f"{_kv(area='teacher_upload', assignment=assignment_name, question_label=question_label)} Saved question bank entry")
                        else:
                            st.error("Uploaded images, but failed to save metadata to DB. Check errors below.")
                            LOGGER.error(f"{_kv(area='teacher_upload')} Uploaded images but DB insert failed")
                    else:
                        st.error("Failed to upload one or both images to Supabase Storage. Check errors below.")
                        LOGGER.error(f"{_kv(area='teacher_upload')} Image upload failed")

            st.write("")
            st.write("### Recent uploaded questions")
            df_bank = load_custom_questions_df(limit=50)
            if df_bank.empty:
                st.info("No uploaded questions yet.")
            else:
                st.dataframe(df_bank, use_container_width=True)

            if st.session_state.get("db_last_error"):
                st.error(f"Error: {st.session_state['db_last_error']}")
                if st.button("Clear Error", key="clear_bank_err"):
                    st.session_state["db_last_error"] = ""
                    st.rerun()