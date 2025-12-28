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
from typing import Tuple, Optional, Dict, Any

# ============================================================
# 4) LOGGING FOR DEBUGGING (configure at startup)
# ============================================================
class KVFormatter(logging.Formatter):
    """Formatter that appends structured key-value pairs: [k=v] ..."""

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
        logger.warning("File logging not available", extra={"ctx": {"component": "logger"}})

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

# Rate limiting constants
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW_SECONDS = 60 * 60  # 1 hour

# Image limits
MAX_DIM_PX = 4000
QUESTION_MAX_MB = 5.0
MARKSCHEME_MAX_MB = 5.0
CANVAS_MAX_MB = 2.0

# ============================================================
# Built-in question bank (used for seeding into DB)
# ============================================================
BUILTIN_ASSIGNMENT_NAME = "Built-in Bank"

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

# ============================================================
# 1) NEW DATABASE TABLE DDL (rate_limits)
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

# ============================================================
# Unified Question Bank table (replaces built-in dict + custom_questions_v1)
# ============================================================
QUESTION_BANK_DDL = """
create table if not exists public.question_bank_v1 (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  source text not null check (source in ('built_in','teacher')),
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
if "bank_table_ready" not in st.session_state:
    st.session_state["bank_table_ready"] = False
if "is_teacher" not in st.session_state:
    st.session_state["is_teacher"] = False

# canvas robustness
if "last_canvas_image_data" not in st.session_state:
    st.session_state["last_canvas_image_data"] = None

# Cache question selection and images
if "selected_qid" not in st.session_state:
    st.session_state["selected_qid"] = None
if "cached_q_row" not in st.session_state:
    st.session_state["cached_q_row"] = None
if "cached_question_img" not in st.session_state:
    st.session_state["cached_question_img"] = None
if "cached_q_path" not in st.session_state:
    st.session_state["cached_q_path"] = None
if "cached_ms_path" not in st.session_state:
    st.session_state["cached_ms_path"] = None

# Cache list for student selector
if "cached_bank_df" not in st.session_state:
    st.session_state["cached_bank_df"] = None
if "cached_assignments" not in st.session_state:
    st.session_state["cached_assignments"] = ["All"]
if "cached_labels_map" not in st.session_state:
    st.session_state["cached_labels_map"] = {}
if "cached_labels" not in st.session_state:
    st.session_state["cached_labels"] = []
if "cached_labels_map_key" not in st.session_state:
    st.session_state["cached_labels_map_key"] = None

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
            conn.execute(text(ddl_create))
            conn.execute(text(ddl_alter))
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
            conn.execute(text(RATE_LIMITS_DDL))
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
            conn.execute(text(QUESTION_BANK_DDL))
        st.session_state["bank_table_ready"] = True
        LOGGER.info("Question bank table ready", extra={"ctx": {"component": "db", "table": "question_bank_v1"}})
    except Exception as e:
        st.session_state["db_last_error"] = f"Question Bank Table Error: {type(e).__name__}: {e}"
        st.session_state["bank_table_ready"] = False
        LOGGER.error("Question bank table ensure failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})

# ============================================================
# 1) RATE LIMITING (stored in Postgres)
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


def check_rate_limit(student_id: str) -> Tuple[bool, int, str]:
    if st.session_state.get("is_teacher", False):
        return True, RATE_LIMIT_MAX, ""

    eng = get_db_engine()
    if eng is None:
        LOGGER.warning("Rate limit DB not ready, allowing request", extra={"ctx": {"component": "rate_limit"}})
        return True, RATE_LIMIT_MAX, ""

    ensure_rate_limits_table()

    sid = (student_id or "").strip() or f"anon_{st.session_state['anon_id']}"
    now_utc = datetime.now(timezone.utc)

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

    except Exception as e:
        LOGGER.error("Rate limit check failed", extra={"ctx": {"component": "rate_limit", "student_id": sid, "error": type(e).__name__}})
        return True, RATE_LIMIT_MAX, ""


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
        LOGGER.info("Rate limit incremented", extra={"ctx": {"component": "rate_limit", "student_id": sid}})
    except Exception as e:
        LOGGER.error("Rate limit increment failed", extra={"ctx": {"component": "rate_limit", "student_id": sid, "error": type(e).__name__}})

# ============================================================
# Storage helpers
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
        LOGGER.info("Storage upload success", extra={"ctx": {"component": "storage", "path": path, "bytes": len(file_bytes)}})
        return True
    except Exception as e:
        st.session_state["db_last_error"] = f"Storage Upload Error: {type(e).__name__}: {e}"
        LOGGER.error("Storage upload failed", extra={"ctx": {"component": "storage", "path": path, "error": type(e).__name__}})
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
        LOGGER.error("Storage download failed", extra={"ctx": {"component": "storage", "path": path, "error": type(e).__name__}})
        return b""


def bytes_to_pil(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img

# ============================================================
# 3) FILE SIZE VALIDATION + COMPRESSION
# ============================================================
def _human_mb(num_bytes: int) -> str:
    return f"{(num_bytes / (1024 * 1024)):.1f}MB"


def validate_image_file(file_bytes: bytes, max_mb: float, purpose: str) -> Tuple[bool, str]:
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
    purpose: str,
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
        extra={"ctx": {"component": "image", "purpose": purpose, "from": _human_mb(size_bytes), "to": _human_mb(len(best_bytes)), "quality": best_quality}},
    )
    return True, best_bytes, ct, ""

# ============================================================
# Canvas helpers
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
# JSON helpers
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
# 2) BETTER PROGRESS INDICATORS
# ============================================================
def _run_ai_with_progress(task_fn, mode: str, ctx: dict, typical_range: str, est_seconds: float) -> dict:
    status_label = f"Marking… (typically {typical_range})" if mode == "text" else f"Analyzing handwriting… (typically {typical_range})"

    with st.status(status_label, expanded=True) as status:
        progress = st.progress(0)
        start = time.monotonic()
        still_working_shown = False

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(task_fn)

            while not fut.done():
                elapsed = time.monotonic() - start
                frac = min(0.95, max(0.02, elapsed / max(1e-6, est_seconds)))
                progress.progress(int(frac * 100))

                if elapsed > 15 and not still_working_shown:
                    status.update(label="Still working… (complex answer detected)", state="running", expanded=True)
                    still_working_shown = True

                time.sleep(0.12)

            report = fut.result()

        elapsed = time.monotonic() - start
        progress.progress(100)
        status.update(label="✓ Analysis complete!", state="complete", expanded=False)

    LOGGER.info("OpenAI marking completed", extra={"ctx": {**ctx, "duration_s": f"{elapsed:.2f}"}})
    return report

# ============================================================
# DB operations: attempts + question bank
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
        LOGGER.info("Attempt inserted", extra={"ctx": {"component": "db", "student_id": sid, "question": question_key, "mode": mode, "marks": f"{m_awarded}/{m_max}"}})
    except Exception as e:
        st.session_state["db_last_error"] = f"Insert Error: {type(e).__name__}: {e}"
        LOGGER.error("Attempt insert failed", extra={"ctx": {"component": "db", "student_id": sid, "question": question_key, "mode": mode, "error": type(e).__name__}})


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
        LOGGER.error("Load attempts failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})
        return pd.DataFrame()


def load_question_bank_df(limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
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
        LOGGER.error("Load question bank failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})
        return pd.DataFrame()


def load_question_by_id(qid: int) -> Dict[str, Any]:
    eng = get_db_engine()
    if eng is None:
        return {}
    ensure_question_bank_table()
    try:
        with eng.connect() as conn:
            row = conn.execute(
                text("""
                    select *
                    from public.question_bank_v1
                    where id = :id
                    limit 1
                """),
                {"id": int(qid)}
            ).mappings().first()
        return dict(row) if row else {}
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Question Error: {type(e).__name__}: {e}"
        LOGGER.error("Load question by id failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})
        return {}


def insert_question_bank_row(
    source: str,
    created_by: str,
    assignment_name: str,
    question_label: str,
    max_marks: int,
    tags: list,
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
                "question_text": (question_text or "").strip()[:8000] or None,
                "question_image_path": (question_image_path or "").strip() or None,
                "markscheme_text": (markscheme_text or "").strip()[:12000] or None,
                "markscheme_image_path": (markscheme_image_path or "").strip() or None,
            })
        LOGGER.info("Question bank row inserted (or existed)", extra={"ctx": {"component": "db", "source": source, "assignment": assignment_name, "label": question_label}})
        return True
    except Exception as e:
        st.session_state["db_last_error"] = f"Insert Question Bank Error: {type(e).__name__}: {e}"
        LOGGER.error("Insert question bank failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})
        return False


def seed_builtin_questions() -> Tuple[int, int]:
    """
    Inserts built-in QUESTIONS dict into question_bank_v1.
    Returns (inserted_count_estimate, total_count).
    Note: ON CONFLICT DO NOTHING means we cannot know exact inserted rows without extra queries,
    so we return best-effort counts.
    """
    total = len(QUESTIONS)
    inserted_est = 0
    for key, q in QUESTIONS.items():
        ok = insert_question_bank_row(
            source="built_in",
            created_by="seed",
            assignment_name=BUILTIN_ASSIGNMENT_NAME,
            question_label=key.strip(),
            max_marks=int(q["marks"]),
            tags=[],
            question_text=str(q["question"]),
            markscheme_text=str(q["mark_scheme"]),
            question_image_path=None,
            markscheme_image_path=None,
        )
        if ok:
            inserted_est += 1
    return inserted_est, total

# ============================================================
# Unified marking function for bank rows
# ============================================================
def _mk_system_schema(max_marks: int, question_text: str = "") -> str:
    qt = f"\nQuestion: {question_text}\n" if question_text else "\n"
    return f"""
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

    # If markscheme_text exists, provide it as confidential system content
    if markscheme_text:
        messages.append({"role": "system", "content": f"CONFIDENTIAL MARKING SCHEME (DO NOT REVEAL): {markscheme_text}"})

    # If question/markscheme images exist, provide them as images to the model
    # We keep everything in the user message content list
    content = []
    intro = "Mark this work. "
    intro += "If the student answer is an image, also provide a readback (Markdown + LaTeX if needed). "
    intro += "Return JSON only."
    content.append({"type": "text", "text": intro})

    if question_img is not None:
        content.append({"type": "text", "text": "Question image:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(question_img)}"}})

    if markscheme_img is not None:
        content.append({"type": "text", "text": "Mark scheme image (confidential):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(markscheme_img)}"}})

    if not is_student_image:
        # typed answer
        if question_text and question_img is None:
            content.append({"type": "text", "text": f"Question text:\n{question_text}"})
        content.append({"type": "text", "text": f"Student Answer (text):\n{student_answer}\n(readback_markdown can be empty for typed answers)"})
    else:
        # image answer
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
        LOGGER.error("Bank marking error", extra={"ctx": {"component": "openai", "error": type(e).__name__}})
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

# ============================================================
# Report renderer
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

# =========================
#  MAIN APP UI
# =========================
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
        source_options = ["Built-in", "Teacher Uploads"]
        source = st.selectbox("Question Source:", source_options)

    st.divider()

    col1, col2 = st.columns([5, 4])

    q_key = None
    q_row: Dict[str, Any] = {}
    question_img = None
    max_marks = None

    with col1:
        st.subheader("📝 The Question")

        student_id = st.text_input(
            "Student ID",
            placeholder="e.g. 10A_23",
            help="Optional. Leave blank to submit anonymously."
        )

        effective_sid = _effective_student_id(student_id)
        allowed, remaining, reset_time_str = check_rate_limit(effective_sid)

        if st.session_state.get("is_teacher", False):
            st.caption("Teacher mode: rate limits bypassed.")
        else:
            if db_ready():
                st.caption(f"{remaining}/{RATE_LIMIT_MAX} attempts remaining this hour.")
            else:
                st.caption("Rate limit indicator unavailable (database not ready).")

        if not db_ready():
            st.error("Database not ready. Configure DATABASE_URL first.")
        else:
            ensure_question_bank_table()

            # Load and cache the question list
            if st.session_state["cached_bank_df"] is None:
                dfb = load_question_bank_df(limit=5000, include_inactive=False)
                st.session_state["cached_bank_df"] = dfb
                st.session_state["cached_assignments"] = ["All"] + sorted(dfb["assignment_name"].dropna().unique().tolist()) if not dfb.empty else ["All"]
            else:
                dfb = st.session_state["cached_bank_df"]

            # Filter by source
            src = "built_in" if source == "Built-in" else "teacher"
            df_src = dfb[dfb["source"] == src].copy() if dfb is not None and not dfb.empty else pd.DataFrame()

            if df_src.empty:
                if src == "built_in":
                    st.info("No built-in questions in the database yet. Ask your teacher to seed the built-in bank in the Question Bank tab.")
                else:
                    st.info("No teacher-uploaded questions yet.")
            else:
                assignment_filter = st.selectbox("Assignment:", st.session_state["cached_assignments"], key="student_assignment_filter")

                map_key = f"labels_{source}_{assignment_filter}"
                if st.session_state.get("cached_labels_map_key") != map_key:
                    if assignment_filter != "All":
                        df2 = df_src[df_src["assignment_name"] == assignment_filter].copy()
                    else:
                        df2 = df_src.copy()

                    df2["label"] = df2.apply(
                        lambda r: f"{r['assignment_name']} | {r['question_label']} ({int(r['max_marks'])} marks) [id {int(r['id'])}]",
                        axis=1
                    )
                    labels_map = {row["label"]: int(row["id"]) for _, row in df2.iterrows()}
                    st.session_state["cached_labels_map"] = labels_map
                    st.session_state["cached_labels"] = list(labels_map.keys())
                    st.session_state["cached_labels_map_key"] = map_key

                choices = st.session_state.get("cached_labels", [])
                if not choices:
                    st.info("No questions in this assignment filter.")
                else:
                    choice = st.selectbox("Select Question:", choices, key="student_choice")
                    chosen_id = int(st.session_state["cached_labels_map"][choice])

                    # Load question details only when selection changes
                    if st.session_state["selected_qid"] != chosen_id:
                        st.session_state["selected_qid"] = chosen_id
                        q_row = load_question_by_id(chosen_id)
                        st.session_state["cached_q_row"] = q_row

                        st.session_state["cached_q_path"] = q_row.get("question_image_path")
                        st.session_state["cached_ms_path"] = q_row.get("markscheme_image_path")

                        q_bytes = download_from_storage(st.session_state["cached_q_path"]) if st.session_state["cached_q_path"] else b""
                        st.session_state["cached_question_img"] = bytes_to_pil(q_bytes) if q_bytes else None

                        st.session_state["feedback"] = None
                        st.session_state["canvas_key"] += 1

                    q_row = st.session_state.get("cached_q_row") or {}
                    question_img = st.session_state.get("cached_question_img")

                    if q_row:
                        max_marks = int(q_row.get("max_marks", 1))
                        q_key = f"QB:{int(q_row['id'])}:{q_row.get('source','')}:{q_row.get('assignment_name','')}:{q_row.get('question_label','')}"

                        q_text = (q_row.get("question_text") or "").strip()

                        if question_img is not None:
                            st.image(question_img, caption="Question image", use_container_width=True)
                        elif q_text:
                            st.markdown(f"**{q_text}**")
                        else:
                            st.warning("This question has no question text or image.")

                        st.caption(f"Max Marks: {max_marks}")

        st.write("")
        tab_type, tab_write = st.tabs(["⌨️ Type Answer", "✍️ Write Answer"])

        # -------------------------
        # Type Answer
        # -------------------------
        with tab_type:
            answer = st.text_area("Type your working:", height=200, placeholder="Enter your answer here...")

            if st.button("Submit Text", type="primary", disabled=not AI_READY or not db_ready()):
                sid = _effective_student_id(student_id)
                ctx = {"student_id": sid, "question": q_key or "", "mode": "text"}

                if not answer.strip():
                    st.toast("Please type an answer first.", icon="⚠️")
                elif not q_row:
                    st.error("Please select a question first.")
                else:
                    allowed_now, remaining_now, reset_str = check_rate_limit(sid)
                    if not allowed_now:
                        msg = f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}."
                        st.error(msg)
                        LOGGER.warning("Rate limit reached", extra={"ctx": {**ctx, "remaining": remaining_now}})
                    else:
                        increment_rate_limit(sid)
                        LOGGER.info("Submission received", extra={"ctx": ctx})

                        def task():
                            # markscheme image (optional)
                            ms_path = st.session_state.get("cached_ms_path") or q_row.get("markscheme_image_path")
                            ms_bytes = download_from_storage(ms_path) if ms_path else b""
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
                            mode="text",
                            ctx=ctx,
                            typical_range="5-10 seconds",
                            est_seconds=9.0
                        )

                        rep = st.session_state["feedback"] or {}
                        LOGGER.info("Feedback generated", extra={"ctx": {**ctx, "marks": f"{rep.get('marks_awarded', 0)}/{rep.get('max_marks', 0)}"}})

                        if db_ready() and q_key:
                            insert_attempt(student_id, q_key, st.session_state["feedback"], mode="text")

        # -------------------------
        # Write Answer (Canvas) - form prevents rerun per stroke
        # -------------------------
        with tab_write:
            tool_row = st.columns([2, 1])
            with tool_row[0]:
                tool = st.radio("Tool", ["Pen", "Eraser"], horizontal=True, label_visibility="collapsed")
            clear_clicked = tool_row[1].button("🗑️ Clear", use_container_width=True)

            if clear_clicked:
                st.session_state["feedback"] = None
                st.session_state["last_canvas_image_data"] = None
                st.session_state["canvas_key"] += 1
                st.rerun()

            stroke_width = 2 if tool == "Pen" else 30
            stroke_color = "#000000" if tool == "Pen" else CANVAS_BG_HEX

            with st.form(key=f"write_form_{st.session_state['canvas_key']}"):
                canvas_result = st_canvas(
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=CANVAS_BG_HEX,
                    height=400,
                    width=600,
                    drawing_mode="freedraw",
                    key=f"canvas_{st.session_state['canvas_key']}",
                    display_toolbar=False,
                    update_streamlit=True,  # keep True so image_data is available at submit
                )
                submitted_writing = st.form_submit_button("Submit Writing", type="primary", disabled=not AI_READY or not db_ready())

            # Cache last inked image
            if canvas_result is not None and canvas_result.image_data is not None:
                if canvas_has_ink(canvas_result.image_data):
                    st.session_state["last_canvas_image_data"] = canvas_result.image_data

            if submitted_writing:
                sid = _effective_student_id(student_id)
                ctx = {"student_id": sid, "question": q_key or "", "mode": "writing"}

                if not q_row:
                    st.error("Please select a question first.")
                else:
                    img_data = canvas_result.image_data if (canvas_result and canvas_result.image_data is not None) else st.session_state.get("last_canvas_image_data")

                    if img_data is None or not canvas_has_ink(img_data):
                        st.toast("Canvas is empty!", icon="⚠️")
                    else:
                        allowed_now, remaining_now, reset_str = check_rate_limit(sid)
                        if not allowed_now:
                            msg = f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}."
                            st.error(msg)
                            LOGGER.warning("Rate limit reached", extra={"ctx": {**ctx, "remaining": remaining_now}})
                        else:
                            img_for_ai = preprocess_canvas_image(img_data)

                            # Validate canvas size after preprocessing
                            canvas_bytes = _encode_image_bytes(img_for_ai, "JPEG", quality=80)
                            ok_canvas, msg_canvas = validate_image_file(canvas_bytes, CANVAS_MAX_MB, "canvas")
                            if not ok_canvas:
                                okc, outb, outct, err = _compress_bytes_to_limit(
                                    canvas_bytes, CANVAS_MAX_MB, purpose="canvas", prefer_fmt="JPEG"
                                )
                                if not okc:
                                    st.error(err or msg_canvas)
                                    LOGGER.warning("Canvas validation failed", extra={"ctx": {**ctx, "reason": err or msg_canvas}})
                                    st.stop()
                                img_for_ai = Image.open(io.BytesIO(outb)).convert("RGB")

                            increment_rate_limit(sid)
                            LOGGER.info("Submission received", extra={"ctx": ctx})

                            def task():
                                ms_path = st.session_state.get("cached_ms_path") or q_row.get("markscheme_image_path")
                                ms_bytes = download_from_storage(ms_path) if ms_path else b""
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
                                mode="writing",
                                ctx=ctx,
                                typical_range="8-15 seconds",
                                est_seconds=13.0
                            )

                            rep = st.session_state["feedback"] or {}
                            LOGGER.info("Feedback generated", extra={"ctx": {**ctx, "marks": f"{rep.get('marks_awarded', 0)}/{rep.get('max_marks', 0)}"}})

                            if db_ready() and q_key:
                                insert_attempt(student_id, q_key, st.session_state["feedback"], mode="writing")

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
            st.session_state["bank_table_ready"] = False
            st.session_state["cached_bank_df"] = None
            st.session_state["cached_labels_map_key"] = None
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
            st.session_state["is_teacher"] = True

            with st.status("Loading class data…", expanded=False):
                df = load_attempts_df(limit=5000)

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
    st.subheader("📚 Question Bank")

    with st.expander("Database tools"):
        if st.button("Reconnect to database", key="reconnect_db_bank"):
            _cached_engine.clear()
            st.session_state["db_table_ready"] = False
            st.session_state["bank_table_ready"] = False
            st.session_state["cached_bank_df"] = None
            st.session_state["cached_labels_map_key"] = None
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
            st.session_state["is_teacher"] = True
            ensure_question_bank_table()

            st.write("### Seed built-in question bank")
            seed_col1, seed_col2 = st.columns([1, 2])
            with seed_col1:
                if st.button("Seed built-in questions", type="primary", use_container_width=True):
                    with st.status("Seeding built-in questions…", expanded=False):
                        inserted_est, total = seed_builtin_questions()
                    st.session_state["cached_bank_df"] = None
                    st.session_state["cached_labels_map_key"] = None
                    st.success(f"Seed complete. Built-in bank now contains up to {total} questions (idempotent).")
            with seed_col2:
                st.caption("This is safe to click multiple times. It will not duplicate questions.")

            st.divider()
            st.write("### Upload a teacher question (image question + image mark scheme)")

            with st.form("upload_q_form", clear_on_submit=True):
                c1, c2 = st.columns([2, 1])
                with c1:
                    assignment_name = st.text_input("Assignment name", placeholder="e.g. AQA Paper 1 (Electricity)")
                    question_label = st.text_input("Question label", placeholder="e.g. Q3b")
                with c2:
                    max_marks_in = st.number_input("Max marks", min_value=1, max_value=50, value=3, step=1)

                tags_str = st.text_input("Tags (comma separated)", placeholder="forces, resultant, newton")
                q_text_opt = st.text_area("Optional: extracted question text (teacher edit)", height=100)

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

                    q_bytes_raw = q_file.getvalue()
                    ms_bytes_raw = ms_file.getvalue()

                    ok_q, msg_q = validate_image_file(q_bytes_raw, QUESTION_MAX_MB, "question image")
                    ok_ms, msg_ms = validate_image_file(ms_bytes_raw, MARKSCHEME_MAX_MB, "mark scheme image")

                    if not ok_q:
                        okc, q_bytes, q_ct, err = _compress_bytes_to_limit(q_bytes_raw, QUESTION_MAX_MB, purpose="question image")
                        if not okc:
                            st.error(err or msg_q)
                            LOGGER.error("Teacher upload rejected (question image)", extra={"ctx": {"component": "teacher_upload", "reason": err or msg_q}})
                            st.stop()
                    else:
                        q_bytes = q_bytes_raw
                        q_ct = "image/png" if (q_file.name or "").lower().endswith(".png") else "image/jpeg"

                    if not ok_ms:
                        okc, ms_bytes, ms_ct, err = _compress_bytes_to_limit(ms_bytes_raw, MARKSCHEME_MAX_MB, purpose="mark scheme image")
                        if not okc:
                            st.error(err or msg_ms)
                            LOGGER.error("Teacher upload rejected (mark scheme image)", extra={"ctx": {"component": "teacher_upload", "reason": err or msg_ms}})
                            st.stop()
                    else:
                        ms_bytes = ms_bytes_raw
                        ms_ct = "image/png" if (ms_file.name or "").lower().endswith(".png") else "image/jpeg"

                    q_ext = ".jpg" if q_ct == "image/jpeg" else ".png"
                    ms_ext = ".jpg" if ms_ct == "image/jpeg" else ".png"

                    q_path = f"{assignment_slug}/{token}/{qlabel_slug}_question{q_ext}"
                    ms_path = f"{assignment_slug}/{token}/{qlabel_slug}_markscheme{ms_ext}"

                    LOGGER.info("Teacher upload starting", extra={"ctx": {"component": "teacher_upload", "assignment": assignment_name, "label": question_label}})

                    ok1 = upload_to_storage(q_path, q_bytes, q_ct)
                    ok2 = upload_to_storage(ms_path, ms_bytes, ms_ct)

                    tags = [t.strip() for t in (tags_str or "").split(",") if t.strip()]

                    if ok1 and ok2:
                        ok_db = insert_question_bank_row(
                            source="teacher",
                            created_by="teacher",
                            assignment_name=assignment_name,
                            question_label=question_label,
                            max_marks=int(max_marks_in),
                            tags=tags,
                            question_text=q_text_opt or "",
                            markscheme_text="",
                            question_image_path=q_path,
                            markscheme_image_path=ms_path
                        )
                        if ok_db:
                            st.session_state["cached_bank_df"] = None
                            st.session_state["cached_labels_map_key"] = None
                            st.success("Saved. This question is now available under 'Teacher Uploads' in the Student tab.")
                        else:
                            st.error("Uploaded images, but failed to save metadata to DB. Check errors below.")
                    else:
                        st.error("Failed to upload one or both images to Supabase Storage. Check errors below.")

            st.write("")
            st.write("### Recent question bank entries")
            df_bank = load_question_bank_df(limit=50, include_inactive=False)
            if df_bank.empty:
                st.info("No questions yet.")
            else:
                st.dataframe(df_bank, use_container_width=True)

            if st.session_state.get("db_last_error"):
                st.error(f"Error: {st.session_state['db_last_error']}")
                if st.button("Clear Error", key="clear_bank_err"):
                    st.session_state["db_last_error"] = ""
                    st.rerun()