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
import pathlib

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

# AI question generation (legacy fallback list; spec packs are preferred)
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

# =========================
# --- SUBJECTS (AQA GCSE Higher) ---
# =========================
DEFAULT_SUBJECT_KEY = "aqa_gcse_physics"

SUBJECT_CATALOG: Dict[str, str] = {
    # Separate sciences
    "aqa_gcse_biology": "AQA GCSE Biology (Separate, Higher)",
    "aqa_gcse_chemistry": "AQA GCSE Chemistry (Separate, Higher)",
    "aqa_gcse_physics": "AQA GCSE Physics (Separate, Higher)",
    # Combined Science Trilogy (discipline-specific)
    "aqa_combined_biology": "AQA GCSE Combined Science: Biology (Higher)",
    "aqa_combined_chemistry": "AQA GCSE Combined Science: Chemistry (Higher)",
    "aqa_combined_physics": "AQA GCSE Combined Science: Physics (Higher)",
}

SUBJECT_KEYS_ORDERED: List[str] = [
    "aqa_gcse_biology",
    "aqa_gcse_chemistry",
    "aqa_gcse_physics",
    "aqa_combined_biology",
    "aqa_combined_chemistry",
    "aqa_combined_physics",
]


def subject_label(subject_key: str) -> str:
    return SUBJECT_CATALOG.get(subject_key, subject_key or DEFAULT_SUBJECT_KEY)


# =========================
# --- SPEC PACK LOADING ---
# =========================
def _spec_path_for(subject_key: str) -> pathlib.Path:
    # Support both repo-root runs and Streamlit Cloud runs.
    sk = (subject_key or DEFAULT_SUBJECT_KEY).strip() or DEFAULT_SUBJECT_KEY
    p1 = pathlib.Path("specs") / f"{sk}.json"
    if p1.exists():
        return p1
    try:
        here = pathlib.Path(__file__).resolve().parent
        p2 = here / "specs" / f"{sk}.json"
        return p2
    except Exception:
        return p1


@st.cache_data(ttl=3600)
def load_spec_pack_cached(subject_key: str, mtime: float) -> Dict[str, Any]:
    p = _spec_path_for(subject_key)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        LOGGER.error("Spec pack parse failed", extra={"ctx": {"component": "spec", "error": type(e).__name__, "path": str(p)}})
        return {}


def load_spec_pack(subject_key: str) -> Dict[str, Any]:
    p = _spec_path_for(subject_key)
    try:
        mt = p.stat().st_mtime if p.exists() else 0.0
    except Exception:
        mt = 0.0
    return load_spec_pack_cached(subject_key, mt)


def allowed_topics_from_spec(spec_pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    topics = spec_pack.get("allowed_topics", [])
    if isinstance(topics, list):
        out = []
        for t in topics:
            if isinstance(t, dict) and str(t.get("code", "")).strip():
                out.append({
                    "code": str(t.get("code", "")).strip(),
                    "title": str(t.get("title", "")).strip(),
                    "ht_only": bool(t.get("ht_only", False)),
                })
        return out
    return []


def allowed_topic_codes(spec_pack: Dict[str, Any]) -> List[str]:
    return [t["code"] for t in allowed_topics_from_spec(spec_pack)]


def _summarize_allowed_topics_for_prompt(spec_pack: Dict[str, Any], max_items: int = 120) -> str:
    topics = allowed_topics_from_spec(spec_pack)
    if not topics:
        return "(No spec pack topics found. Use the official AQA specification.)"
    # Prefer higher-level codes (shorter) for compactness
    def depth(code: str) -> int:
        return len([p for p in code.split(".") if p.strip()])

    topics_sorted = sorted(topics, key=lambda t: (depth(t["code"]), t["code"]))
    items = topics_sorted[:max_items]
    lines = []
    for t in items:
        code = t["code"]
        title = t["title"]
        suffix = " (HT only)" if t.get("ht_only") else ""
        lines.append(f"- {code} {title}{suffix}".strip())
    if len(topics_sorted) > max_items:
        lines.append(f"... ({len(topics_sorted) - max_items} more)")
    return "\n".join(lines)

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

QUESTION_BANK_DDL = """
create table if not exists public.question_bank_v1 (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  subject_key text not null default 'aqa_gcse_physics',

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

create unique index if not exists uq_question_bank_subject_source_assignment_label
  on public.question_bank_v1 (subject_key, source, assignment_name, question_label);

create index if not exists idx_question_bank_subject
  on public.question_bank_v1 (subject_key);

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
_ss_init("subject_key", DEFAULT_SUBJECT_KEY)

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
      subject_key text not null default 'aqa_gcse_physics',
      student_id text not null,
      question_key text not null,
      mode text not null,
      marks_awarded int not null,
      max_marks int not null,
      summary text,
      feedback_points jsonb,
      next_steps jsonb
    );
    create index if not exists idx_attempts_subject_created_at
      on public.physics_attempts_v1 (subject_key, created_at desc);
    """

    ddl_alter = """
    alter table public.physics_attempts_v1
      add column if not exists subject_key text;
    update public.physics_attempts_v1
      set subject_key = 'aqa_gcse_physics'
      where subject_key is null or subject_key = '';
    alter table public.physics_attempts_v1
      alter column subject_key set not null;

    alter table public.physics_attempts_v1
      add column if not exists readback_type text;
    alter table public.physics_attempts_v1
      add column if not exists readback_markdown text;
    alter table public.physics_attempts_v1
      add column if not exists readback_warnings jsonb;

    create index if not exists idx_attempts_subject_created_at
      on public.physics_attempts_v1 (subject_key, created_at desc);
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
    ddl_migrate = """
    alter table public.question_bank_v1
      add column if not exists subject_key text;
    update public.question_bank_v1
      set subject_key = 'aqa_gcse_physics'
      where subject_key is null or subject_key = '';

    alter table public.question_bank_v1
      alter column subject_key set not null;

    drop index if exists public.uq_question_bank_source_assignment_label;

    create unique index if not exists uq_question_bank_subject_source_assignment_label
      on public.question_bank_v1 (subject_key, source, assignment_name, question_label);

    create index if not exists idx_question_bank_subject
      on public.question_bank_v1 (subject_key);
    """
    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, QUESTION_BANK_DDL)
            _exec_sql_many(conn, ddl_migrate)
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
        extra={"ctx": {"component": "image", "purpose": _purpose, "quality": best_quality, "bytes": len(best_bytes)}},
    )
    return True, best_bytes, ct, ""


def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def clamp_int(x, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        return int(default)
    return max(int(lo), min(int(hi), int(v)))


def safe_parse_json(s: str) -> dict:
    if not isinstance(s, str):
        return {}
    s2 = s.strip()
    if not s2:
        return {}
    try:
        return json.loads(s2)
    except Exception:
        # Try to salvage by extracting first {...}
        m = re.search(r"\{.*\}", s2, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


# ============================================================
# ATTEMPTS INSERT + LOAD
# ============================================================
def insert_attempt(student_id: str, subject_key: str, question_key: str, report: dict, mode: str):
    eng = get_db_engine()
    if eng is None:
        return
    ensure_attempts_table()

    sid = (student_id or "").strip() or f"anon_{st.session_state['anon_id']}"
    sk = (subject_key or DEFAULT_SUBJECT_KEY).strip() or DEFAULT_SUBJECT_KEY

    m_awarded = int(report.get("marks_awarded", 0))
    m_max = int(report.get("max_marks", 1))
    summ = str(report.get("summary", ""))[:1000]
    fb_json = json.dumps(report.get("feedback_points", [])[:6])
    ns_json = json.dumps(report.get("next_steps", [])[:6])

    rb_type = str(report.get("readback_type", "") or "")[:40]
    rb_md = str(report.get("readback_markdown", "") or "")[:6000]
    rb_warn = json.dumps(report.get("readback_warnings", [])[:6])

    query = """
    insert into public.physics_attempts_v1
        (student_id, subject_key, question_key, mode, marks_awarded, max_marks, summary, feedback_points, next_steps,
         readback_type, readback_markdown, readback_warnings)
    values
        (:student_id, :subject_key, :question_key, :mode, :marks_awarded, :max_marks, :summary,
         CAST(:feedback_points AS jsonb), CAST(:next_steps AS jsonb),
         :readback_type, :readback_markdown, CAST(:readback_warnings AS jsonb))
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(query), {
                "student_id": sid,
                "subject_key": sk,
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
def load_attempts_df_cached(_fp: str, subject_key: str, limit: int = 5000) -> pd.DataFrame:
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_attempts_table()
    sk = (subject_key or DEFAULT_SUBJECT_KEY).strip() or DEFAULT_SUBJECT_KEY
    with eng.connect() as conn:
        df = pd.read_sql(
            text("""
                select created_at, subject_key, student_id, question_key, mode, marks_awarded, max_marks, readback_type
                from public.physics_attempts_v1
                where subject_key = :sk
                order by created_at desc
                limit :limit
            """),
            conn,
            params={"limit": int(limit), "sk": sk},
        )
    if not df.empty:
        df["marks_awarded"] = pd.to_numeric(df["marks_awarded"], errors="coerce").fillna(0).astype(int)
        df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce").fillna(0).astype(int)
    return df


def load_attempts_df(subject_key: str, limit: int = 5000) -> pd.DataFrame:
    fp = (st.secrets.get("DATABASE_URL", "") or "")[:40]
    try:
        return load_attempts_df_cached(fp, subject_key=subject_key, limit=limit)
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Error: {type(e).__name__}: {e}"
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_question_bank_df_cached(_fp: str, subject_key: str, limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
    """
    NOTE: This returns a *summary* table for listing/filtering, so we include
    light metadata + tags/question_text (for search). Full row is loaded by id.
    """
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_question_bank_table()

    sk = (subject_key or DEFAULT_SUBJECT_KEY).strip() or DEFAULT_SUBJECT_KEY
    where_bits = ["subject_key = :sk"]
    if not include_inactive:
        where_bits.append("is_active = true")
    where = "where " + " and ".join(where_bits)

    with eng.connect() as conn:
        df = pd.read_sql(
            text(f"""
                select
                  id, created_at, updated_at,
                  subject_key,
                  source, assignment_name, question_label,
                  max_marks, tags, question_text,
                  is_active
                from public.question_bank_v1
                {where}
                order by created_at desc
                limit :limit
            """),
            conn,
            params={"limit": int(limit), "sk": sk},
        )
    return df


def load_question_bank_df(subject_key: str, limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
    fp = (st.secrets.get("DATABASE_URL", "") or "")[:40]
    try:
        return load_question_bank_df_cached(fp, subject_key=subject_key, limit=limit, include_inactive=include_inactive)
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
    subject_key: str,
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

    sk = (subject_key or DEFAULT_SUBJECT_KEY).strip() or DEFAULT_SUBJECT_KEY

    query = """
    insert into public.question_bank_v1
      (subject_key, source, created_by, assignment_name, question_label, max_marks, tags,
       question_text, question_image_path,
       markscheme_text, markscheme_image_path,
       is_active, updated_at)
    values
      (:subject_key, :source, :created_by, :assignment_name, :question_label, :max_marks,
       CAST(:tags AS jsonb),
       :question_text, :question_image_path,
       :markscheme_text, :markscheme_image_path,
       true, now())
    on conflict (subject_key, source, assignment_name, question_label) do update set
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
                "subject_key": sk,
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
def _mk_system_schema(subject_key: str, max_marks: int, question_text: str = "") -> str:
    qt = f"\nQuestion (student-facing):\n{question_text}\n" if question_text else "\n"
    subj = subject_label(subject_key)
    return f"""
You are a strict AQA GCSE examiner for {subj}.

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

    system_instr = _mk_system_schema(subject_key=q_row.get('subject_key', DEFAULT_SUBJECT_KEY), max_marks=max_marks, question_text=question_text if question_text else "")
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
# AI QUESTION GENERATOR (teacher-only, spec-grounded, vet, then save)
# ============================================================
MARKDOWN_LATEX_RULES = """
FORMATTING RULES:
- question_markdown MUST be valid Markdown and must render well in Streamlit's st.markdown.
- Use LaTeX only inside $...$ or $$...$$.
- Use (a), (b), (c) subparts as plain text, and put any equations in LaTeX.
- Use SI units and clear formatting.
""".strip()


PHYSICS_FORBIDDEN_PATTERNS = [
    (r"\\mu_0|\bmu0\b|\bμ0\b", "Uses μ0 (not GCSE)"),
    (r"\\epsilon_0|\bepsilon0\b|\bε0\b", "Uses ε0 (not GCSE)"),
    (r"\bB\s*=\s*\\mu_0\s*n\s*I\b|\bB\s*=\s*μ0\s*n\s*I\b", "Uses solenoid field equation B=μ0 n I (not GCSE)"),
    (r"\bflux\b|\bflux linkage\b|\binductance\b", "Uses flux/inductance language (not GCSE here)"),
    (r"\bFaraday\b|\bLenz\b", "Uses Faraday/Lenz law (not GCSE equation form here)"),
    (r"\bcalculus\b|\bdifferentiat|\bintegrat", "Uses calculus (not GCSE)"),
]


def _spec_focus_snippet(spec_pack: Dict[str, Any], topic_code: str, max_items: int = 10) -> str:
    topic_code = (topic_code or "").strip()
    if not topic_code:
        return ""
    cmap = spec_pack.get("content_map", {})
    if not isinstance(cmap, dict):
        return ""
    entry = cmap.get(topic_code, {})
    if not isinstance(entry, dict):
        return ""
    items: List[str] = []
    for k in ["content", "skills"]:
        v = entry.get(k, [])
        if isinstance(v, list):
            for x in v[:max_items]:
                s = str(x).strip()
                if s:
                    items.append(f"- {s}")
    return "\n".join(items[:max_items])


def _extract_total_from_marksheme(ms: str) -> Optional[int]:
    m = re.search(r"\bTOTAL\b\s*[:=]\s*(\d+)\b", ms or "", flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _has_part_marking(ms: str) -> bool:
    s = ms or ""
    return bool(re.search(r"(\([a-z]\)|\b[a-z]\))\s.*\[\s*\d+\s*\]", s, flags=re.IGNORECASE | re.DOTALL))


def _deterministic_draft_checks(subject_key: str, draft: Dict[str, Any], marks: int) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    qmd = str(draft.get("question_markdown", "") or "").strip()
    msmd = str(draft.get("mark_scheme_markdown", "") or "").strip()
    if not qmd:
        reasons.append("Missing question_markdown.")
    if not msmd:
        reasons.append("Missing mark_scheme_markdown.")

    try:
        tm = int(draft.get("total_marks", marks))
    except Exception:
        tm = None
    if tm != int(marks):
        reasons.append(f"total_marks must equal {int(marks)}.")

    total_line = _extract_total_from_marksheme(msmd)
    if total_line != int(marks):
        reasons.append(f"Mark scheme must end with: TOTAL = {int(marks)}")

    if not _has_part_marking(msmd):
        reasons.append("Mark scheme must include part-by-part marks like '(a) ... [2]'.")

    if "$" in qmd and "\\(" in qmd:
        reasons.append("Use $...$ for LaTeX, avoid \\(...\\).")

    sk = (subject_key or DEFAULT_SUBJECT_KEY)
    if sk in ("aqa_gcse_physics", "aqa_combined_physics"):
        combined = qmd + "\n" + msmd
        for pat, label in PHYSICS_FORBIDDEN_PATTERNS:
            if re.search(pat, combined, flags=re.IGNORECASE):
                reasons.append(label)

    # topic_tag presence
    tt = str(draft.get("topic_tag", "") or "").strip()
    if not tt:
        reasons.append("Missing topic_tag.")
    return (len(reasons) == 0), reasons


def _topic_code_set(spec_pack: Dict[str, Any]) -> set:
    return set(allowed_topic_codes(spec_pack))


def _build_generation_system_prompt(subject_key: str, spec_pack: Dict[str, Any]) -> str:
    subj = subject_label(subject_key)
    spec_topics = _summarize_allowed_topics_for_prompt(spec_pack, max_items=140)
    command_words = spec_pack.get("command_words", [])
    must_avoid = spec_pack.get("must_avoid", [])
    mark_rules = spec_pack.get("mark_scheme_rules", [])

    def _as_bullets(x, max_n: int = 20) -> str:
        if not isinstance(x, list):
            return ""
        xs = [str(i).strip() for i in x if str(i).strip()]
        xs = xs[:max_n]
        return "\n".join([f"- {i}" for i in xs])

    return f"""
You are an expert AQA GCSE question writer and examiner for {subj}.

Hard rules:
1) Create an ORIGINAL practice question. Do not reproduce copyrighted exam questions.
2) Must match AQA GCSE style and standard for the given subject and tier.
3) Stay strictly within the specification.
4) Marks must be allocated per part, and the total must equal total_marks EXACTLY.
5) The mark scheme must include marking points for EVERY part.
6) Return ONLY valid JSON, nothing else.

{MARKDOWN_LATEX_RULES}

Specification grounding:
Allowed topic codes and titles (choose ONE code for topic_tag that best matches the question):
{spec_topics}

Command words (preferred where appropriate):
{_as_bullets(command_words, max_n=20)}

Must avoid:
{_as_bullets(must_avoid, max_n=18)}

Mark scheme rules:
{_as_bullets(mark_rules, max_n=18)}

Output JSON schema (must match exactly):
{{
  "question_markdown": "string",
  "mark_scheme_markdown": "string",
  "total_marks": integer,
  "topic_tag": "string (one code from allowed_topics, e.g. '4.1.1')",
  "tags": ["string", "..."],
  "warnings": ["string", "..."]
}}

Mark scheme formatting requirements inside mark_scheme_markdown:
- Use a part-by-part breakdown with explicit mark allocation, for example:
  (a) ... [2]
  (b) ... [3]
- End with EXACTLY: TOTAL = <total_marks>
""".strip()


def _call_structured_generator(
    subject_key: str,
    spec_pack: Dict[str, Any],
    topic_tag_hint: str,
    topic_free_text: str,
    difficulty: str,
    qtype: str,
    marks: int,
    extra_instructions: str,
    repair: bool = False,
    repair_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    system = _build_generation_system_prompt(subject_key, spec_pack)
    sk = (subject_key or DEFAULT_SUBJECT_KEY).strip() or DEFAULT_SUBJECT_KEY

    hint = (topic_tag_hint or "").strip()
    free = (topic_free_text or "").strip()

    focus_snip = _spec_focus_snippet(spec_pack, hint) if hint else ""
    if focus_snip:
        focus_snip = "\nKey spec statements for this topic:\n" + focus_snip

    base_user = f"""
Topic code hint (if provided, you MUST use it as topic_tag): {hint if hint else "(none)"}
Topic focus / subtopic: {free if free else "(none)"}
Difficulty: {difficulty}
Question type: {qtype}
total_marks: {int(marks)}

Additional teacher instructions (optional):
{(extra_instructions or "").strip() if extra_instructions else "(none)"}

Constraints:
- Keep the question clearly GCSE and within the specification.
- Ensure the question is in Markdown and uses LaTeX only inside $...$ or $$...$$.
- End mark_scheme_markdown with EXACTLY: TOTAL = {int(marks)}.
{focus_snip}
""".strip()

    if not repair:
        user = base_user
    else:
        notes = "\n".join([f"- {n}" for n in (repair_notes or [])]) or "- (unspecified)"
        user = f"""
You previously generated a draft that failed checks. Fix it and return corrected JSON only.

Failures:
{notes}

You MUST:
- Keep total_marks unchanged: {int(marks)}.
- Use topic_tag = {hint} exactly if a hint was provided; otherwise choose a valid topic code.
- Fix all failures listed above.
""" + "\n\n" + base_user

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=2600,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = safe_parse_json(raw) or {}
        if not isinstance(data, dict):
            return {}
        # Normalise
        out = {
            "question_markdown": str(data.get("question_markdown", "") or "").strip(),
            "mark_scheme_markdown": str(data.get("mark_scheme_markdown", "") or "").strip(),
            "total_marks": clamp_int(data.get("total_marks", marks), 1, 50, default=int(marks)),
            "topic_tag": str(data.get("topic_tag", "") or "").strip(),
            "tags": data.get("tags", []),
            "warnings": data.get("warnings", []),
        }
        if not isinstance(out["tags"], list):
            out["tags"] = []
        out["tags"] = [str(t).strip() for t in out["tags"] if str(t).strip()][:12]
        if not isinstance(out["warnings"], list):
            out["warnings"] = []
        out["warnings"] = [str(w).strip() for w in out["warnings"] if str(w).strip()][:12]

        # Force topic_tag to hint when given
        if hint:
            out["topic_tag"] = hint

        # Ensure total_marks exact
        out["total_marks"] = int(marks)
        return out
    except Exception as e:
        LOGGER.error("AI generator failed", extra={"ctx": {"component": "ai_gen", "error": type(e).__name__, "subject": sk}})
        return {}


def spec_compliance_check(
    subject_key: str,
    spec_pack: Dict[str, Any],
    draft: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Second-pass compliance check (cheap): returns JSON:
      compliant true/false, reasons, suggested_fixes, topic_tag_valid
    """
    sk = (subject_key or DEFAULT_SUBJECT_KEY).strip() or DEFAULT_SUBJECT_KEY
    subj = subject_label(sk)

    allowed = allowed_topics_from_spec(spec_pack)
    # Keep payload compact
    allowed_compact = [{"code": t["code"], "title": t["title"]} for t in allowed[:260]]

    must_avoid = spec_pack.get("must_avoid", [])
    if not isinstance(must_avoid, list):
        must_avoid = []
    must_avoid = [str(x).strip() for x in must_avoid if str(x).strip()][:18]

    system = f"""
You are a strict AQA GCSE specification compliance auditor for {subj}.
You will be given a proposed ORIGINAL practice question and mark scheme.
Decide if it is compliant with the allowed topic codes list and GCSE level.

Output ONLY valid JSON in this schema:
{{
  "compliant": true/false,
  "topic_tag_valid": true/false,
  "reasons": ["..."],
  "suggested_fixes": ["..."]
}}

Rules:
- topic_tag_valid is true only if topic_tag is exactly one of the allowed topic codes provided.
- compliant should be false if the content is beyond-spec, not GCSE, or mismatched topic_tag.
- Do not be overly strict about phrasing, but be strict about content scope and exam level.
""".strip()

    user = {
        "topic_tag": str(draft.get("topic_tag", "") or "").strip(),
        "total_marks": int(draft.get("total_marks", 0) or 0),
        "question_markdown": str(draft.get("question_markdown", "") or "").strip(),
        "mark_scheme_markdown": str(draft.get("mark_scheme_markdown", "") or "").strip(),
        "allowed_topics": allowed_compact,
        "must_avoid": must_avoid,
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            max_completion_tokens=900,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        data = safe_parse_json(raw) or {}
        if not isinstance(data, dict):
            data = {}
    except Exception as e:
        data = {"compliant": False, "topic_tag_valid": False, "reasons": [f"Compliance check failed: {type(e).__name__}"], "suggested_fixes": []}

    # Defensive normalisation
    compliant = bool(data.get("compliant", False))
    tvalid = bool(data.get("topic_tag_valid", False))
    reasons = data.get("reasons", [])
    fixes = data.get("suggested_fixes", [])
    if not isinstance(reasons, list):
        reasons = []
    if not isinstance(fixes, list):
        fixes = []
    reasons = [str(r).strip() for r in reasons if str(r).strip()][:10]
    fixes = [str(f).strip() for f in fixes if str(f).strip()][:10]

    # Deterministic topic validation
    topic_tag = str(draft.get("topic_tag", "") or "").strip()
    if topic_tag and _topic_code_set(spec_pack):
        tvalid_det = topic_tag in _topic_code_set(spec_pack)
        tvalid = tvalid and tvalid_det
        if not tvalid_det:
            reasons.insert(0, "topic_tag is not one of the allowed topic codes.")
        compliant = compliant and tvalid_det

    # Physics deterministic forbidden patterns
    sk = (subject_key or DEFAULT_SUBJECT_KEY)
    if sk in ("aqa_gcse_physics", "aqa_combined_physics"):
        combined = (draft.get("question_markdown") or "") + "\n" + (draft.get("mark_scheme_markdown") or "")
        for pat, label in PHYSICS_FORBIDDEN_PATTERNS:
            if re.search(pat, combined, flags=re.IGNORECASE):
                compliant = False
                if label not in reasons:
                    reasons.append(label)

    return {
        "compliant": bool(compliant),
        "topic_tag_valid": bool(tvalid),
        "reasons": reasons,
        "suggested_fixes": fixes,
    }


def generate_practice_question_with_ai(
    subject_key: str,
    spec_pack: Dict[str, Any],
    topic_tag_hint: str,
    topic_free_text: str,
    difficulty: str,
    qtype: str,
    marks: int,
    extra_instructions: str = "",
) -> Dict[str, Any]:
    """
    Generation pipeline:
    - Structured generation (JSON)
    - Deterministic checks
    - Spec compliance check (AI)
    - Auto-revise once if non-compliant
    """
    draft = _call_structured_generator(
        subject_key=subject_key,
        spec_pack=spec_pack,
        topic_tag_hint=topic_tag_hint,
        topic_free_text=topic_free_text,
        difficulty=difficulty,
        qtype=qtype,
        marks=int(marks),
        extra_instructions=extra_instructions or "",
        repair=False,
    )

    ok_det, det_reasons = _deterministic_draft_checks(subject_key, draft, marks=int(marks))
    compliance = spec_compliance_check(subject_key, spec_pack, draft) if draft else {"compliant": False, "topic_tag_valid": False, "reasons": ["Empty draft"], "suggested_fixes": []}

    if ok_det and compliance.get("compliant", False):
        draft["compliance"] = compliance
        return draft

    # Auto-revise once
    repair_notes = det_reasons[:]
    if compliance and not compliance.get("compliant", False):
        repair_notes.extend([f"Compliance: {r}" for r in compliance.get("reasons", [])])

    draft2 = _call_structured_generator(
        subject_key=subject_key,
        spec_pack=spec_pack,
        topic_tag_hint=topic_tag_hint,
        topic_free_text=topic_free_text,
        difficulty=difficulty,
        qtype=qtype,
        marks=int(marks),
        extra_instructions=extra_instructions or "",
        repair=True,
        repair_notes=repair_notes[:14],
    )

    ok_det2, det_reasons2 = _deterministic_draft_checks(subject_key, draft2, marks=int(marks))
    compliance2 = spec_compliance_check(subject_key, spec_pack, draft2) if draft2 else {"compliant": False, "topic_tag_valid": False, "reasons": ["Empty draft"], "suggested_fixes": []}

    warnings = []
    if not ok_det2:
        warnings.extend(det_reasons2[:8])
    if not compliance2.get("compliant", False):
        warnings.extend([f"Compliance: {r}" for r in compliance2.get("reasons", [])][:8])

    draft2["warnings"] = list(dict.fromkeys([str(w).strip() for w in warnings if str(w).strip()]))[:12]
    draft2["compliance"] = compliance2
    draft2["save_blocked"] = not (ok_det2 and compliance2.get("compliant", False))
    return draft2

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
# AI HELPER: progress wrapper (existing)
# ============================================================
def _run_ai_with_progress(task_fn, ctx: dict, typical_range: str = "a few seconds", est_seconds: float = 8.0):
    start = time.time()
    try:
        with st.status("🤖 Working…", expanded=False) as status:
            status.update(label=f"🤖 Working… (usually {typical_range})", state="running")
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(task_fn)
                out = fut.result()
            status.update(label=f"✅ Done in {time.time() - start:.1f}s", state="complete")
        return out
    except Exception as e:
        LOGGER.error("AI task failed", extra={"ctx": {"component": "ai", "error": type(e).__name__, **(ctx or {})}})
        raise e

# ============================================================
# RENDER MD BOX (existing)
# ============================================================
def render_md_box(title: str, md: str, empty_text: str = "(empty)"):
    st.markdown(f"**{title}**")
    with st.container(border=True):
        if (md or "").strip():
            st.markdown(md)
        else:
            st.caption(empty_text)

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
        sel0, sel1, sel2 = st.columns([2, 2, 2])
        with sel0:
            subject_key = st.selectbox(
                "Subject",
                SUBJECT_KEYS_ORDERED,
                format_func=subject_label,
                key="student_subject",
            )
            prev_sk = st.session_state.get("active_subject_student", DEFAULT_SUBJECT_KEY)
            st.session_state["subject_key"] = subject_key
            if subject_key != prev_sk:
                st.session_state["active_subject_student"] = subject_key
                st.session_state["selected_qid"] = None
                st.session_state["cached_q_row"] = None
                st.session_state["cached_q_path"] = ""
                st.session_state["cached_ms_path"] = ""
                st.session_state["cached_question_img"] = None
                st.session_state["feedback"] = None
                st.session_state["last_canvas_image_data"] = None
                st.session_state["canvas_key"] += 1

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
            dfb = load_question_bank_df(subject_key=subject_key, limit=5000)
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

                        choice_key = f"student_question_choice::{subject_key}::{source}::{assignment_filter}"
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
                            insert_attempt(student_id, subject_key, q_key, st.session_state["feedback"], mode="text")

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

            if canvas_result.image_data is not None:
                st.session_state["last_canvas_image_data"] = canvas_result.image_data

            if st.button("Submit Drawing", type="primary", disabled=not AI_READY or not db_ready(), key="submit_canvas_btn"):
                sid = _effective_student_id(student_id)
                if not q_row:
                    st.error("Please select a question first.")
                elif st.session_state.get("last_canvas_image_data") is None:
                    st.warning("Please draw your answer first.")
                else:
                    try:
                        allowed_now, _, reset_str = _check_rate_limit_db(sid)
                    except Exception:
                        allowed_now, reset_str = True, ""
                    if not allowed_now:
                        st.error(f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}.")
                    else:
                        increment_rate_limit(sid)

                        img_arr = st.session_state["last_canvas_image_data"]
                        img = Image.fromarray(img_arr.astype("uint8"), mode="RGBA").convert("RGB")

                        def task2():
                            ms_path = (st.session_state.get("cached_ms_path") or q_row.get("markscheme_image_path") or "").strip()
                            ms_img = None
                            if ms_path:
                                fp = (st.secrets.get("SUPABASE_URL", "") or "")[:40]
                                ms_bytes = cached_download_from_storage(ms_path, fp)
                                ms_img = bytes_to_pil(ms_bytes) if ms_bytes else None
                            return get_gpt_feedback_from_bank(
                                student_answer=img,
                                q_row=q_row,
                                is_student_image=True,
                                question_img=question_img,
                                markscheme_img=ms_img
                            )

                        st.session_state["feedback"] = _run_ai_with_progress(
                            task_fn=task2,
                            ctx={"student_id": sid, "question": q_key or "", "mode": "image"},
                            typical_range="6-12 seconds",
                            est_seconds=10.0
                        )

                        if db_ready() and q_key:
                            insert_attempt(student_id, subject_key, q_key, st.session_state["feedback"], mode="image")

    with col2:
        st.subheader("🧠 AI Feedback")
        if not AI_READY:
            st.warning("AI is not connected. Configure OPENAI_API_KEY.")
        elif not q_row:
            st.info("Select a question first.")
        else:
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

            subject_key_dash = st.selectbox(
                "Subject",
                SUBJECT_KEYS_ORDERED,
                format_func=subject_label,
                key="teacher_dash_subject",
            )
            st.session_state["subject_key"] = subject_key_dash

            df = load_attempts_df(subject_key=subject_key_dash, limit=5000)

            if st.session_state.get("db_last_error"):
                st.error(f"Database Error: {st.session_state['db_last_error']}")

            if df.empty:
                st.info("No attempts logged yet for this subject.")
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

            subject_key_bank = st.selectbox(
                "Subject",
                SUBJECT_KEYS_ORDERED,
                format_func=subject_label,
                key="teacher_bank_subject",
            )
            st.session_state["subject_key"] = subject_key_bank
            spec_pack = load_spec_pack(subject_key_bank)

            st.write("### Question Bank manager")
            st.caption("Use the tabs below to (1) browse and preview what is already in the bank, (2) generate AI practice questions, or (3) upload scanned questions.")

            tab_browse, tab_ai, tab_upload = st.tabs(["🔎 Browse & preview", "🤖 AI generator", "🖼️ Upload scans"])

            # -------------------------
            # Browse & preview
            # -------------------------
            with tab_browse:
                st.write("## 🔎 Browse & preview")
                df_all = load_question_bank_df(subject_key=subject_key_bank, limit=5000, include_inactive=False)

                if df_all.empty:
                    st.info("No questions yet for this subject.")
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
                            srcx = str(r.get("source") or "").strip()
                            try:
                                mk = int(r.get("max_marks") or 0)
                            except Exception:
                                mk = 0
                            try:
                                qid = int(r.get("id"))
                            except Exception:
                                qid = -1
                            return f"{asg} | {ql} ({mk} marks) [{srcx}] [id {qid}]"

                        df_f["label"] = df_f.apply(_fmt_label, axis=1)
                        options = df_f["label"].tolist()

                        if "bank_preview_pick" in st.session_state and st.session_state["bank_preview_pick"] not in options:
                            st.session_state["bank_preview_pick"] = options[0]

                        pick = st.selectbox("Select a question to preview", options, key="bank_preview_pick")
                        pick_id = int(df_f.loc[df_f["label"] == pick, "id"].iloc[0])

                        row = load_question_by_id(pick_id) or {}
                        q_text2 = (row.get("question_text") or "").strip()
                        ms_text2 = (row.get("markscheme_text") or "").strip()

                        st.write("### Preview")
                        p1, p2 = st.columns([2, 2])
                        with p1:
                            st.markdown("**Question**")
                            with st.container(border=True):
                                qpath = (row.get("question_image_path") or "").strip()
                                if qpath:
                                    fp = (st.secrets.get("SUPABASE_URL", "") or "")[:40]
                                    b = cached_download_from_storage(qpath, fp)
                                    img = safe_bytes_to_pil(b)
                                    if img is not None:
                                        st.image(img, use_container_width=True)
                                    else:
                                        st.warning("Question image missing or could not be decoded.")
                                if q_text2:
                                    st.markdown(q_text2)
                                if (not qpath) and (not q_text2):
                                    st.caption("(no question content)")
                        with p2:
                            st.markdown("**Mark scheme (teacher only)**")
                            with st.container(border=True):
                                mspath = (row.get("markscheme_image_path") or "").strip()
                                if mspath:
                                    fp = (st.secrets.get("SUPABASE_URL", "") or "")[:40]
                                    b2 = cached_download_from_storage(mspath, fp)
                                    img2 = safe_bytes_to_pil(b2)
                                    if img2 is not None:
                                        st.image(img2, use_container_width=True)
                                    else:
                                        st.warning("Mark scheme image missing or could not be decoded.")
                                if ms_text2:
                                    st.markdown(ms_text2)
                                if (not mspath) and (not ms_text2):
                                    st.caption("(no mark scheme content)")

            # -------------------------
            # AI generator
            # -------------------------
            with tab_ai:
                st.write("## 🤖 AI generator")
                st.caption("Spec-grounded structured generation + compliance check. Teacher vetting required.")

                gen_c1, gen_c2 = st.columns([2, 1])
                with gen_c1:
                    topics = allowed_topics_from_spec(spec_pack)
                    topic_options: List[str] = []
                    if topics:
                        def _depth(code: str) -> int:
                            return len([p for p in (code or '').split('.') if p.strip()])
                        topics_small = [t for t in topics if _depth(t.get('code','')) <= 3]
                        topics_small = topics_small[:600]
                        topic_options = [f"{t['code']} {t['title']}".strip() for t in topics_small]

                    if topic_options:
                        topic_pick = st.selectbox(
                            "Topic (from specification)",
                            topic_options,
                            key="topic_pick",
                            help="Pick a spec code (recommended). You can add extra focus below.",
                        )
                        topic_tag_hint = (topic_pick.split(" ", 1)[0] if topic_pick else "").strip()
                    else:
                        st.caption("Spec pack not found or empty for this subject. You can still describe a topic below.")
                        topic_tag_hint = ""

                    topic_free_text = st.text_input(
                        "Focus / subtopic (optional)",
                        placeholder="e.g. stopping distance with thinking vs braking distance",
                        key="topic_text"
                    )

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
                    if not ((topic_tag_hint or "").strip() or (topic_free_text or "").strip()):
                        st.warning("Please choose a spec topic or describe a topic first.")
                    else:
                        def task_generate():
                            return generate_practice_question_with_ai(
                                subject_key=subject_key_bank,
                                spec_pack=spec_pack,
                                topic_tag_hint=topic_tag_hint,
                                topic_free_text=topic_free_text,
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

                        qtxt = str(draft_raw.get("question_markdown", "") or "").strip()
                        mstxt = str(draft_raw.get("mark_scheme_markdown", "") or "").strip()
                        mm = clamp_int(draft_raw.get("total_marks", int(marks_req)), 1, 50, default=int(marks_req))
                        tags = draft_raw.get("tags", [])
                        topic_tag = str(draft_raw.get("topic_tag", "") or "").strip()
                        warnings = draft_raw.get("warnings", [])
                        if not isinstance(tags, list):
                            tags = []
                        if not isinstance(warnings, list):
                            warnings = []

                        if not qtxt or not mstxt:
                            st.error("AI did not return a valid draft. Please try again.")
                        else:
                            token = pysecrets.token_hex(3)
                            topic_for_label = (topic_tag_hint or topic_free_text or "topic")
                            default_label = f"AI-{slugify(topic_for_label)[:24]}-{token}"

                            st.session_state["ai_draft"] = {
                                "assignment_name": (assignment_name_ai or "").strip() or "AI Practice",
                                "question_label": default_label,
                                "max_marks": int(mm),
                                "tags": [str(t).strip() for t in tags if str(t).strip()][:10],
                                "question_text": qtxt,
                                "topic_tag": topic_tag,
                                "markscheme_text": mstxt,
                                "warnings": warnings[:10],
                                "compliance": draft_raw.get("compliance", {}),
                                "save_blocked": bool(draft_raw.get("save_blocked", False)),
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
                        d_topic_tag = st.text_input("Topic tag (spec code)", value=d.get("topic_tag", ""), key="draft_topic_tag")
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
                            draft_for_check = {
                                "question_markdown": d_qtext.strip(),
                                "mark_scheme_markdown": d_mstext.strip(),
                                "total_marks": int(d_marks),
                                "topic_tag": (d_topic_tag or "").strip(),
                            }

                            ok_det_now, det_reasons_now = _deterministic_draft_checks(
                                subject_key_bank, draft_for_check, marks=int(d_marks)
                            )
                            compliance_now = spec_compliance_check(subject_key_bank, spec_pack, draft_for_check)

                            if not ok_det_now or not compliance_now.get("compliant", False):
                                st.error("Spec/format checks failed. Please fix the issues below before saving.")
                                if det_reasons_now:
                                    st.warning(
                                        "Format/marking issues:\n\n"
                                        + "\n".join([f"- {r}" for r in det_reasons_now[:10]])
                                    )
                                if compliance_now.get("reasons"):
                                    st.warning(
                                        "Spec compliance issues:\n\n"
                                        + "\n".join([f"- {r}" for r in compliance_now.get("reasons", [])[:10]])
                                    )
                                if compliance_now.get("suggested_fixes"):
                                    st.info(
                                        "Suggested fixes:\n\n"
                                        + "\n".join([f"- {f}" for f in compliance_now.get("suggested_fixes", [])[:10]])
                                    )
                                st.stop()

                            tags = [t.strip() for t in (d_tags_str or "").split(",") if t.strip()]
                            tt = (d_topic_tag or "").strip()
                            if tt:
                                topic_tag_token = f"topic:{tt}"
                                if topic_tag_token not in tags:
                                    tags = [topic_tag_token] + tags

                            ok = insert_question_bank_row(
                                subject_key=subject_key_bank,
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

                        q_path = f"{subject_key_bank}/{assignment_slug}/{token}/{qlabel_slug}_question{q_ext}"
                        ms_path = f"{subject_key_bank}/{assignment_slug}/{token}/{qlabel_slug}_markscheme{ms_ext}"

                        ok1 = upload_to_storage(q_path, q_bytes, q_ct)
                        ok2 = upload_to_storage(ms_path, ms_bytes, ms_ct)

                        tags = [t.strip() for t in (tags_str or "").split(",") if t.strip()]

                        if ok1 and ok2:
                            ok_db = insert_question_bank_row(
                                subject_key=subject_key_bank,
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
