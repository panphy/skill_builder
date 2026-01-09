import streamlit as st
try:
    from components.panphy_stylus_canvas import stylus_canvas
except Exception:
    stylus_canvas = None  # fallback handled later
from PIL import Image
import io
import base64
import json
import re
import numpy as np
import pandas as pd
from sqlalchemy import text

import logging
from logging.handlers import RotatingFileHandler
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Dict, Any, List

import secrets as pysecrets

from ai_generation import AI_READY, MODEL_NAME, client, _render_template
from config import FEEDBACK_SYSTEM_TPL, SUBJECT_SITE
from db import db_ready, get_db_driver_type, get_db_engine, insert_question_bank_row
from ui_student import render_student_page
from ui_teacher import render_teacher_page

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

st.markdown(
    """
<style>
div[data-testid="stAppViewContainer"] > .main .block-container {
  padding-top: 1rem;
}
div[data-testid="stButton"] button[aria-label="⤢"] span,
div[data-testid="stButton"] button[aria-label="⤡"] span {
  font-size: 22px;
  line-height: 1;
  font-weight: 700;
}
.pp-hero {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  align-items: center;
  justify-content: space-between;
  padding: 2rem 2.5rem;
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(33, 99, 255, 0.08), rgba(138, 43, 226, 0.08));
  border: 1px solid rgba(148, 163, 184, 0.35);
  margin-bottom: 1.5rem;
}
.pp-hero-content {
  flex: 1 1 360px;
  min-width: 280px;
}
.pp-hero-kicker {
  font-size: 0.9rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #2563eb;
  font-weight: 700;
  margin-bottom: 0.5rem;
}
.pp-hero-title {
  font-size: 2.2rem;
  font-weight: 800;
  line-height: 1.1;
  margin-bottom: 0.75rem;
}
.pp-hero-subtitle {
  font-size: 1.05rem;
  color: #475569;
  margin-bottom: 1rem;
}
.pp-hero-list {
  margin: 0;
  padding-left: 1.25rem;
  color: #334155;
}
.pp-hero-card {
  flex: 0 1 260px;
  background: rgba(255, 255, 255, 0.7);
  border-radius: 18px;
  padding: 1.25rem 1.5rem;
  border: 1px solid rgba(148, 163, 184, 0.35);
  box-shadow: 0 16px 30px rgba(15, 23, 42, 0.08);
}
.pp-hero-card h4 {
  margin: 0 0 0.5rem 0;
}
.pp-hero-metric {
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}
footer {
  position: static;
  font-size: 15px;
  text-align: center;
  padding: 7px;
  background: transparent;
  color: #555;
  margin: 4px 0;
  width: 100%;
}
footer a {
  color: #ff5f1f;
  text-decoration: none;
}
footer a:hover {
  text-decoration: underline;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# --- CONSTANTS ---
# =========================
CANVAS_BG_HEX = "#ffffff"
CANVAS_BG_RGB = (255, 255, 255)
MAX_IMAGE_WIDTH = 1024
TEXTAREA_HEIGHT_DEFAULT = 320
TEXTAREA_HEIGHT_EXPANDED = 520
CANVAS_HEIGHT_DEFAULT = 520
CANVAS_HEIGHT_EXPANDED = 720

STORAGE_BUCKET = "physics-bank"

# Rate limiting
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW_SECONDS = 60 * 60  # 1 hour

# Image limits
MAX_DIM_PX = 4000
QUESTION_MAX_MB = 5.0
MARKSCHEME_MAX_MB = 5.0
CANVAS_MAX_MB = 2.0

# =========================
# --- SESSION STATE ---
# =========================
def _ss_init(k: str, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_init("canvas_key", 0)
_ss_init("feedback", None)
_ss_init("student_answer_text_single", "")
_ss_init("student_answer_text_journey", "")
_ss_init("anon_id", pysecrets.token_hex(4))
_ss_init("db_last_error", "")
_ss_init("db_table_ready", False)
_ss_init("bank_table_ready", False)
_ss_init("is_teacher", False)

# Canvas robustness cache
_ss_init("last_canvas_image_data", None)  # legacy
_ss_init("last_canvas_image_data_single", None)
_ss_init("last_canvas_image_data_journey", None)
_ss_init("last_canvas_data_url_single", None)
_ss_init("last_canvas_data_url_journey", None)
_ss_init("stylus_only_enabled", True)
_ss_init("canvas_cmd_nonce_single", 0)
_ss_init("canvas_cmd_nonce_journey", 0)

# Question selection cache
_ss_init("selected_qid", None)
_ss_init("cached_q_row", None)
_ss_init("cached_question_img", None)
_ss_init("cached_q_path", None)
_ss_init("cached_ms_path", None)

# AI generator draft cache (teacher-only)
_ss_init("ai_draft", None)

# Topic Journey state (student)
_ss_init("journey_step_index", 0)          # 0-based
_ss_init("journey_step_reports", [])       # list of per-step reports
_ss_init("journey_checkpoint_notes", {})   # step_index -> markdown
_ss_init("journey_active_id", None)        # question_bank_v2.id of current journey
_ss_init("journey_json_cache", None)       # parsed journey JSON for current selection

# Topic Journey draft (teacher)
_ss_init("journey_draft", None)

_ss_init("journey_topics_selected", [])
_ss_init("journey_gen_error_details", None)
_ss_init("journey_show_error", False)

# ============================================================
# TRACK (combined vs separate) - sticky via device localStorage + URL query param
# ============================================================
TRACK_PARAM = "track"
TRACK_DEFAULT = "combined"
TRACK_ALLOWED = {"combined", "separate"}
TRACK_STORAGE_KEY = f"panphy_track_{SUBJECT_SITE}"

def _get_query_param(key: str) -> str:
    # Streamlit changed query param APIs across versions; support both.
    try:
        v = st.query_params.get(key)
        if isinstance(v, list):
            return (v[0] or "").strip()
        return (v or "").strip()
    except Exception:
        qp = st.experimental_get_query_params()
        v = qp.get(key, [""])[0]
        return (v or "").strip()

def _set_query_param(**kwargs):
    try:
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

def _inject_track_restore_script():
    # If URL has no ?track=..., restore from localStorage and hard-reload once.
    st.markdown(
        f"""
<script>
(function() {{
  const KEY = {json.dumps(TRACK_STORAGE_KEY)};
  const DEFAULT = {json.dumps(TRACK_DEFAULT)};
  const url = new URL(window.location.href);
  const hasTrack = url.searchParams.has({json.dumps(TRACK_PARAM)});
  if (!hasTrack) {{
    const saved = window.localStorage.getItem(KEY);
    const useVal = (saved === "combined" || saved === "separate") ? saved : DEFAULT;
    url.searchParams.set({json.dumps(TRACK_PARAM)}, useVal);
    window.location.replace(url.toString());
  }}
}})();
</script>
""",
        unsafe_allow_html=True
    )

def _persist_track_to_browser(track_value: str):
    track_value = (track_value or "").strip().lower()
    if track_value not in TRACK_ALLOWED:
        track_value = TRACK_DEFAULT
    st.markdown(
        f"""
<script>
(function() {{
  const KEY = {json.dumps(TRACK_STORAGE_KEY)};
  try {{ window.localStorage.setItem(KEY, {json.dumps(track_value)}); }} catch (e) {{}}
}})();
</script>
""",
        unsafe_allow_html=True
    )

def init_track_state():
    # Run restore script first so first load picks up localStorage
    if "track_init_done" not in st.session_state:
        st.session_state["track_init_done"] = True
        _inject_track_restore_script()

    qp_track = _get_query_param(TRACK_PARAM).lower()
    if qp_track not in TRACK_ALLOWED:
        qp_track = TRACK_DEFAULT

    if st.session_state.get("track") not in TRACK_ALLOWED:
        st.session_state["track"] = qp_track

    # Keep URL in sync if needed
    if _get_query_param(TRACK_PARAM).lower() != st.session_state["track"]:
        _set_query_param(**{TRACK_PARAM: st.session_state["track"]})


# Initialize track selection (combined vs separate) early
init_track_state()

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

# =========================
# --- STYLUS CANVAS HELPERS ---
# =========================
def data_url_to_image_data(data_url: str) -> np.ndarray:
    """Convert data:image/png;base64,... into an RGBA numpy array."""
    if not data_url or not isinstance(data_url, str):
        raise ValueError("Missing data URL")
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    _header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    return np.array(img)


def _stylus_canvas_available() -> bool:
    return stylus_canvas is not None

# ============================================================
#  DATABASE HELPERS
# ============================================================
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
      question_bank_id bigint,
      step_index int,
      mode text not null,
      marks_awarded int not null,
      max_marks int not null,
      summary text,
      feedback_points jsonb,
      next_steps jsonb
    );
    """

    ddl_alter = f"""
    alter table public.physics_attempts_v1
      add column if not exists question_bank_id bigint;
    alter table public.physics_attempts_v1
      add column if not exists step_index int;

    alter table public.physics_attempts_v1
      add column if not exists readback_type text;
    alter table public.physics_attempts_v1
      add column if not exists readback_markdown text;
    alter table public.physics_attempts_v1
      add column if not exists readback_warnings jsonb;
    alter table public.physics_attempts_v1
      add column if not exists subject_site text not null default '{SUBJECT_SITE}';
    alter table public.physics_attempts_v1
      add column if not exists track text not null default 'combined';

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
    except Exception:
        return None
    return create_client(url, key)


def supabase_ready() -> bool:
    return get_supabase_client() is not None

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
    """Best-effort extractor for a JSON object from an LLM response.

    Priority:
    1) Parse entire string as JSON
    2) Parse any fenced ```json ... ``` blocks (prefer ones containing 'steps')
    3) Parse balanced JSON objects found in the text (prefer ones containing 'steps')
    """
    s = (text_str or "").strip()
    if not s:
        return None

    # 1) Direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    def _prefer(obj_list):
        """Prefer dicts containing a 'steps' list; else first dict."""
        for o in obj_list:
            if isinstance(o, dict) and isinstance(o.get("steps"), list) and len(o.get("steps")) > 0:
                return o
        for o in obj_list:
            if isinstance(o, dict):
                return o
        return obj_list[0] if obj_list else None

    # 2) Fenced json blocks
    objs = []
    for m in re.finditer(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE):
        try:
            objs.append(json.loads(m.group(1)))
        except Exception:
            continue
    picked = _prefer(objs)
    if picked is not None:
        return picked

    # 3) Balanced braces extraction: collect all top-level {...} objects
    candidates = []
    depth = 0
    in_str = False
    esc = False
    start = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(s[start:i+1])
                    start = None

    parsed = []
    for cand in candidates:
        try:
            parsed.append(json.loads(cand))
        except Exception:
            continue

    picked = _prefer(parsed)
    if picked is not None:
        return picked

    return None



def clamp_int(value, lo, hi, default=0):
    try:
        v = int(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))


# ============================================================
# MARKDOWN + LaTeX RENDER HELPERS (robust)
# ============================================================
_MD_TOKEN_CODEBLOCK = re.compile(r"```.*?```", re.DOTALL)
_MD_TOKEN_INLINECODE = re.compile(r"`[^`\n]+`")
_MD_TOKEN_MATHBLOCK = re.compile(r"\$\$.*?\$\$", re.DOTALL)
# Inline math: a single $...$ on one line (avoid $$...$$ which is handled above)
_MD_TOKEN_MATHINLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

def _protect_segments(pattern: re.Pattern, text_in: str, store: Dict[str, str], prefix: str) -> str:
    def _repl(m):
        key = f"@@{prefix}{len(store)}@@"
        store[key] = m.group(0)
        return key
    return pattern.sub(_repl, text_in)

def _restore_segments(text_in: str, store: Dict[str, str]) -> str:
    out = text_in
    # restore in reverse insertion order to reduce accidental nested replacement
    for k in sorted(store.keys(), key=lambda x: -len(x)):
        out = out.replace(k, store[k])
    return out

def normalize_markdown_math(md_text: str) -> str:
    r"""
    Heuristic normalizer for Streamlit Markdown + MathJax rendering.

    Goals:
    - Preserve Markdown structure and existing math ($...$, $$...$$)
    - Preserve fenced code blocks and inline code
    - Convert \(...\) -> $...$ and \[...] -> $$...$$
    - Wrap simple 'standalone' math tokens like 3^2, v^2, a_t, a_{t} in $...$
    """
    s = (md_text or "")
    if not s.strip():
        return ""

    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Fix double-escaped LaTeX sequences from JSON-encoded model output.
    # Example: "\\Delta" -> "\Delta"
    s = re.sub(r"\\\\([A-Za-z])", r"\\\1", s)
    s = re.sub(r"\\\\([\\[\\]{}()^_])", r"\\\1", s)

    protected: Dict[str, str] = {}
    s = _protect_segments(_MD_TOKEN_CODEBLOCK, s, protected, "CB")
    s = _protect_segments(_MD_TOKEN_INLINECODE, s, protected, "IC")

    # Convert common LaTeX delimiters to MathJax-friendly ones
    s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", s, flags=re.DOTALL)

    def _normalize_unit_escapes(math_text: str) -> str:
        out = math_text
        out = re.sub(r"\\m\s*\^\s*\{?\s*-?1\s*\}?", r"\\,m^{-1}", out)
        out = re.sub(r"\\m\b", r"\\,m", out)
        out = re.sub(r"\\s\b", r"\\,s", out)
        out = re.sub(r"\\kg\b", r"\\,kg", out)
        return out

    def _normalize_units_in_math(text_in: str) -> str:
        def _fix_block(m: re.Match) -> str:
            return f"$${_normalize_unit_escapes(m.group(1))}$$"

        def _fix_inline(m: re.Match) -> str:
            return f"${_normalize_unit_escapes(m.group(1))}$"

        out = re.sub(r"\$\$(.*?)\$\$", _fix_block, text_in, flags=re.DOTALL)
        out = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _fix_inline, out, flags=re.DOTALL)
        return out

    s = _normalize_units_in_math(s)

    s = _protect_segments(_MD_TOKEN_MATHBLOCK, s, protected, "MB")
    s = _protect_segments(_MD_TOKEN_MATHINLINE, s, protected, "MI")

    # Wrap simple math tokens not already protected
    # Examples: 3^2, v^2, a_t, a_{t}, m/s^2 (will wrap s^2)
    token_pat = re.compile(
        r"(?<!\$)(?<!\\)(?<![A-Za-z0-9])"
        r"([A-Za-z0-9]+(?:/[A-Za-z0-9]+)*"
        r"(?:\s*(?:\^\s*[-+]?\d+|_\{[^}]+\}|_[A-Za-z0-9]+)))"
        r"(?![A-Za-z0-9])"
    )

    def _wrap(m: re.Match) -> str:
        expr = m.group(1)
        # tighten spacing around ^ and _
        expr = re.sub(r"\s*(\^|_)\s*", r"\1", expr)
        expr = expr.strip()
        return f"${expr}$"

    s = token_pat.sub(_wrap, s)

    s = _restore_segments(s, protected)
    return s


def render_md_box(title: str, md_text: str, caption: str = "", empty_text: str = ""):
    st.markdown(f"**{title}**")
    with st.container(border=True):
        txt = normalize_markdown_math((md_text or "").strip())
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
    """Run a blocking task while showing a full-page overlay to prevent mid-run interaction.

    IMPORTANT: Keep the progress display simple and avoid presenting a precise ETA.
    """
    overlay = st.empty()
    start_t = time.monotonic()

    def _render_overlay(subtitle: str, percent: int):
        # Render a full-page blocker with a simple percent progress bar.
        overlay.markdown(
            f"""
<style>
/* PanPhy full-page UI blocker */
.pp-overlay {{
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0,0,0,0.25);
  z-index: 999999;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: all;
}}
.pp-overlay-card {{
  width: min(560px, 92vw);
  background: rgba(255,255,255,0.96);
  border-radius: 16px;
  padding: 18px 18px 16px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.22);
  border: 1px solid rgba(0,0,0,0.12);
}}
.pp-row {{
  display: flex;
  gap: 14px;
  align-items: flex-start;
}}
.pp-spinner {{
  width: 26px;
  height: 26px;
  border-radius: 999px;
  border: 4px solid rgba(0,0,0,0.15);
  border-top-color: rgba(0,0,0,0.55);
  animation: pp-spin 0.9s linear infinite;
  flex: 0 0 auto;
  margin-top: 2px;
}}
@keyframes pp-spin {{
  from {{ transform: rotate(0deg); }}
  to   {{ transform: rotate(360deg); }}
}}
.pp-title {{
  font-size: 16px;
  font-weight: 700;
  color: rgba(0,0,0,0.85);
  margin: 0;
  line-height: 1.2;
}}
.pp-subtitle {{
  font-size: 13px;
  color: rgba(0,0,0,0.72);
  margin-top: 4px;
  line-height: 1.35;
}}
.pp-meta {{
  font-size: 12px;
  color: rgba(0,0,0,0.58);
  margin-top: 6px;
}}
.pp-progress {{
  margin-top: 14px;
  width: 100%;
  height: 10px;
  background: rgba(0,0,0,0.10);
  border-radius: 999px;
  overflow: hidden;
  position: relative;
}}
.pp-progress-fill {{
  height: 100%;
  background: #2f6df6;
  border-radius: 999px;
  transition: width 0.35s ease;
}}
.pp-note {{
  font-size: 12px;
  color: rgba(0,0,0,0.52);
  margin-top: 6px;
}}
</style>

<div class="pp-overlay">
  <div class="pp-overlay-card">
    <div class="pp-row">
      <div class="pp-spinner"></div>
      <div>
        <div class="pp-title">AI is working. Please wait...</div>
        <div class="pp-subtitle">{subtitle}</div>
        <div class="pp-meta">{percent}% • Typical: {typical_range}</div>
        <div class="pp-note">May take longer for complex tasks.</div>
      </div>
    </div>
    <div class="pp-progress">
      <div class="pp-progress-fill" style="width: {percent}%;"></div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    def _calc_percent(elapsed_s: float, done: bool = False) -> int:
        if done:
            return 100
        if est_seconds <= 0:
            return 0
        return min(95, max(0, int((elapsed_s / est_seconds) * 100)))

    _render_overlay("", 0)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(task_fn)
            # Update the elapsed timer a few times per second.
            while not fut.done():
                elapsed = time.monotonic() - start_t
                _render_overlay("", _calc_percent(elapsed))
                time.sleep(0.35)

            report = fut.result()

        _render_overlay("Done. Updating the page…", _calc_percent(time.monotonic() - start_t, done=True))
        time.sleep(0.08)
        return report
    finally:
        # Always remove the overlay, even if the AI call errors.
        overlay.empty()



def insert_attempt(student_id: str, question_key: str, report: dict, mode: str, question_bank_id: Optional[int] = None, step_index: Optional[int] = None):
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
        (subject_site, track, student_id, question_key, question_bank_id, step_index, mode, marks_awarded, max_marks, summary, feedback_points, next_steps,
         readback_type, readback_markdown, readback_warnings)
        values
        (:subject_site, :track, :student_id, :question_key, :question_bank_id, :step_index, :mode, :marks_awarded, :max_marks, :summary,
         CAST(:feedback_points AS jsonb), CAST(:next_steps AS jsonb),
         :readback_type, :readback_markdown, CAST(:readback_warnings AS jsonb))
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(query), {
                "subject_site": SUBJECT_SITE,
                "track": st.session_state.get("track","combined"),
                "student_id": sid,
                "question_key": question_key,
                "question_bank_id": int(question_bank_id) if question_bank_id is not None else None,
                "step_index": int(step_index) if step_index is not None else None,
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
def load_attempts_df_cached(_fp: str, subject_site: str, limit: int = 5000) -> pd.DataFrame:
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_attempts_table()
    subject_site = (subject_site or "").strip().lower() or SUBJECT_SITE
    with eng.connect() as conn:
        df = pd.read_sql(
            text("""
                select id, created_at, student_id, question_key, question_bank_id, step_index, mode,
                       marks_awarded, max_marks, readback_type
                from public.physics_attempts_v1
                where subject_site = :subject_site
                order by created_at desc
                limit :limit
            """),
            conn,
            params={"limit": int(limit), "subject_site": subject_site},
        )
    if not df.empty:
        df["marks_awarded"] = pd.to_numeric(df["marks_awarded"], errors="coerce").fillna(0).astype(int)
        df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce").fillna(0).astype(int)
    return df


def load_attempts_df(limit: int = 5000) -> pd.DataFrame:
    fp = (st.secrets.get("DATABASE_URL", "") or "")[:40]
    subject_site = (SUBJECT_SITE or "").strip().lower()
    try:
        return load_attempts_df_cached(fp, subject_site=subject_site, limit=limit)
    except Exception as e:
        st.session_state["db_last_error"] = f"Load Error: {type(e).__name__}: {e}"
        return pd.DataFrame()


def delete_attempt_by_id(attempt_id: int) -> bool:
    eng = get_db_engine()
    if eng is None:
        return False
    ensure_attempts_table()
    try:
        with eng.begin() as conn:
            res = conn.execute(
                text("delete from public.physics_attempts_v1 where id = :id and subject_site = :subject_site"),
                {"id": int(attempt_id), "subject_site": SUBJECT_SITE},
            )
        st.session_state["db_last_error"] = ""
        try:
            load_attempts_df_cached.clear()
        except Exception:
            pass
        return res.rowcount > 0
    except Exception as e:
        st.session_state["db_last_error"] = f"Delete Attempt Error: {type(e).__name__}: {e}"
        return False

# ============================================================
# MARKING (unified for question_bank_v2 rows)
# ============================================================
def _mk_system_schema(max_marks: int, question_text: str = "") -> str:
    qt = f"\nQuestion (student-facing):\n{question_text}\n" if question_text else "\n"
    tpl = (FEEDBACK_SYSTEM_TPL or "").strip()
    if not tpl:
        # Fallback: should not happen if prompts.json is present
        tpl = "You are a strict GCSE examiner. Output ONLY JSON."
    return _render_template(tpl, {
        "QT": qt,
        "MAX_MARKS": int(max_marks),
    })



def _finalize_report(data: dict, max_marks: int) -> dict:
    def _norm(s: str) -> str:
        return normalize_markdown_math(str(s or "").strip())

    readback_md = _norm(data.get("readback_markdown", ""))
    readback_type = str(data.get("readback_type", "") or "").strip()
    readback_warn = data.get("readback_warnings", [])
    if not isinstance(readback_warn, list):
        readback_warn = []

    feedback_points = data.get("feedback_points", [])
    if not isinstance(feedback_points, list):
        feedback_points = []
    next_steps = data.get("next_steps", [])
    if not isinstance(next_steps, list):
        next_steps = []

    return {
        "readback_type": readback_type,
        "readback_markdown": readback_md,
        "readback_warnings": [str(x) for x in readback_warn][:6],
        "marks_awarded": clamp_int(data.get("marks_awarded", 0), 0, int(max_marks)),
        "max_marks": int(max_marks),
        "summary": _norm(data.get("summary", "")),
        "feedback_points": [_norm(x) for x in feedback_points][:6],
        "next_steps": [_norm(x) for x in next_steps][:6]
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
# REPORT RENDERER
# ============================================================
def render_report(report: dict):
    readback_md = (report.get("readback_markdown") or "").strip()
    if readback_md:
        st.markdown("**AI readback (what it thinks you wrote/drew):**")
        with st.container(border=True):
            st.markdown(normalize_markdown_math(readback_md))

        rb_warn = report.get("readback_warnings", [])
        if rb_warn:
            st.caption("Readback notes:")
            for w in rb_warn[:6]:
                st.markdown(normalize_markdown_math(f"- {w}"))
        st.divider()

    st.markdown(f"**Marks:** {int(report.get('marks_awarded', 0))} / {int(report.get('max_marks', 0))}")
    if report.get("summary"):
        st.markdown(normalize_markdown_math(f"**Summary:** {report.get('summary')}"))
    if report.get("feedback_points"):
        st.markdown("**Feedback:**")
        for p in report["feedback_points"]:
            st.markdown(normalize_markdown_math(f"- {p}"))
    if report.get("next_steps"):
        st.markdown("**Next steps:**")
        for n in report["next_steps"]:
            st.markdown(normalize_markdown_math(f"- {n}"))

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

# Track selector (sticky via localStorage + URL param)
_sb_track_label = st.sidebar.selectbox(
    "Track",
    ["Combined", "Separate"],
    index=0 if st.session_state.get("track", TRACK_DEFAULT) == "combined" else 1,
    key="sidebar_track_label",
    help="Combined hides Separate-only topics/questions. Separate shows everything.",
)
_sb_track = "combined" if _sb_track_label == "Combined" else "separate"
if _sb_track != st.session_state.get("track", TRACK_DEFAULT):
    st.session_state["track"] = _sb_track
    _set_query_param(**{TRACK_PARAM: _sb_track})
_persist_track_to_browser(st.session_state.get("track", TRACK_DEFAULT))

with st.sidebar:
    if st.session_state.get("track", TRACK_DEFAULT) == "combined":
        if hasattr(st, "badge"):
            st.badge("COMBINED", color="orange")
        else:
            st.markdown(":orange-badge[COMBINED]")
    else:
        if hasattr(st, "badge"):
            st.badge("SEPARATE", color="primary")
        else:
            st.markdown(":blue-badge[SEPARATE]")
    st.caption("The badge shows whether COMBINED or SEPARATED Physics selected.")

header_left, header_mid, header_right = st.columns([3, 2, 1])

# Track badge (visual cue)
_track = st.session_state.get("track", TRACK_DEFAULT)

def _render_badge(label: str, *, color: str, icon: str | None = None):
    # st.badge exists in newer Streamlit. Fall back to the markdown badge directive if needed.
    if hasattr(st, "badge"):
        st.badge(label, color=color, icon=icon)
    else:
        # markdown directive supports only the basic palette (no 'primary')
        md_color = color if color in {"red","orange","yellow","blue","green","violet","gray","grey"} else "blue"
        st.markdown(f":{md_color}-badge[{label}]")

with header_right:
    if _track == "combined":
        _render_badge("COMBINED", color="orange", icon=":material/merge_type:")
    else:
        _render_badge("SEPARATE", color="primary", icon=":material/call_split:")

with header_left:

    st.title("⚛️ PanPhy Skill Builder")
    st.caption(f"Powered by OpenAI {MODEL_NAME}")
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

st.markdown(
    """
<div class="pp-hero">
  <div class="pp-hero-content">
    <div class="pp-hero-kicker">Personalised practice</div>
    <div class="pp-hero-title">Build confident GCSE physicists in minutes.</div>
    <div class="pp-hero-subtitle">
      Generate spec-aligned questions, capture handwritten working, and give instant feedback—
      all in one place.
    </div>
    <ul class="pp-hero-list">
      <li>AI practice questions tailored to topic and difficulty.</li>
      <li>Topic Journeys that guide students step-by-step.</li>
      <li>Teacher-uploaded questions with instant feedback and marks.</li>
    </ul>
  </div>
  <div class="pp-hero-card">
    <h4>Getting started</h4>
    <div class="pp-hero-metric">3 steps</div>
    <div>Pick a topic, choose a track, and start practicing.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# STUDENT / TEACHER PAGES
# ============================================================
_ui_helpers = {
    "CANVAS_BG_HEX": CANVAS_BG_HEX,
    "CANVAS_HEIGHT_DEFAULT": CANVAS_HEIGHT_DEFAULT,
    "CANVAS_HEIGHT_EXPANDED": CANVAS_HEIGHT_EXPANDED,
    "CANVAS_MAX_MB": CANVAS_MAX_MB,
    "RATE_LIMIT_MAX": RATE_LIMIT_MAX,
    "TEXTAREA_HEIGHT_DEFAULT": TEXTAREA_HEIGHT_DEFAULT,
    "TEXTAREA_HEIGHT_EXPANDED": TEXTAREA_HEIGHT_EXPANDED,
    "QUESTION_MAX_MB": QUESTION_MAX_MB,
    "MARKSCHEME_MAX_MB": MARKSCHEME_MAX_MB,
    "_check_rate_limit_db": _check_rate_limit_db,
    "_compress_bytes_to_limit": _compress_bytes_to_limit,
    "_effective_student_id": _effective_student_id,
    "_encode_image_bytes": _encode_image_bytes,
    "_run_ai_with_progress": _run_ai_with_progress,
    "_stylus_canvas_available": _stylus_canvas_available,
    "cached_download_from_storage": cached_download_from_storage,
    "canvas_has_ink": canvas_has_ink,
    "data_url_to_image_data": data_url_to_image_data,
    "db_ready": db_ready,
    "get_db_driver_type": get_db_driver_type,
    "ensure_attempts_table": ensure_attempts_table,
    "load_attempts_df": load_attempts_df,
    "delete_attempt_by_id": delete_attempt_by_id,
    "supabase_ready": supabase_ready,
    "get_gpt_feedback_from_bank": get_gpt_feedback_from_bank,
    "increment_rate_limit": increment_rate_limit,
    "insert_attempt": insert_attempt,
    "normalize_markdown_math": normalize_markdown_math,
    "preprocess_canvas_image": preprocess_canvas_image,
    "render_report": render_report,
    "render_md_box": render_md_box,
    "safe_bytes_to_pil": safe_bytes_to_pil,
    "slugify": slugify,
    "stylus_canvas": stylus_canvas,
    "upload_to_storage": upload_to_storage,
    "validate_image_file": validate_image_file,
    "bytes_to_pil": bytes_to_pil,
    "insert_question_bank_row": insert_question_bank_row,
}

if nav == "🧑‍🎓 Student":
    render_student_page(_ui_helpers)
elif nav in ("🔒 Teacher Dashboard", "📚 Question Bank"):
    render_teacher_page(nav, _ui_helpers)

st.markdown(
    """
<footer>
  &copy; <a href="https://panphy.github.io/" target="_blank" rel="noopener noreferrer">PanPhy</a> |
  <a href="https://buymeacoffee.com/panphy" target="_blank" rel="noopener noreferrer">Support My Projects</a>
</footer>
""",
    unsafe_allow_html=True,
)
