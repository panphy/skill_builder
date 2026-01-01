import streamlit as st
from pathlib import Path
from openai import OpenAI
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
from sqlalchemy import create_engine, text
import secrets as pysecrets

import logging
from logging.handlers import RotatingFileHandler
import os
import time
import traceback
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

st.markdown(
    """
<style>
div[data-testid="stButton"] button[aria-label="⤢"] span,
div[data-testid="stButton"] button[aria-label="⤡"] span {
  font-size: 22px;
  line-height: 1;
  font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# --- CONSTANTS ---
# =========================
MODEL_NAME = "gpt-5-mini"
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

# ============================================================
# SUBJECT CONTENT (topics + prompts)
# ============================================================
# For multi-subject scaling: keep the core app identical, and store subject-specific
# topic lists and AI prompt packs under: subjects/<subject_site>/.
#
# Recommended deployment pattern for separate subject sites:
#   - set SUBJECT_SITE in Streamlit Secrets (or environment variable)
#   - e.g. SUBJECT_SITE="physics"
SUBJECT_SITE = (st.secrets.get("SUBJECT_SITE") if hasattr(st, "secrets") else None) or os.getenv("SUBJECT_SITE", "physics")
SUBJECT_SITE = (SUBJECT_SITE or "physics").strip().lower()


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

@st.cache_data(show_spinner=False)
def _load_subject_pack(subject_site: str) -> dict:
    base = Path(__file__).resolve().parent
    subj_dir = base / "subjects" / subject_site

    topics_path = subj_dir / "topics.json"
    prompts_path = subj_dir / "prompts.json"
    settings_path = subj_dir / "settings.json"
    equations_path = subj_dir / "equations.json"

    if not topics_path.exists():
        raise FileNotFoundError(f"Missing topics file: {topics_path}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing prompts file: {prompts_path}")

    topics = json.loads(topics_path.read_text(encoding="utf-8"))
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    equations = json.loads(equations_path.read_text(encoding="utf-8")) if equations_path.exists() else {}

    return {"topics": topics, "prompts": prompts, "settings": settings, "equations": equations}

def _render_template(tpl: str, mapping: Dict[str, Any]) -> str:
    # Simple token replacement. Tokens look like: <<TOKEN_NAME>>
    out = str(tpl or "")
    for k, v in (mapping or {}).items():
        out = out.replace(f"<<{k}>>", str(v))
    return out

try:
    SUBJECT_PACK = _load_subject_pack(SUBJECT_SITE)
except Exception as _e:
    st.error(f"❌ Subject pack failed to load for SUBJECT_SITE='{SUBJECT_SITE}'.\n\n{type(_e).__name__}: {_e}")
    st.stop()

SUBJECT_SETTINGS = SUBJECT_PACK.get("settings", {}) or {}
SUBJECT_TOPICS_RAW = SUBJECT_PACK.get("topics", {}) or {}
SUBJECT_PROMPTS = SUBJECT_PACK.get("prompts", {}) or {}
SUBJECT_EQUATIONS = SUBJECT_PACK.get("equations", {}) or {}

# Topics for dropdowns (student + teacher)
TOPICS_CATALOG = SUBJECT_TOPICS_RAW.get("topics", [])

def get_topic_names_for_track(track: str) -> List[str]:
    track = (track or "").strip().lower()
    names: List[str] = []
    for t in TOPICS_CATALOG:
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        track_ok = str(t.get("track_ok", "both")).strip().lower() or "both"
        if track == "combined" and track_ok == "separate_only":
            continue
        names.append(name)
    return names



def get_topic_track_ok(topic_name: str) -> str:
    """Return track eligibility for a topic name in TOPICS_CATALOG: 'both' or 'separate_only'."""
    name_norm = (topic_name or "").strip().lower()
    if not name_norm:
        return "both"
    for t in TOPICS_CATALOG:
        nm = str(t.get("name", "")).strip().lower()
        if nm == name_norm:
            tok = str(t.get("track_ok", "both") or "both").strip().lower()
            return tok if tok in ("both", "separate_only") else "both"
    return "both"
# UI option lists (can be overridden per subject via settings.json)
QUESTION_TYPES = SUBJECT_SETTINGS.get("question_types") or ["Calculation", "Explanation", "Practical/Methods", "Graph/Analysis", "Mixed"]
DIFFICULTIES = SUBJECT_SETTINGS.get("difficulties") or ["Easy", "Medium", "Hard"]

# Prompt components (loaded from prompts.json)
GCSE_ONLY_GUARDRAILS = str(SUBJECT_PROMPTS.get("gcse_only_guardrails", "") or "").strip()
MARKDOWN_LATEX_RULES = str(SUBJECT_PROMPTS.get("markdown_latex_rules", "") or "").strip()


def _build_equation_guardrails(eq_pack: dict) -> str:
    """Build a compact, prompt-friendly guardrail block from subjects/<site>/equations.json."""
    if not isinstance(eq_pack, dict):
        return ""
    notes = eq_pack.get("notation_rules") or []
    eqs = eq_pack.get("key_equations") or []
    forb = eq_pack.get("forbidden_notes") or []
    out: List[str] = []
    if notes:
        out.append("EQUATION SHEET / NOTATION (from subject pack):")
        for line in notes[:12]:
            s = str(line).strip()
            if s:
                out.append(f"- {s}")
    if eqs:
        out.append("Key equations (use these forms and symbols):")
        for e in eqs[:18]:
            if isinstance(e, dict):
                name = str(e.get("name", "")).strip()
                latex = str(e.get("latex", "")).strip()
                if latex:
                    if name:
                        out.append(f"- {name}: {latex}")
                    else:
                        out.append(f"- {latex}")
            else:
                s = str(e).strip()
                if s:
                    out.append(f"- {s}")
    if forb:
        out.append("Explicitly NOT in AQA GCSE scope for this app:")
        for line in forb[:12]:
            s = str(line).strip()
            if s:
                out.append(f"- {s}")
    return "\n".join(out).strip()

EQUATION_GUARDRAILS = _build_equation_guardrails(SUBJECT_EQUATIONS)
if EQUATION_GUARDRAILS:
    GCSE_ONLY_GUARDRAILS = (GCSE_ONLY_GUARDRAILS + "\n\n" + EQUATION_GUARDRAILS).strip()


# Prompt templates
QGEN_SYSTEM_TPL = str(SUBJECT_PROMPTS.get("qgen_system", "") or "")
QGEN_USER_TPL = str(SUBJECT_PROMPTS.get("qgen_user", "") or "")
QGEN_REPAIR_PREFIX_TPL = str(SUBJECT_PROMPTS.get("qgen_repair_prefix", "") or "")

JOURNEY_SYSTEM_TPL = str(SUBJECT_PROMPTS.get("journey_system", "") or "")
JOURNEY_USER_TPL = str(SUBJECT_PROMPTS.get("journey_user", "") or "")
JOURNEY_REPAIR_PREFIX_TPL = str(SUBJECT_PROMPTS.get("journey_repair_prefix", "") or "")

FEEDBACK_SYSTEM_TPL = str(SUBJECT_PROMPTS.get("feedback_system", "") or "")

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


QUESTION_BANK_ALTER_DDL = f"""
alter table public.question_bank_v1
  add column if not exists question_type text default 'single';
alter table public.question_bank_v1
  add column if not exists journey_json jsonb;
alter table public.question_bank_v1
  add column if not exists subject_site text not null default '{SUBJECT_SITE}';
alter table public.question_bank_v1
  add column if not exists track_ok text not null default 'both';

drop index if exists public.uq_question_bank_source_assignment_label;
create unique index if not exists uq_question_bank_subject_source_assignment_label
  on public.question_bank_v1 (subject_site, source, assignment_name, question_label);

"""

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
_ss_init("journey_active_id", None)        # question_bank_v1.id of current journey
_ss_init("journey_json_cache", None)       # parsed journey JSON for current selection

# Topic Journey draft (teacher)
_ss_init("journey_draft", None)

_ss_init("journey_topics_selected", [])
_ss_init("journey_gen_error_details", None)
_ss_init("journey_show_error", False)


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


def ensure_question_bank_table():
    if st.session_state.get("bank_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return
    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, QUESTION_BANK_DDL)
            _exec_sql_many(conn, QUESTION_BANK_ALTER_DDL)
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
    - Convert \(...\) -> $...$ and \[...\] -> $$...$$
    - Wrap simple 'standalone' math tokens like 3^2, v^2, a_t, a_{t} in $...$
    """
    s = (md_text or "")
    if not s.strip():
        return ""

    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    protected: Dict[str, str] = {}
    s = _protect_segments(_MD_TOKEN_CODEBLOCK, s, protected, "CB")
    s = _protect_segments(_MD_TOKEN_MATHBLOCK, s, protected, "MB")
    s = _protect_segments(_MD_TOKEN_MATHINLINE, s, protected, "MI")
    s = _protect_segments(_MD_TOKEN_INLINECODE, s, protected, "IC")

    # Convert common LaTeX delimiters to MathJax-friendly ones
    s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", s, flags=re.DOTALL)

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
        <div class="pp-title">Working…</div>
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

    _render_overlay("AI is working. Please wait.", 0)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(task_fn)
            # Update the elapsed timer a few times per second.
            while not fut.done():
                elapsed = time.monotonic() - start_t
                _render_overlay("AI is working. Please wait.", _calc_percent(elapsed))
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


@st.cache_data(ttl=30)
def load_question_bank_df_cached(_fp: str, track: str, subject_site: str, limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
    """
    NOTE: This returns a *summary* table for listing/filtering, so we include
    light metadata + tags/question_text (for search). Full row is loaded by id.
    """
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_question_bank_table()
    track = (track or "").strip().lower()
    subject_site = (subject_site or "").strip().lower() or SUBJECT_SITE
    clauses = []
    if not include_inactive:
        clauses.append("is_active = true")
    clauses.append("subject_site = :subject_site")
    if track == "combined":
        clauses.append("track_ok = 'both'")
    where = ("where " + " and ".join(clauses)) if clauses else ""
    with eng.connect() as conn:
        df = pd.read_sql(
            text(f"""
                select
                  id, created_at, updated_at,
                  source, assignment_name, question_label,
                  max_marks, question_type, tags, question_text,
                  subject_site, track_ok, is_active
                from public.question_bank_v1
                {where}
                order by created_at desc
                limit :limit
            """),
            conn,
            params={"limit": int(limit), "subject_site": subject_site},
        )
    return df


def load_question_bank_df(limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
    fp = (st.secrets.get("DATABASE_URL", "") or "")[:40]
    try:
        return load_question_bank_df_cached(fp, track=st.session_state.get('track','combined'), subject_site=SUBJECT_SITE, limit=limit, include_inactive=include_inactive)
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
            text("select * from public.question_bank_v1 where id = :id and subject_site = :subject_site limit 1"),
            {"id": int(qid), "subject_site": SUBJECT_SITE}
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
    question_type: str = "single",
    journey_json: Optional[dict] = None,
    subject_site: Optional[str] = None,
    track_ok: str = "both",
) -> bool:
    eng = get_db_engine()
    if eng is None:
        return False
    ensure_question_bank_table()

    subject_site = (subject_site or "").strip().lower() or SUBJECT_SITE
    track_ok = (track_ok or "").strip().lower() or "both"
    if track_ok not in ("both", "separate_only"):
        track_ok = "both"

    qtype = (question_type or "single").strip().lower()
    if qtype not in ("single", "journey"):
        qtype = "single"

    query = """
    insert into public.question_bank_v1
      (source, created_by, subject_site, track_ok, assignment_name, question_label, max_marks, question_type, journey_json, tags,
       question_text, question_image_path,
       markscheme_text, markscheme_image_path,
       is_active, updated_at)
    values
      (:source, :created_by, :subject_site, :track_ok, :assignment_name, :question_label, :max_marks, :question_type, CAST(:journey_json AS jsonb),
       CAST(:tags AS jsonb),
       :question_text, :question_image_path,
       :markscheme_text, :markscheme_image_path,
       true, now())
    on conflict (subject_site, source, assignment_name, question_label) do update set
       created_by = excluded.created_by,
       subject_site = excluded.subject_site,
       track_ok = excluded.track_ok,
       max_marks = excluded.max_marks,
       question_type = excluded.question_type,
       journey_json = excluded.journey_json,
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
                "subject_site": subject_site,
                "track_ok": track_ok,
                "assignment_name": assignment_name.strip(),
                "question_label": question_label.strip(),
                "max_marks": int(max_marks),
                "question_type": qtype,
                "journey_json": json.dumps(journey_json or {}) if qtype == "journey" else None,
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


def delete_question_bank_by_id(qid: int) -> bool:
    eng = get_db_engine()
    if eng is None:
        return False
    ensure_question_bank_table()
    try:
        with eng.begin() as conn:
            res = conn.execute(
                text("delete from public.question_bank_v1 where id = :id and subject_site = :subject_site"),
                {"id": int(qid), "subject_site": SUBJECT_SITE},
            )
        st.session_state["db_last_error"] = ""
        try:
            load_question_bank_df_cached.clear()
            load_question_by_id_cached.clear()
        except Exception:
            pass
        return res.rowcount > 0
    except Exception as e:
        st.session_state["db_last_error"] = f"Delete Question Error: {type(e).__name__}: {e}"
        return False

# ============================================================
# MARKING (unified for question_bank_v1 rows)
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
# AI QUESTION GENERATOR (teacher-only, vet, then save)
# ============================================================
@st.cache_data(show_spinner=False)
def _get_equation_whitelist(eq_pack: dict) -> set:
    whitelist: set = set()
    if not isinstance(eq_pack, dict):
        return whitelist
    for eq in eq_pack.get("key_equations", []) or []:
        if isinstance(eq, dict):
            latex = str(eq.get("latex", "") or "").strip()
            if latex:
                whitelist.add(_normalize_equation_text(latex))
    notes = eq_pack.get("notation_rules") or []
    for note in notes:
        for match in _get_equation_regexes()["plain_eq"].finditer(str(note or "")):
            whitelist.add(_normalize_equation_text(match.group(1)))
    return whitelist


@st.cache_data(show_spinner=False)
def _get_equation_regexes() -> Dict[str, re.Pattern]:
    latex_block = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
    latex_inline = re.compile(r"\$(.+?)\$")
    latex_paren = re.compile(r"\\\((.+?)\\\)")
    latex_bracket = re.compile(r"\\\[(.+?)\\\]", flags=re.DOTALL)
    plain_eq = re.compile(
        r"([A-Za-z\\ΔλρθημΩπ][A-Za-z0-9\\ΔλρθημΩπ_{}^+\-*/(). ]{0,60}="
        r"[A-Za-z0-9\\ΔλρθημΩπ_{}^+\-*/(). ]{1,60})"
    )
    return {
        "latex_block": latex_block,
        "latex_inline": latex_inline,
        "latex_paren": latex_paren,
        "latex_bracket": latex_bracket,
        "plain_eq": plain_eq,
    }


def _normalize_equation_text(eq: str) -> str:
    s = str(eq or "").strip()
    if not s:
        return ""
    s = s.replace("−", "-").replace("–", "-")
    greek_map = {
        "Δ": "\\Delta",
        "λ": "\\lambda",
        "ρ": "\\rho",
        "θ": "\\theta",
        "μ": "\\mu",
    }
    for k, v in greek_map.items():
        s = s.replace(k, v)
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def _extract_equation_candidates(text: str) -> List[str]:
    if not text:
        return []
    regexes = _get_equation_regexes()
    candidates: List[str] = []
    for key in ("latex_block", "latex_inline", "latex_paren", "latex_bracket"):
        for match in regexes[key].finditer(text):
            cand = (match.group(1) or "").strip()
            if cand:
                candidates.append(cand)
    for line in (text or "").splitlines():
        for match in regexes["plain_eq"].finditer(line):
            cand = (match.group(1) or "").strip()
            if cand:
                candidates.append(cand)
    return candidates


def _find_non_whitelisted_equations(text: str) -> List[str]:
    whitelist = _get_equation_whitelist(SUBJECT_EQUATIONS)
    if not whitelist:
        return []
    candidates = _extract_equation_candidates(text)
    non_whitelisted: List[str] = []
    for cand in candidates:
        if re.match(r"^\s*TOTAL\s*=", cand, flags=re.IGNORECASE):
            continue
        norm = _normalize_equation_text(cand)
        if norm and norm not in whitelist:
            non_whitelisted.append(cand)
    if non_whitelisted:
        LOGGER.info(
            "Equation whitelist validation found non-whitelisted equations",
            extra={"ctx": {"component": "equation_whitelist", "count": len(non_whitelisted)}},
        )
    else:
        LOGGER.info(
            "Equation whitelist validation passed",
            extra={"ctx": {"component": "equation_whitelist", "count": 0}},
        )
    return non_whitelisted


def _self_check_equations(question_text: str, markscheme_text: str, subject_pack: dict) -> List[str]:
    eq_pack = subject_pack or {}
    key_eqs = eq_pack.get("key_equations") or []
    whitelist = [str(e.get("latex", "") or "").strip() for e in key_eqs if isinstance(e, dict)]
    notation = [str(n).strip() for n in (eq_pack.get("notation_rules") or []) if str(n).strip()]
    prompt = _render_template(
        """
You are checking GCSE Physics equations for compliance with the official equation sheet.
List every equation explicitly used in the question and mark scheme.
Then verify each equation appears on the official equation sheet list provided.

Return JSON: {"violations":[{"equation":"...","reason":"..."}]}

Official equations (LaTeX):
<<WHITELIST>>

Notation rules / canonical forms:
<<NOTATION>>

Question:
<<QUESTION>>

Mark scheme:
<<MARKSCHEME>>
""",
        {
            "WHITELIST": "\n".join([f"- {w}" for w in whitelist]) or "(none)",
            "NOTATION": "\n".join([f"- {n}" for n in notation]) or "(none)",
            "QUESTION": question_text or "",
            "MARKSCHEME": markscheme_text or "",
        },
    ).strip()

    try:
        response = client.with_options(timeout=10).chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = safe_parse_json(raw) or {}
        violations = data.get("violations", [])
        if isinstance(violations, list):
            out = []
            for v in violations:
                if isinstance(v, dict):
                    eq = str(v.get("equation", "") or "").strip()
                    reason = str(v.get("reason", "") or "").strip()
                    if eq:
                        out.append(f"{eq} ({reason})" if reason else eq)
                else:
                    s = str(v).strip()
                    if s:
                        out.append(s)
            LOGGER.info(
                "Equation self-check completed",
                extra={"ctx": {"component": "equation_self_check", "count": len(out)}},
            )
            return out
    except Exception as exc:
        LOGGER.warning(
            "Equation self-check failed; skipping",
            extra={"ctx": {"component": "equation_self_check", "error": type(exc).__name__}},
        )
    return []


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
            (r"\bcalculus\b|\bdifferentiat|\bintegrat", "Uses calculus (not GCSE)"),            (r"\bz\s*=\s*(\\Delta|Δ)\s*\\lambda\s*/\s*\\lambda|\bz\s*=\s*(delta|Δ)\s*lambda\s*/\s*lambda|Δ\s*λ\s*/\s*λ|Δ\s*lambda\s*/\s*lambda", "Uses red-shift calculation z=Δλ/λ (not required at AQA GCSE; red-shift is qualitative)"),
            (r"\\frac\{1\}\{2\}\s*k\s*x\s*\^\s*2|\b0\.5\s*k\s*x\s*\^\s*2\b|\b1\s*/\s*2\s*k\s*x\s*\^\s*2\b|\bk\s*x\s*\^\s*2\b", "Uses elastic potential energy with x: use AQA notation Ee = 1/2 k e^2 (extension e)"),
            (r"\bF\s*=\s*k\s*x\b|\bF\s*=\s*kx\b", "Uses Hooke’s law with x: use AQA notation F = k e (extension e)"),

        ]
        # Add subject-pack forbidden patterns (equations.json), if provided
        try:
            fps = SUBJECT_EQUATIONS.get("forbidden_patterns") or []
            for fp in fps:
                if isinstance(fp, dict):
                    rx = str(fp.get("regex", "") or "").strip()
                    rs = str(fp.get("reason", "") or "").strip()
                    if rx and rs:
                        patterns.append((rx, rs))
        except Exception:
            pass

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

        non_whitelisted = _find_non_whitelisted_equations(f"{qtxt}\n{mstxt}")
        if non_whitelisted:
            offending = ", ".join(sorted(set(non_whitelisted)))
            reasons.append(f"Contains non-whitelisted equations: {offending}")

        if "$" in qtxt and "\\(" in qtxt:
            reasons.append("Use $...$ for LaTeX, avoid \\(...\\).")

        # Subject-pack forbidden patterns (equations.json): reject out-of-scope content early
        try:
            t_all = json.dumps(d, ensure_ascii=False)
            fps = SUBJECT_EQUATIONS.get("forbidden_patterns") or []
            for fp in fps:
                if isinstance(fp, dict):
                    rx = str(fp.get("regex", "") or "").strip()
                    rs = str(fp.get("reason", "") or "").strip()
                    if rx and rs and re.search(rx, t_all, flags=re.IGNORECASE):
                        reasons.append(f"Journey contains forbidden content: {rs}")
        except Exception:
            pass

        return (len(reasons) == 0), reasons

    def _call_model(
        repair: bool,
        reasons: Optional[List[str]] = None,
        extra_hint: str | None = None,
    ) -> Dict[str, Any]:
        system = _render_template(QGEN_SYSTEM_TPL, {
            "GCSE_ONLY_GUARDRAILS": GCSE_ONLY_GUARDRAILS,
            "MARKDOWN_LATEX_RULES": MARKDOWN_LATEX_RULES,
            "TRACK": st.session_state.get("track", TRACK_DEFAULT),
        })
        system = (system or "").strip()

        base_user = _render_template(QGEN_USER_TPL, {
            "TOPIC": (topic_text or "").strip(),
            "DIFFICULTY": str(difficulty),
            "QTYPE": str(qtype),
            "MARKS": int(marks),
            "EXTRA_INSTRUCTIONS": (extra_instructions or "").strip() or "(none)",
            "TRACK": st.session_state.get("track", TRACK_DEFAULT),
        })
        base_user = (base_user or "").strip()

        if not repair:
            user = base_user
        else:
            bullet_reasons = "\n".join([f"- {r}" for r in (reasons or [])]) or "- (unspecified)"
            user = _render_template(QGEN_REPAIR_PREFIX_TPL, {
                "BULLET_REASONS": bullet_reasons,
                "MARKS": int(marks),
            })
            user = (user or "").strip() + "\n\n" + base_user


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
    self_check = _self_check_equations(
        str(data.get("question_text", "") or ""),
        str(data.get("markscheme_text", "") or ""),
        SUBJECT_EQUATIONS,
    )
    if self_check:
        reasons.append("Self-check found equation sheet violations: " + "; ".join(self_check))
        ok = False

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
# TOPIC JOURNEY GENERATOR (teacher-only)
# ============================================================
DURATION_TO_STEPS = {10: 5}
JOURNEY_CHECKPOINT_EVERY = 3

def generate_topic_journey_with_ai(
    topic_plain_english: str,
    duration_minutes: int,
) -> Dict[str, Any]:
    steps_n = DURATION_TO_STEPS.get(int(duration_minutes), 8)
    topic_plain_english = (topic_plain_english or "").strip()
    def _validate(d: Dict[str, Any]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if not isinstance(d, dict):
            return False, ["Output is not a JSON object."]
        if str(d.get("topic", "")).strip() == "":
            reasons.append("Missing topic.")
        steps = d.get("steps", [])
        if not isinstance(steps, list) or len(steps) != steps_n:
            reasons.append(f"steps must be a list of length {steps_n}.")
            return False, reasons

        for i, stp in enumerate(steps):
            if not isinstance(stp, dict):
                reasons.append(f"Step {i+1} is not an object.")
                continue
            if not str(stp.get("objective", "")).strip():
                reasons.append(f"Step {i+1}: missing objective.")
            if not str(stp.get("question_text", "")).strip():
                reasons.append(f"Step {i+1}: missing question_text.")
            if not str(stp.get("markscheme_text", "")).strip():
                reasons.append(f"Step {i+1}: missing markscheme_text.")
            try:
                mm = int(stp.get("max_marks", 0))
            except Exception:
                mm = 0
            if mm <= 0 or mm > 12:
                reasons.append(f"Step {i+1}: max_marks must be 1-12.")
            ms = str(stp.get("markscheme_text", "") or "")
            if f"TOTAL = {mm}" not in ms:
                reasons.append(f"Step {i+1}: markscheme_text must end with 'TOTAL = {mm}'.")
            non_whitelisted = _find_non_whitelisted_equations(
                f"{stp.get('question_text', '')}\n{stp.get('markscheme_text', '')}"
            )
            if non_whitelisted:
                offending = ", ".join(sorted(set(non_whitelisted)))
                reasons.append(f"Step {i+1}: contains non-whitelisted equations: {offending}")
        return (len(reasons) == 0), reasons

    def _call_model(repair: bool, reasons: Optional[List[str]] = None) -> Dict[str, Any]:
        system = _render_template(JOURNEY_SYSTEM_TPL, {
            "GCSE_ONLY_GUARDRAILS": GCSE_ONLY_GUARDRAILS,
            "MARKDOWN_LATEX_RULES": MARKDOWN_LATEX_RULES,
            "TRACK": st.session_state.get("track", TRACK_DEFAULT),
        })
        system = (system or "").strip()

        base_user = _render_template(JOURNEY_USER_TPL, {
            "TOPIC_PLAIN": (topic_plain_english or "").strip(),
            "DURATION_MIN": int(duration_minutes),
            "STEPS_N": int(steps_n),
                    })
        base_user = (base_user or "").strip()

        if not repair:
            user = base_user
        else:
            bullet_reasons = "\n".join([f"- {r}" for r in (reasons or [])]) or "- (unspecified)"
            user = _render_template(JOURNEY_REPAIR_PREFIX_TPL, {
                "BULLET_REASONS": bullet_reasons,
                "STEPS_N": int(steps_n),
            })
            user = (user or "").strip() + "\n\n" + base_user
            if extra_hint:
                user = user + "\n\nExtra constraints:\n" + extra_hint.strip()


        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return safe_parse_json(raw) or {}

    data = _call_model(repair=False)
    ok, reasons = _validate(data)
    steps_for_check = data.get("steps", []) if isinstance(data, dict) else []
    if isinstance(steps_for_check, list):
        for idx, stp in enumerate(steps_for_check):
            if not isinstance(stp, dict):
                continue
            self_check = _self_check_equations(
                str(stp.get("question_text", "") or ""),
                str(stp.get("markscheme_text", "") or ""),
                SUBJECT_EQUATIONS,
            )
            if self_check:
                reasons.append(
                    f"Step {idx+1}: self-check found equation sheet violations: " + "; ".join(self_check)
                )
                ok = False
    if not ok:
        data2 = _call_model(repair=True, reasons=reasons)
        ok2, reasons2 = _validate(data2)
        if ok2:
            data = data2
        else:
            if isinstance(data2, dict):
                steps_candidate = data2.get("steps", None)
                steps_list = steps_candidate if isinstance(steps_candidate, list) else []
            else:
                steps_list = []
            needs_steps = (
                "steps must be a list" in " ".join(reasons2).lower()
                or len(steps_list) == 0
            )
            if needs_steps:
                extra_hint = (
                    "Return a complete steps list of exactly the requested length. "
                    "Keep each step concise (1-2 sentences per question/markscheme) to avoid truncation."
                )
                data3 = _call_model(repair=True, reasons=reasons2, extra_hint=extra_hint)
                ok3, reasons3 = _validate(data3)
                if ok3:
                    data = data3
                else:
                    data = data3 if isinstance(data3, dict) and data3 else data2
                    data["warnings"] = reasons3[:12]
            else:
                data = data2 if isinstance(data2, dict) and data2 else data
                data["warnings"] = reasons2[:12]

    # Final clean-up / normalization (display-time normalization will still run)
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    steps = steps[:steps_n]
    for stp in steps:
        if isinstance(stp, dict):
            stp["objective"] = str(stp.get("objective", "") or "").strip()
            stp["question_text"] = str(stp.get("question_text", "") or "").strip()
            stp["markscheme_text"] = str(stp.get("markscheme_text", "") or "").strip()
            try:
                stp["max_marks"] = int(stp.get("max_marks", 1))
            except Exception:
                stp["max_marks"] = 1
            if not isinstance(stp.get("misconceptions", []), list):
                stp["misconceptions"] = []
            stp["misconceptions"] = [str(x).strip() for x in stp.get("misconceptions", []) if str(x).strip()][:6]
            if not isinstance(stp.get("spec_refs", []), list):
                stp["spec_refs"] = []
            stp["spec_refs"] = [str(x).strip() for x in stp.get("spec_refs", []) if str(x).strip()][:6]

    # If we still failed to get usable steps after repair, fail loudly so UI can show 'Explain error'.
    if not steps or len(steps) == 0:
        raise ValueError("AI returned invalid Journey JSON: missing a non-empty 'steps' list.")


    return {
        "topic": str(data.get("topic", "") or topic_plain_english).strip(),
        "duration_minutes": int(duration_minutes),
        "checkpoint_every": int(data.get("checkpoint_every", JOURNEY_CHECKPOINT_EVERY) or JOURNEY_CHECKPOINT_EVERY),
        "plan_markdown": str(data.get("plan_markdown", "") or "").strip(),
        "spec_alignment": [str(x).strip() for x in (data.get("spec_alignment", []) or []) if str(x).strip()][:20],
        "steps": steps,
        "warnings": [str(x) for x in (data.get("warnings", []) or [])][:12],
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
                            lambda r: f"{r['assignment_name']} | {r['question_label']} ({int(r['max_marks'])} marks) [{r.get('question_type','single')}] [id {int(r['id'])}]",
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

                                # Reset attempt state
                                st.session_state["feedback"] = None
                                st.session_state["canvas_key"] += 1
                                st.session_state["last_canvas_image_data"] = None  # legacy
                                st.session_state["last_canvas_image_data_single"] = None
                                st.session_state["last_canvas_data_url_single"] = None
                                st.session_state["last_canvas_image_data_journey"] = None
                                st.session_state["last_canvas_data_url_journey"] = None

                                # Reset Topic Journey state (if applicable)
                                st.session_state["journey_step_index"] = 0
                                st.session_state["journey_step_reports"] = []
                                st.session_state["journey_checkpoint_notes"] = {}
                                st.session_state["journey_active_id"] = int(chosen_id)
                                st.session_state["journey_json_cache"] = None
                                st.session_state["student_answer_text_single"] = ""
                                st.session_state["student_answer_text_journey"] = ""

    if st.session_state.get("cached_q_row"):
        _qr = st.session_state["cached_q_row"]
        st.caption(f"Selected: {_qr.get('assignment_name', '')} | {_qr.get('question_label', '')}")

    student_id = st.session_state.get("student_id", "") or ""
    q_row: Dict[str, Any] = st.session_state.get("cached_q_row") or {}
    question_img = st.session_state.get("cached_question_img")
    q_type = str(q_row.get("question_type", "single") or "single").strip().lower() if q_row else "single"

    q_key = None
    qid = None
    if q_row and q_row.get("id") is not None:
        try:
            qid = int(q_row["id"])
            q_key = f"QB:{qid}:{q_row.get('assignment_name','')}:{q_row.get('question_label','')}"
        except Exception:
            q_key = None

    # If journey, parse journey JSON once per selection
    journey_obj = None
    if q_row and q_type == "journey":
        if st.session_state.get("journey_json_cache") and st.session_state.get("journey_active_id") == qid:
            journey_obj = st.session_state.get("journey_json_cache")
        else:
            raw = q_row.get("journey_json")
            try:
                if isinstance(raw, str):
                    journey_obj = json.loads(raw) if raw.strip() else {}
                elif isinstance(raw, dict):
                    journey_obj = raw
                else:
                    journey_obj = {}
            except Exception:
                journey_obj = {}
            st.session_state["journey_json_cache"] = journey_obj

    col1, col2 = st.columns([5, 4])

        # -------------------------
    # LEFT: Question + Answer
    # -------------------------
    with col1:
        if not q_row:
            st.subheader("📝 Question")
            st.info("Select a question above to begin.")
        elif q_type != "journey":
            st.subheader("📝 Question")
            max_marks = int(q_row.get("max_marks", 1))
            q_text = (q_row.get("question_text") or "").strip()

            with st.container(border=True):
                if question_img is not None:
                    st.image(question_img, caption="Question image", use_container_width=True)
                if q_text:
                    st.markdown(normalize_markdown_math(q_text))
                if (question_img is None) and (not q_text):
                    st.warning("This question has no question text or image.")
            st.caption(f"Max Marks: {max_marks}")

            st.write("")
            st.markdown("**Answer in the box below.**")
            mode_row = st.columns([0.88, 0.12])
            with mode_row[0]:
                mode_single = st.radio(
                    "Answer mode",
                    ["⌨️ Type answer", "✍️ Write answer"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="answer_mode_single",
                )
            with mode_row[1]:
                if str(mode_single).startswith("⌨️"):
                    text_expanded = bool(st.session_state.get("text_expanded_single", False))
                    if st.button(
                        "⤡" if text_expanded else "⤢",
                        help=("Collapse working area" if text_expanded else "Expand working area"),
                        key="text_expand_btn_single",
                    ):
                        st.session_state["text_expanded_single"] = not text_expanded
                        text_expanded = not text_expanded
                else:
                    canvas_expanded = bool(st.session_state.get("canvas_expanded_single", False))
                    if st.button(
                        "⤡" if canvas_expanded else "⤢",
                        help=("Collapse working area" if canvas_expanded else "Expand working area"),
                        key="canvas_expand_btn_single",
                    ):
                        st.session_state["canvas_expanded_single"] = not canvas_expanded
                        canvas_expanded = not canvas_expanded

            if str(mode_single).startswith("⌨️"):
                text_height = TEXTAREA_HEIGHT_EXPANDED if text_expanded else TEXTAREA_HEIGHT_DEFAULT
                answer_single = st.text_area(
                    "Type your working:",
                    height=text_height,
                    placeholder="Enter your answer here...",
                    key="student_answer_text_single",
                )

                if st.button(
                    "Submit Text",
                    type="primary",
                    disabled=not AI_READY or not db_ready(),
                    key="submit_text_btn_single",
                ):
                    sid = _effective_student_id(student_id)

                    if not str(answer_single).strip():
                        st.toast("Please type an answer first.", icon="⚠️")
                    else:
                        try:
                            allowed_now, _, reset_str = _check_rate_limit_db(sid)
                        except Exception:
                            allowed_now, reset_str = True, ""
                        if not allowed_now:
                            st.error(
                                f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}."
                            )
                        else:
                            increment_rate_limit(sid)
                            st.session_state["text_expanded_single"] = False

                            def task():
                                ms_path = (st.session_state.get("cached_ms_path") or q_row.get("markscheme_image_path") or "").strip()
                                ms_img = None
                                if ms_path:
                                    fp = (st.secrets.get("SUPABASE_URL", "") or "")[:40]
                                    ms_bytes = cached_download_from_storage(ms_path, fp)
                                    ms_img = bytes_to_pil(ms_bytes) if ms_bytes else None
                                return get_gpt_feedback_from_bank(
                                    student_answer=answer_single,
                                    q_row=q_row,
                                    is_student_image=False,
                                    question_img=question_img,
                                    markscheme_img=ms_img,
                                )

                            st.session_state["feedback"] = _run_ai_with_progress(
                                task_fn=task,
                                ctx={"student_id": sid, "question": q_key or "", "mode": "text"},
                                typical_range="5-10 seconds",
                                est_seconds=9.0,
                            )

                            if db_ready() and q_key:
                                insert_attempt(
                                    student_id,
                                    q_key,
                                    st.session_state["feedback"],
                                    mode="text",
                                    question_bank_id=qid,
                                )
            else:
                canvas_height = CANVAS_HEIGHT_EXPANDED if canvas_expanded else CANVAS_HEIGHT_DEFAULT
                canvas_storage_key = (
                    f"panphy_canvas_h_{SUBJECT_SITE}_single_expanded"
                    if canvas_expanded
                    else f"panphy_canvas_h_{SUBJECT_SITE}_single"
                )
                if _stylus_canvas_available():
                    tool_row = st.columns([2.2, 1.4, 1, 1])
                    with tool_row[0]:
                        tool = st.radio(
                            "Tool",
                            ["Pen", "Eraser"],
                            horizontal=True,
                            label_visibility="collapsed",
                            key="canvas_tool_single",
                        )
                    with tool_row[1]:
                        st.checkbox(
                            "Stylus-only",
                            help="Best on iPad. When enabled, finger/palm touches are ignored.",
                            key="stylus_only_enabled",
                        )
                    undo_clicked = tool_row[2].button("↩️ Undo", use_container_width=True, key="canvas_undo_single")
                    clear_clicked = tool_row[3].button("🗑️ Clear", use_container_width=True, key="canvas_clear_single")
                    cmd = None
                    if undo_clicked:
                        st.session_state["feedback"] = None
                        st.session_state["canvas_cmd_nonce_single"] = int(st.session_state.get("canvas_cmd_nonce_single", 0) or 0) + 1
                        cmd = "undo"
                    if clear_clicked:
                        st.session_state["feedback"] = None
                        st.session_state["last_canvas_data_url_single"] = None
                        st.session_state["last_canvas_image_data_single"] = None
                        st.session_state["canvas_cmd_nonce_single"] = int(st.session_state.get("canvas_cmd_nonce_single", 0) or 0) + 1
                        st.session_state["canvas_key"] = int(st.session_state.get("canvas_key", 0) or 0) + 1
                        cmd = "clear"

                    stroke_width = 2 if tool == "Pen" else 30
                    stroke_color = "#000000" if tool == "Pen" else CANVAS_BG_HEX
                    canvas_value = stylus_canvas(
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_color=CANVAS_BG_HEX,
                        height=canvas_height,
                        width=None,
                        storage_key=canvas_storage_key,
                        initial_data_url=st.session_state.get("last_canvas_data_url_single"),
                        pen_only=bool(st.session_state.get("stylus_only_enabled", True)),
                        tool=("pen" if tool == "Pen" else "eraser"),
                        command=cmd,
                        command_nonce=int(st.session_state.get("canvas_cmd_nonce_single", 0) or 0),
                        key=f"stylus_canvas_single_{qid or 'none'}_{st.session_state['canvas_key']}",
                    )
                    if isinstance(canvas_value, dict) and (not canvas_value.get("is_empty")) and canvas_value.get("data_url"):
                        st.session_state["last_canvas_data_url_single"] = canvas_value.get("data_url")
                else:
                    tool_row = st.columns([2, 1])
                    with tool_row[0]:
                        tool = st.radio(
                            "Tool",
                            ["Pen", "Eraser"],
                            horizontal=True,
                            label_visibility="collapsed",
                            key="canvas_tool_single",
                        )
                    clear_clicked = tool_row[1].button("🗑️ Clear", use_container_width=True, key="canvas_clear_single")
                    if clear_clicked:
                        st.session_state["feedback"] = None
                        st.session_state["last_canvas_image_data_single"] = None
                        st.session_state["last_canvas_data_url_single"] = None
                        st.session_state["canvas_key"] = int(st.session_state.get("canvas_key", 0) or 0) + 1
                        st.rerun()
                    stroke_width = 2 if tool == "Pen" else 30
                    stroke_color = "#000000" if tool == "Pen" else CANVAS_BG_HEX
                    try:
                        from streamlit_drawable_canvas import st_canvas as _st_canvas
                    except Exception:
                        _st_canvas = None
                    if _st_canvas is None:
                        st.warning("Canvas unavailable. Add components folder or install streamlit-drawable-canvas.")
                        canvas_result = None
                    else:
                        canvas_result = _st_canvas(
                            stroke_width=stroke_width,
                            stroke_color=stroke_color,
                            background_color=CANVAS_BG_HEX,
                            height=canvas_height,
                            width=600,
                            drawing_mode="freedraw",
                            key=f"canvas_single_{st.session_state['canvas_key']}",
                            display_toolbar=False,
                            update_streamlit=True,
                        )
                        if canvas_result is not None and getattr(canvas_result, "image_data", None) is not None:
                            if canvas_has_ink(canvas_result.image_data):
                                st.session_state["last_canvas_image_data_single"] = canvas_result.image_data

                submitted_writing = st.button(
                    "Submit Writing",
                    type="primary",
                    disabled=not AI_READY or not db_ready(),
                    key="submit_writing_btn_single",
                )

                if submitted_writing:
                    sid = _effective_student_id(student_id)

                    img_data = None
                    if _stylus_canvas_available():
                        data_url = None
                        try:
                            data_url = (canvas_value or {}).get("data_url") if isinstance(canvas_value, dict) else None
                        except Exception:
                            data_url = None
                        if not data_url:
                            data_url = st.session_state.get("last_canvas_data_url_single")
                        if data_url:
                            try:
                                img_data = data_url_to_image_data(data_url)
                                # Keep legacy cache in sync
                                st.session_state["last_canvas_image_data_single"] = img_data
                            except Exception:
                                img_data = None
                    else:
                        if canvas_result is not None and getattr(canvas_result, "image_data", None) is not None:
                            img_data = canvas_result.image_data
                        if img_data is None:
                            img_data = st.session_state.get("last_canvas_image_data_single")
                    if img_data is None:
                        img_data = st.session_state.get("last_canvas_image_data_single")

                    if img_data is None or (not canvas_has_ink(img_data)):
                        st.toast("Canvas is blank. Write your answer first, then press Submit.", icon="⚠️")
                        st.stop()

                    try:
                        allowed_now, _, reset_str = _check_rate_limit_db(sid)
                    except Exception:
                        allowed_now, reset_str = True, ""
                    if not allowed_now:
                        st.error(
                            f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}."
                        )
                        st.stop()

                    st.session_state["canvas_expanded_single"] = False
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
                            markscheme_img=ms_img,
                        )

                    st.session_state["feedback"] = _run_ai_with_progress(
                        task_fn=task,
                        ctx={"student_id": sid, "question": q_key or "", "mode": "writing"},
                        typical_range="8-15 seconds",
                        est_seconds=13.0,
                    )

                    if db_ready() and q_key:
                        insert_attempt(
                            student_id,
                            q_key,
                            st.session_state["feedback"],
                            mode="writing",
                            question_bank_id=qid,
                        )


        else:
            # -------------------------
            # Topic Journey student flow (one step at a time)
            # -------------------------
            st.subheader("🧭 Topic Journey")
            if not isinstance(journey_obj, dict) or not journey_obj.get("steps"):
                st.error("This Topic Journey is missing its steps JSON.")
            else:
                steps = journey_obj.get("steps", [])
                if not isinstance(steps, list) or not steps:
                    st.error("This Topic Journey has no steps.")
                else:
                    step_i = int(st.session_state.get("journey_step_index", 0) or 0)
                    step_i = max(0, min(step_i, len(steps) - 1))
                    st.session_state["journey_step_index"] = step_i

                    checkpoint_every = int(
                        journey_obj.get("checkpoint_every", JOURNEY_CHECKPOINT_EVERY) or JOURNEY_CHECKPOINT_EVERY
                    )

                    st.caption(f"Step {step_i + 1} of {len(steps)}")
                    step = steps[step_i] if step_i < len(steps) else {}

                    obj = str(step.get("objective", "") or "").strip()
                    qtxt = str(step.get("question_text", "") or "").strip()
                    mm = clamp_int(step.get("max_marks", 1), 1, 50, default=1)

                    with st.container(border=True):
                        if obj:
                            st.markdown(normalize_markdown_math(f"**Objective:** {obj}"))
                        if qtxt:
                            st.markdown(normalize_markdown_math(qtxt))
                        else:
                            st.warning("This step has no question text.")

                    st.caption(f"Max Marks (this step): {mm}")

                    # Build a step-level q_row compatible with the marker
                    step_q_row = {
                        "max_marks": int(mm),
                        "question_text": qtxt,
                        "markscheme_text": str(step.get("markscheme_text", "") or "").strip(),
                    }

                    def _update_checkpoint_notes(reports_list, idx, total_steps):
                        # Deterministic checkpoint summary (no extra AI calls)
                        if not isinstance(reports_list, list):
                            return
                        if not (((idx + 1) % checkpoint_every == 0) or (idx == total_steps - 1)):
                            return

                        last_reports = [
                            r
                            for r in reports_list[max(0, idx - (checkpoint_every - 1)) : idx + 1]
                            if isinstance(r, dict)
                        ]
                        mastered, improve = [], []
                        for r in last_reports:
                            if int(r.get("marks_awarded", 0)) >= int(r.get("max_marks", 1)):
                                mastered.extend((r.get("feedback_points", []) or [])[:2])
                            else:
                                improve.extend((r.get("next_steps", []) or [])[:3])

                        mastered = [m for m in mastered if str(m).strip()][:6]
                        improve = [n for n in improve if str(n).strip()][:6]

                        md = "### Checkpoint\n"
                        md += "**Mastered (recent):**\n"
                        md += "\n".join([f"- {x}" for x in mastered]) if mastered else "- (keep going - you're building foundations)"
                        md += "\n\n**Next improvements:**\n"
                        md += "\n".join([f"- {x}" for x in improve]) if improve else "- (no major issues detected in recent steps)"

                        notes = st.session_state.get("journey_checkpoint_notes", {}) or {}
                        if not isinstance(notes, dict):
                            notes = {}
                        notes[str(idx)] = md
                        st.session_state["journey_checkpoint_notes"] = notes

                    st.write("")
                    st.markdown("**Answer in the box below.**")
                    mode_row = st.columns([0.88, 0.12])
                    with mode_row[0]:
                        mode_journey = st.radio(
                            "Answer mode",
                            ["⌨️ Type answer", "✍️ Write answer"],
                            horizontal=True,
                            label_visibility="collapsed",
                            key="answer_mode_journey",
                        )
                    with mode_row[1]:
                        if str(mode_journey).startswith("⌨️"):
                            text_expanded = bool(st.session_state.get("text_expanded_journey", False))
                            if st.button(
                                "⤡" if text_expanded else "⤢",
                                help=("Collapse working area" if text_expanded else "Expand working area"),
                                key="text_expand_btn_journey",
                            ):
                                st.session_state["text_expanded_journey"] = not text_expanded
                                text_expanded = not text_expanded
                        else:
                            canvas_expanded = bool(st.session_state.get("canvas_expanded_journey", False))
                            if st.button(
                                "⤡" if canvas_expanded else "⤢",
                                help=("Collapse working area" if canvas_expanded else "Expand working area"),
                                key="canvas_expand_btn_journey",
                            ):
                                st.session_state["canvas_expanded_journey"] = not canvas_expanded
                                canvas_expanded = not canvas_expanded

                    if str(mode_journey).startswith("⌨️"):
                        text_height = TEXTAREA_HEIGHT_EXPANDED if text_expanded else TEXTAREA_HEIGHT_DEFAULT
                        answer_journey = st.text_area(
                            "Type your working:",
                            height=text_height,
                            placeholder="Enter your answer here...",
                            key="student_answer_text_journey",
                        )

                        if st.button(
                            "Submit Text",
                            type="primary",
                            disabled=not AI_READY or not db_ready(),
                            key="submit_text_btn_journey",
                        ):
                            sid = _effective_student_id(student_id)

                            if not str(answer_journey).strip():
                                st.toast("Please type an answer first.", icon="⚠️")
                            else:
                                try:
                                    allowed_now, _, reset_str = _check_rate_limit_db(sid)
                                except Exception:
                                    allowed_now, reset_str = True, ""
                                if not allowed_now:
                                    st.error(
                                        f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}."
                                    )
                                else:
                                    increment_rate_limit(sid)
                                    st.session_state["text_expanded_journey"] = False

                                    def task():
                                        return get_gpt_feedback_from_bank(
                                            student_answer=answer_journey,
                                            q_row=step_q_row,
                                            is_student_image=False,
                                            question_img=None,
                                            markscheme_img=None,
                                        )

                                    st.session_state["feedback"] = _run_ai_with_progress(
                                        task_fn=task,
                                        ctx={"student_id": sid, "question": q_key or "", "mode": f"journey_text_s{step_i}"},
                                        typical_range="5-10 seconds",
                                        est_seconds=9.0,
                                    )

                                    # Store step report history
                                    reports = st.session_state.get("journey_step_reports", [])
                                    if not isinstance(reports, list):
                                        reports = []
                                    while len(reports) <= step_i:
                                        reports.append(None)
                                    reports[step_i] = st.session_state["feedback"]
                                    st.session_state["journey_step_reports"] = reports

                                    _update_checkpoint_notes(reports, step_i, len(steps))

                                    if db_ready() and q_key:
                                        insert_attempt(
                                            student_id,
                                            q_key,
                                            st.session_state["feedback"],
                                            mode="journey_text",
                                            question_bank_id=qid,
                                            step_index=step_i,
                                        )
                    else:
                        canvas_height = CANVAS_HEIGHT_EXPANDED if canvas_expanded else CANVAS_HEIGHT_DEFAULT
                        canvas_storage_key = (
                            f"panphy_canvas_h_{SUBJECT_SITE}_journey_expanded"
                            if canvas_expanded
                            else f"panphy_canvas_h_{SUBJECT_SITE}_journey"
                        )
                        if _stylus_canvas_available():
                            tool_row = st.columns([2.2, 1.4, 1, 1])
                            with tool_row[0]:
                                tool = st.radio(
                                    "Tool",
                                    ["Pen", "Eraser"],
                                    horizontal=True,
                                    label_visibility="collapsed",
                                    key="canvas_tool_journey",
                                )
                            with tool_row[1]:
                                st.checkbox(
                                    "Stylus-only",
                                    help="Best on iPad. When enabled, finger/palm touches are ignored.",
                                    key="stylus_only_enabled",
                                )
                            undo_clicked = tool_row[2].button("↩️ Undo", use_container_width=True, key="canvas_undo_journey")
                            clear_clicked = tool_row[3].button("🗑️ Clear", use_container_width=True, key="canvas_clear_journey")
                            cmd = None
                            if undo_clicked:
                                st.session_state["feedback"] = None
                                st.session_state["canvas_cmd_nonce_journey"] = int(st.session_state.get("canvas_cmd_nonce_journey", 0) or 0) + 1
                                cmd = "undo"
                            if clear_clicked:
                                st.session_state["feedback"] = None
                                st.session_state["last_canvas_data_url_journey"] = None
                                st.session_state["last_canvas_image_data_journey"] = None
                                st.session_state["canvas_cmd_nonce_journey"] = int(st.session_state.get("canvas_cmd_nonce_journey", 0) or 0) + 1
                                st.session_state["canvas_key"] = int(st.session_state.get("canvas_key", 0) or 0) + 1
                                cmd = "clear"

                            stroke_width = 2 if tool == "Pen" else 30
                            stroke_color = "#000000" if tool == "Pen" else CANVAS_BG_HEX
                            canvas_value = stylus_canvas(
                                stroke_width=stroke_width,
                                stroke_color=stroke_color,
                                background_color=CANVAS_BG_HEX,
                                height=canvas_height,
                                width=None,
                                storage_key=canvas_storage_key,
                                initial_data_url=st.session_state.get("last_canvas_data_url_journey"),
                                pen_only=bool(st.session_state.get("stylus_only_enabled", True)),
                                tool=("pen" if tool == "Pen" else "eraser"),
                                command=cmd,
                                command_nonce=int(st.session_state.get("canvas_cmd_nonce_journey", 0) or 0),
                                key=f"stylus_canvas_journey_{qid or 'none'}_{step_i}_{st.session_state['canvas_key']}",
                            )
                            if isinstance(canvas_value, dict) and (not canvas_value.get("is_empty")) and canvas_value.get("data_url"):
                                st.session_state["last_canvas_data_url_journey"] = canvas_value.get("data_url")
                        else:
                            tool_row = st.columns([2, 1])
                            with tool_row[0]:
                                tool = st.radio(
                                    "Tool",
                                    ["Pen", "Eraser"],
                                    horizontal=True,
                                    label_visibility="collapsed",
                                    key="canvas_tool_journey",
                                )
                            clear_clicked = tool_row[1].button("🗑️ Clear", use_container_width=True, key="canvas_clear_journey")
                            if clear_clicked:
                                st.session_state["feedback"] = None
                                st.session_state["last_canvas_image_data_journey"] = None
                                st.session_state["canvas_key"] = int(st.session_state.get("canvas_key", 0) or 0) + 1
                                st.rerun()
                            stroke_width = 2 if tool == "Pen" else 30
                            stroke_color = "#000000" if tool == "Pen" else CANVAS_BG_HEX
                            try:
                                from streamlit_drawable_canvas import st_canvas as _st_canvas
                            except Exception:
                                _st_canvas = None
                            if _st_canvas is None:
                                st.warning("Canvas unavailable. Add components folder or install streamlit-drawable-canvas.")
                                canvas_result = None
                            else:
                                canvas_result = _st_canvas(
                                    stroke_width=stroke_width,
                                    stroke_color=stroke_color,
                                    background_color=CANVAS_BG_HEX,
                                    height=canvas_height,
                                    width=600,
                                    drawing_mode="freedraw",
                                    key=f"canvas_journey_{st.session_state['canvas_key']}",
                                    display_toolbar=False,
                                    update_streamlit=True,
                                )
                                if canvas_result is not None and getattr(canvas_result, "image_data", None) is not None:
                                    if canvas_has_ink(canvas_result.image_data):
                                        st.session_state["last_canvas_image_data_journey"] = canvas_result.image_data

                        submitted_writing = st.button(
                            "Submit Writing",
                            type="primary",
                            disabled=not AI_READY or not db_ready(),
                            key="submit_writing_btn_journey",
                        )

                        if submitted_writing:
                            sid = _effective_student_id(student_id)

                            img_data = None
                            if _stylus_canvas_available():
                                data_url = None
                                try:
                                    data_url = (canvas_value or {}).get("data_url") if isinstance(canvas_value, dict) else None
                                except Exception:
                                    data_url = None
                                if not data_url:
                                    data_url = st.session_state.get("last_canvas_data_url_journey")
                                if data_url:
                                    try:
                                        img_data = data_url_to_image_data(data_url)
                                        # Keep legacy cache in sync
                                        st.session_state["last_canvas_image_data_journey"] = img_data
                                    except Exception:
                                        img_data = None
                            else:
                                if canvas_result is not None and getattr(canvas_result, "image_data", None) is not None:
                                    img_data = canvas_result.image_data
                                if img_data is None:
                                    img_data = st.session_state.get("last_canvas_image_data_journey")
                            if img_data is None:
                                img_data = st.session_state.get("last_canvas_image_data_journey")

                            if img_data is None or (not canvas_has_ink(img_data)):
                                st.toast("Canvas is blank. Write your answer first, then press Submit.", icon="⚠️")
                                st.stop()

                            try:
                                allowed_now, _, reset_str = _check_rate_limit_db(sid)
                            except Exception:
                                allowed_now, reset_str = True, ""
                            if not allowed_now:
                                st.error(
                                    f"You’ve reached the limit of {RATE_LIMIT_MAX} submissions per hour. Please try again at {reset_str}."
                                )
                                st.stop()

                            st.session_state["canvas_expanded_journey"] = False
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
                                return get_gpt_feedback_from_bank(
                                    student_answer=img_for_ai,
                                    q_row=step_q_row,
                                    is_student_image=True,
                                    question_img=None,
                                    markscheme_img=None,
                                )

                            st.session_state["feedback"] = _run_ai_with_progress(
                                task_fn=task,
                                ctx={"student_id": sid, "question": q_key or "", "mode": f"journey_writing_s{step_i}"},
                                typical_range="8-15 seconds",
                                est_seconds=13.0,
                            )

                            # Store step report history
                            reports = st.session_state.get("journey_step_reports", [])
                            if not isinstance(reports, list):
                                reports = []
                            while len(reports) <= step_i:
                                reports.append(None)
                            reports[step_i] = st.session_state["feedback"]
                            st.session_state["journey_step_reports"] = reports

                            _update_checkpoint_notes(reports, step_i, len(steps))

                            if db_ready() and q_key:
                                insert_attempt(
                                    student_id,
                                    q_key,
                                    st.session_state["feedback"],
                                    mode="journey_writing",
                                    question_bank_id=qid,
                                    step_index=step_i,
                                )


    # -------------------------
    # RIGHT: Feedback
    # -------------------------
    with col2:
        st.subheader("👨‍🏫 Feedback")
        with st.container(border=True):
            if st.session_state.get("feedback"):
                render_report(st.session_state["feedback"])

                # Journey checkpoint notes (if any)
                if q_row and q_type == "journey":
                    step_i = int(st.session_state.get("journey_step_index", 0) or 0)
                    notes = st.session_state.get("journey_checkpoint_notes", {}) or {}
                    note_md = notes.get(str(step_i))
                    if note_md:
                        st.divider()
                        st.markdown(normalize_markdown_math(note_md))

                st.divider()

                if q_row and q_type == "journey":
                    step_i = int(st.session_state.get("journey_step_index", 0) or 0)
                    steps = (journey_obj or {}).get("steps", [])
                    total_steps = len(steps) if isinstance(steps, list) else 0

                    def _reset_answer_inputs_for_step():
                        st.session_state["last_canvas_image_data"] = None  # legacy
                        st.session_state["last_canvas_image_data_journey"] = None
                        st.session_state["canvas_key"] = int(st.session_state.get("canvas_key", 0) or 0) + 1
                        st.session_state["student_answer_text_journey"] = ""

                    def _journey_redo_cb():
                        st.session_state["feedback"] = None
                        _reset_answer_inputs_for_step()

                    def _journey_next_cb(step_i: int, total_steps: int):
                        st.session_state["feedback"] = None
                        _reset_answer_inputs_for_step()
                        st.session_state["journey_step_index"] = min(step_i + 1, max(0, total_steps - 1))

                    cbtn1, cbtn2 = st.columns(2)
                    with cbtn1:
                        st.button("Redo this step", use_container_width=True, key="journey_redo", on_click=_journey_redo_cb)
                    with cbtn2:
                        next_disabled = (total_steps <= 0) or (step_i >= total_steps - 1)
                        label = "Finish" if next_disabled else "Next step"
                        st.button(
                            label,
                            use_container_width=True,
                            disabled=next_disabled,
                            key="journey_next",
                            on_click=_journey_next_cb,
                            args=(step_i, total_steps),
                        )

                    if total_steps > 0 and step_i >= total_steps - 1:
                        st.success("Journey complete! You can redo the final step, or choose a new assignment.")
                else:
                    def _new_attempt_cb():
                        st.session_state["feedback"] = None
                        st.session_state["last_canvas_image_data"] = None  # legacy
                        st.session_state["last_canvas_image_data_single"] = None
                        st.session_state["canvas_key"] = int(st.session_state.get("canvas_key", 0) or 0) + 1
                        st.session_state["student_answer_text_single"] = ""

                    st.button("Start New Attempt", use_container_width=True, key="new_attempt", on_click=_new_attempt_cb)
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

                attempt_delete_open = bool(st.session_state.get("attempt_delete_picks")) or bool(
                    st.session_state.get("confirm_delete_attempt")
                )
                with st.expander("Delete attempts", expanded=attempt_delete_open):
                    df_del = df.head(200).copy()
                    df_del = df_del[df_del["id"].notna()]
                    if df_del.empty:
                        st.info("No attempts available for deletion.")
                    else:
                        def _request_attempt_delete():
                            st.session_state["attempt_delete_requested"] = True

                        def _fmt_attempt(r):
                            created_at = str(r.get("created_at", "") or "")
                            student_id = str(r.get("student_id", "") or "")
                            question_key = str(r.get("question_key", "") or "")
                            mode = str(r.get("mode", "") or "")
                            try:
                                marks = f"{int(r.get('marks_awarded', 0))}/{int(r.get('max_marks', 0))}"
                            except Exception:
                                marks = ""
                            try:
                                aid = int(r.get("id"))
                            except Exception:
                                aid = -1
                            return f"{created_at} | {student_id} | {question_key} | {mode} | {marks} [id {aid}]"

                        df_del["label"] = df_del.apply(_fmt_attempt, axis=1)
                        delete_status = None
                        if st.session_state.get("attempt_delete_requested"):
                            attempt_picks = st.session_state.get("attempt_delete_picks", [])
                            confirm_delete = st.session_state.get("confirm_delete_attempt", False)
                            if confirm_delete and attempt_picks:
                                delete_ok = True
                                for attempt_pick in attempt_picks:
                                    attempt_id = int(df_del.loc[df_del["label"] == attempt_pick, "id"].iloc[0])
                                    delete_ok = delete_attempt_by_id(attempt_id) and delete_ok
                                if delete_ok:
                                    delete_status = "success"
                                    st.session_state["attempt_delete_picks"] = []
                                    st.session_state["confirm_delete_attempt"] = False
                                else:
                                    delete_status = "failed"
                            else:
                                delete_status = "missing"
                            st.session_state["attempt_delete_requested"] = False

                        attempt_picks = st.multiselect(
                            "Select attempts to delete",
                            df_del["label"].tolist(),
                            key="attempt_delete_picks",
                        )
                        confirm_delete = st.checkbox(
                            "I understand this will permanently delete the selected attempts.",
                            key="confirm_delete_attempt",
                        )
                        if st.button(
                            "Delete selected attempts",
                            type="primary",
                            use_container_width=True,
                            disabled=not (confirm_delete and attempt_picks),
                            key="delete_attempt_btn",
                            on_click=_request_attempt_delete,
                        ):
                            pass
                        if delete_status == "success":
                            st.success("Attempt(s) deleted.")
                            st.rerun()
                        elif delete_status == "failed":
                            st.error("Delete failed. Check database errors above.")
                        elif delete_status == "missing":
                            st.warning("Select attempts and confirm deletion to proceed.")
        else:
            st.caption("Enter the teacher password to view analytics.")

# ============================================================
# QUESTION BANK PAGE
# ============================================================
else:
    st.divider()
    st.subheader("📚 Question Bank")

    # Default track eligibility tag for any question you SAVE (AI drafts, edited, uploads).
    _tt_label = st.selectbox(
        "Track eligibility for saved items",
        ["Both (Combined + Separate)", "Separate only"],
        index=0,
        key="teacher_track_ok_label",
        help="If set to 'Separate only', Combined students will NOT see the item. Use this for separate-only content.",
    )
    st.session_state["teacher_track_ok"] = "both" if _tt_label.startswith("Both") else "separate_only"


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

                    f1, f2, f3 = st.columns([2, 2, 3])
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
                            qtype = str(r.get("question_type") or "single").strip().lower()
                            try:
                                mk = int(r.get("max_marks") or 0)
                            except Exception:
                                mk = 0
                            try:
                                qid = int(r.get("id"))
                            except Exception:
                                qid = -1
                            tag = "JOURNEY" if qtype == "journey" else "SINGLE"
                            return f"{asg} | {ql} ({mk} marks) [{src}] [{tag}] [id {qid}]"

                        df_f["label"] = df_f.apply(_fmt_label, axis=1)
                        options = df_f["label"].tolist()

                        if "bank_preview_pick" in st.session_state and st.session_state["bank_preview_pick"] not in options:
                            st.session_state["bank_preview_pick"] = options[0]

                        pick = st.selectbox("Select an entry to preview", options, key="bank_preview_pick")
                        pick_id = int(df_f.loc[df_f["label"] == pick, "id"].iloc[0])

                        row = load_question_by_id(pick_id) or {}
                        q_text = (row.get("question_text") or "").strip()
                        ms_text = (row.get("markscheme_text") or "").strip()
                        q_type = str(row.get("question_type") or "single").strip().lower()

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

                        if q_type == "journey":
                            # Journey preview
                            rawj = row.get("journey_json")
                            try:
                                if isinstance(rawj, str):
                                    journey = json.loads(rawj) if rawj.strip() else {}
                                elif isinstance(rawj, dict):
                                    journey = rawj
                                else:
                                    journey = {}
                            except Exception:
                                journey = {}

                            plan_md = (journey.get("plan_markdown") or q_text or "").strip()
                            steps = journey.get("steps", [])
                            if not isinstance(steps, list):
                                steps = []

                            with pv1:
                                st.markdown("**Topic Journey plan**")
                                with st.container(border=True):
                                    if plan_md:
                                        st.markdown(normalize_markdown_math(plan_md))
                                    else:
                                        st.caption("No plan text.")
                                st.caption(f"Steps: {len(steps)}")
                                for i, stp in enumerate(steps[:50]):
                                    if not isinstance(stp, dict):
                                        continue
                                    title = str(stp.get("objective") or "").strip() or "Step"
                                    with st.expander(f"Step {i+1}: {title[:80]}", expanded=(i == 0)):
                                        st.markdown(normalize_markdown_math(str(stp.get("question_text", "") or "")))

                            with pv2:
                                st.markdown("**Mark schemes (teacher only)**")
                                for i, stp in enumerate(steps[:50]):
                                    if not isinstance(stp, dict):
                                        continue
                                    with st.expander(f"Step {i+1} mark scheme", expanded=(i == 0)):
                                        st.markdown(normalize_markdown_math(str(stp.get("markscheme_text", "") or "")))
                                        miscon = stp.get("misconceptions", [])
                                        if isinstance(miscon, list) and miscon:
                                            st.markdown("**Common misconceptions:**")
                                            for m in miscon[:6]:
                                                st.markdown(normalize_markdown_math(f"- {m}"))
                        else:
                            with pv1:
                                st.markdown("**Question**")
                                with st.container(border=True):
                                    if q_img is not None:
                                        st.image(q_img, use_container_width=True)
                                    if q_text:
                                        st.markdown(normalize_markdown_math(q_text))
                                    if (q_img is None) and (not q_text):
                                        st.caption("No question text/image (image-only teacher uploads are supported).")

                            with pv2:
                                st.markdown("**Mark scheme (teacher only)**")
                                with st.container(border=True):
                                    if ms_img is not None:
                                        st.image(ms_img, use_container_width=True)
                                    if ms_text:
                                        st.markdown(normalize_markdown_math(ms_text))
                                    if (ms_img is None) and (not ms_text):
                                        st.caption("No mark scheme text/image (image-only teacher uploads are supported).")

                        if st.session_state.get("bank_delete_reset"):
                            st.session_state["bank_delete_picks"] = []
                            st.session_state["confirm_delete_bank_entry"] = False
                            st.session_state["bank_delete_reset"] = False

                        bank_delete_open = bool(st.session_state.get("bank_delete_picks")) or bool(
                            st.session_state.get("confirm_delete_bank_entry")
                        )
                        with st.expander("Delete question bank entries", expanded=bank_delete_open):
                            st.warning("Deleting a question removes it from the database permanently.")
                            delete_picks = st.multiselect(
                                "Select entries to delete",
                                options,
                                key="bank_delete_picks",
                            )
                            confirm_delete_q = st.checkbox(
                                "I understand this will permanently delete the selected entries.",
                                key="confirm_delete_bank_entry",
                            )
                            if st.button(
                                "Delete selected entries",
                                type="primary",
                                use_container_width=True,
                                disabled=not (confirm_delete_q and delete_picks),
                                key="delete_bank_entry_btn",
                            ):
                                delete_ok = True
                                for label in delete_picks:
                                    delete_id = int(df_f.loc[df_f["label"] == label, "id"].iloc[0])
                                    delete_ok = delete_question_bank_by_id(delete_id) and delete_ok
                                if delete_ok:
                                    st.success("Question bank entry(ies) deleted.")
                                    st.session_state["bank_delete_reset"] = True
                                    st.rerun()
                                else:
                                    st.error("Delete failed. Check database errors above.")

                        st.divider()
                        st.write("### Recent question bank entries")
                        df_bank = load_question_bank_df(limit=50, include_inactive=False)
                        if not df_bank.empty:
                            show_cols = [c for c in ["created_at", "source", "assignment_name", "question_label", "question_type", "max_marks", "id"] if c in df_bank.columns]
                            st.dataframe(df_bank[show_cols], use_container_width=True)
                        else:
                            st.info("No recent entries.")
            with tab_ai:
                st.write("## 🤖 AI generator (teacher vetting required)")
                gen_mode = st.radio("Generator", ["Single question", "Topic Journey"], horizontal=True, key="gen_mode")

                if gen_mode == "Single question":
                    st.caption("Generate a single GCSE Higher question + mark scheme. You must vet and edit before saving.")

                    gen_c1, gen_c2 = st.columns([2, 1])
                    with gen_c1:
                        topic_choice = st.selectbox(
                            "AQA GCSE Physics topic",
                            options=get_topic_names_for_track(st.session_state.get("track", TRACK_DEFAULT)),
                            key="topic_choice",
                            help="Topics shown depend on Combined/Separate selection.",
                        )
                        topic_text = topic_choice
                        topic_tok = get_topic_track_ok(topic_choice)
                        if topic_tok == "separate_only":
                            st.caption("Eligibility: Separate-only (hidden in Combined).")
                        else:
                            st.caption("Eligibility: Combined + Separate.")

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
                                ctx={"teacher": True, "mode": "single_question"},
                                typical_range="5-12 seconds",
                                est_seconds=10.0
                            )

                            if not isinstance(draft_raw, dict):
                                st.error("AI did not return a valid draft. Please try again.")
                            else:
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
                                    # Track eligibility for this draft
                                    draft_track_ok = get_topic_track_ok(topic_text)

                                    st.session_state["ai_draft"] = {
                                        "assignment_name": (assignment_name_ai or "").strip() or "AI Practice",
                                        "question_label": default_label,
                                        "track_ok": draft_track_ok,
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
                                    subject_site=SUBJECT_SITE,
                                    track_ok=d.get("track_ok", st.session_state.get("teacher_track_ok", "both")),
                                    assignment_name=d_assignment.strip(),
                                    question_label=d_label.strip(),
                                    max_marks=int(d_marks),
                                    tags=tags,
                                    question_text=d_qtext.strip(),
                                    markscheme_text=d_mstext.strip(),
                                    question_image_path=None,
                                    markscheme_image_path=None,
                                    question_type="single",
                                    journey_json=None,
                                )
                                if ok:
                                    st.session_state["ai_draft"] = None
                                    st.success("Approved and saved. Students can now access this under AI Practice.")
                                else:
                                    st.error("Failed to save to database. Check errors below.")

                else:
                    st.caption("Generate a step-by-step Topic Journey (one saved object). You must vet and edit before saving.")

                    jc1, jc2 = st.columns([2, 1])

                    # --- Left column: topic selection + controls ---
                    with jc1:
                        st.write("### Journey topics")

                        if "journey_topics_selected" not in st.session_state:
                            st.session_state["journey_topics_selected"] = []
                        if "journey_show_error" not in st.session_state:
                            st.session_state["journey_show_error"] = False

                        topic_pick = st.selectbox(
                            "Choose a topic to add",
                            options=get_topic_names_for_track(st.session_state.get("track", TRACK_DEFAULT)),
                            key="jour_topic_pick",
                            help="Add one or more topics. The journey will blend these topics into 5 steps (about 10 minutes).",
                        )

                        add_c1, add_c2 = st.columns([1, 1])
                        with add_c1:
                            if st.button("Add topic", key="jour_add_topic", use_container_width=True):
                                sel = list(st.session_state.get("journey_topics_selected", []) or [])
                                if topic_pick and topic_pick not in sel:
                                    sel.append(topic_pick)
                                    st.session_state["journey_topics_selected"] = sel
                                st.session_state["journey_show_error"] = False
                                st.rerun()

                        with add_c2:
                            if st.button("Clear topics", key="jour_clear_topics", use_container_width=True):
                                st.session_state["journey_topics_selected"] = []
                                st.session_state["journey_show_error"] = False
                                st.session_state["journey_draft"] = None
                                st.rerun()

                        sel_topics = list(st.session_state.get("journey_topics_selected", []) or [])
                        if sel_topics:
                            st.markdown("**Selected topics:**")
                            st.markdown("\n".join([f"- {t}" for t in sel_topics]))
                        else:
                            st.info("Add at least one topic to build a journey.")

                        # Fixed journey size: 10 minutes, 5 steps
                        j_duration = 10
                        st.caption("Journey length is fixed: 10 minutes, 5 steps.")

                        st.caption("Focus is chosen automatically based on the selected topic(s).")
                        j_assignment = st.text_input("Assignment name for saving", value="Topic Journey", key="jour_assignment")
                        j_tags = st.text_input("Tags (comma separated)", value="", key="jour_tags")

                    # --- Right column: actions ---
                    with jc2:
                        gen_j = st.button(
                            "Generate journey draft",
                            type="primary",
                            use_container_width=True,
                            disabled=not AI_READY,
                            key="jour_gen_btn",
                        )

                        if st.button("Clear journey draft", use_container_width=True, key="jour_clear_btn"):
                            st.session_state["journey_draft"] = None
                            st.session_state["journey_gen_error_details"] = None
                            st.session_state["journey_show_error"] = False
                            st.rerun()

                        if gen_j:
                            # Clear previous error state for this run
                            st.session_state["journey_gen_error_details"] = None
                            st.session_state["journey_show_error"] = False

                            sel_topics = list(st.session_state.get("journey_topics_selected", []) or [])
                            if not sel_topics:
                                st.warning("Please add at least one topic first.")
                            else:
                                topic_plain = " | ".join(sel_topics)

                                def task_generate():
                                    # 10 minutes, 5 steps is fixed
                                    return generate_topic_journey_with_ai(
                                        topic_plain_english=topic_plain,
                                        duration_minutes=j_duration,
                                    )

                                try:
                                    data = _run_ai_with_progress(
                                        task_fn=task_generate,
                                        ctx={"teacher": True, "mode": "topic_journey"},
                                        typical_range="15-35 seconds",
                                        est_seconds=25.0,
                                    )

                                    if data is None:
                                        raise ValueError("AI returned no usable Journey JSON (failed to parse).")

                                    # Validate basic shape
                                    
                                    # Default label
                                    token = pysecrets.token_hex(3)
                                    default_label = f"JOURNEY-{slugify(topic_plain)[:24]}-{token}"

                                    # Track eligibility: if any chosen topic is separate_only, the whole journey is separate_only.
                                    toks = [get_topic_track_ok(t) for t in sel_topics]
                                    draft_track_ok = "separate_only" if any(tok == "separate_only" for tok in toks) else "both"

                                    st.session_state["journey_draft"] = {
                                        "assignment_name": (j_assignment or "").strip() or "Topic Journey",
                                        "question_label": default_label,
                                        "track_ok": draft_track_ok,
                                        "tags": [t.strip() for t in (j_tags or "").split(",") if t.strip()],
                                        "journey": data,
                                    }
                                    st.success("Journey draft generated. Vet/edit below, then save as one assignment.")

                                except Exception:
                                    import traceback
                                    st.session_state["journey_draft"] = None
                                    st.session_state["journey_gen_error_details"] = traceback.format_exc()
                                    st.error("Failed to generate a Topic Journey. You can try again, or click 'Explain error'.")

                        # Optional: reveal raw error details if the user wants them
                        if st.session_state.get("journey_gen_error_details"):
                            if st.button("Explain error", key="jour_explain_error", use_container_width=True):
                                st.session_state["journey_show_error"] = True

                        if st.session_state.get("journey_show_error") and st.session_state.get("journey_gen_error_details"):
                            with st.expander("Error details", expanded=True):
                                st.code(st.session_state.get("journey_gen_error_details", ""))
                                st.caption("These details help diagnose failures (model output shape, JSON errors, timeouts).")
                    if st.session_state.get("journey_draft"):
                        d = st.session_state["journey_draft"]
                        journey = d.get("journey", {}) if isinstance(d, dict) else {}
                        steps = journey.get("steps", []) if isinstance(journey, dict) else []

                        if journey.get("warnings"):
                            st.warning("Journey draft warnings:\n\n" + "\n".join([f"- {w}" for w in journey.get("warnings", [])]))
                        st.write("### ✅ Vet and edit the journey")
                        hd1, hd2 = st.columns([2, 1])
                        with hd1:
                            d_assignment = st.text_input("Assignment name", value=d.get("assignment_name", "Topic Journey"), key="jour_draft_assignment")
                            d_label = st.text_input("Journey label", value=d.get("question_label", ""), key="jour_draft_label")
                            d_tags_str = st.text_input("Tags (comma separated)", value=", ".join(d.get("tags", [])), key="jour_draft_tags")

                        with hd2:
                            save_j = st.button("Save Topic Journey to bank", type="primary", use_container_width=True, key="jour_save_btn")
                            st.caption("Saved as a single Question Bank entry (type=journey).")

                        plan_md = st.text_area("Journey plan (Markdown)", value=journey.get("plan_markdown", ""), height=140, key="jour_plan_md")
                        render_md_box("Preview: Journey plan", plan_md, empty_text="No plan.")

                        # Optional teacher-only spec alignment preview
                        with st.expander("Show spec alignment (teacher only)", expanded=False):
                            spec_align = journey.get("spec_alignment", [])
                            if isinstance(spec_align, list) and spec_align:
                                for sref in spec_align[:20]:
                                    st.markdown(normalize_markdown_math(f"- {sref}"))
                            else:
                                st.caption("No spec alignment provided.")

                        st.write("### Steps")
                        if not isinstance(steps, list) or not steps:
                            st.error("No steps found in journey JSON.")
                        else:
                            total_marks = 0
                            edited_steps = []
                            for i, stp in enumerate(steps):
                                stp = stp if isinstance(stp, dict) else {}
                                with st.expander(f"Step {i+1}: {stp.get('objective','')[:80]}", expanded=(i == 0)):
                                    obj = st.text_input("Objective", value=str(stp.get("objective", "") or ""), key=f"jour_step_obj_{i}")
                                    mm = st.number_input("Max marks", min_value=1, max_value=12, value=int(stp.get("max_marks", 1) or 1), step=1, key=f"jour_step_mm_{i}")
                                    qtxt = st.text_area("Question text (Markdown + LaTeX)", value=str(stp.get("question_text", "") or ""), height=160, key=f"jour_step_q_{i}")
                                    mstxt = st.text_area("Mark scheme (ends with TOTAL = <max_marks>)", value=str(stp.get("markscheme_text", "") or ""), height=200, key=f"jour_step_ms_{i}")
                                    miscon = st.text_area("Common misconceptions (one per line)", value="\n".join([str(x) for x in (stp.get("misconceptions", []) or [])]), height=90, key=f"jour_step_mis_{i}")

                                    render_md_box("Preview: Question", qtxt, empty_text="No question text.")
                                    render_md_box("Preview: Mark scheme", mstxt, empty_text="No mark scheme.")

                                    total_marks += int(mm)
                                    edited_steps.append({
                                        "objective": str(obj or "").strip(),
                                        "question_text": str(qtxt or "").strip(),
                                        "markscheme_text": str(mstxt or "").strip(),
                                        "max_marks": int(mm),
                                        "misconceptions": [x.strip() for x in (miscon or "").split("\n") if x.strip()][:6],
                                        "spec_refs": [str(x).strip() for x in (stp.get("spec_refs", []) or []) if str(x).strip()][:6],
                                    })

                            if save_j:
                                if not d_assignment.strip() or not d_label.strip():
                                    st.error("Assignment name and Journey label cannot be blank.")
                                else:
                                    # Validate TOTAL lines
                                    bad_total = []
                                    for i, stp in enumerate(edited_steps):
                                        if f"TOTAL = {int(stp['max_marks'])}" not in (stp.get("markscheme_text") or ""):
                                            bad_total.append(i + 1)
                                    if bad_total:
                                        st.error("These steps are missing the required TOTAL line: " + ", ".join([str(x) for x in bad_total]))
                                        st.stop()

                                    tags = [t.strip() for t in (d_tags_str or "").split(",") if t.strip()]
                                    tags = tags[:20]

                                    journey_json = {
                                        "topic": str(journey.get("topic", "")).strip(),
                                        "duration_minutes": 10,
                                        "checkpoint_every": int(journey.get("checkpoint_every", JOURNEY_CHECKPOINT_EVERY) or JOURNEY_CHECKPOINT_EVERY),
                                        "plan_markdown": str(plan_md or "").strip(),
                                        "spec_alignment": [str(x).strip() for x in (journey.get("spec_alignment", []) or []) if str(x).strip()][:20],
                                        "steps": edited_steps,
                                    }

                                    ok = insert_question_bank_row(
                                        source="ai_generated",
                                        created_by="teacher",
                                        subject_site=SUBJECT_SITE,
                                        track_ok=d.get("track_ok", st.session_state.get("teacher_track_ok", "both")),
                                        assignment_name=d_assignment.strip(),
                                        question_label=d_label.strip(),
                                        max_marks=int(total_marks) if total_marks > 0 else 1,
                                        tags=tags,
                                        question_text=str(plan_md or "").strip(),
                                        markscheme_text="",
                                        question_image_path=None,
                                        markscheme_image_path=None,
                                        question_type="journey",
                                        journey_json=journey_json,
                                    )
                                    if ok:
                                        st.session_state["journey_draft"] = None
                                        st.success("Topic Journey saved. Students will see it as a single assignment and progress step-by-step.")
                                    else:
                                        st.error("Failed to save journey to database. Check errors below.")

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
                                subject_site=SUBJECT_SITE,
                                track_ok=st.session_state.get("teacher_track_ok", "both"),
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
