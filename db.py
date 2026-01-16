import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from config import SUBJECT_SITE, _safe_secret

LOGGER = logging.getLogger("panphy")

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
    raw_url = _safe_secret("DATABASE_URL", "") or ""
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


# ============================================================
# DATABASE DDLs
#   IMPORTANT: avoid $$ PL/pgSQL blocks inside app DDL to prevent split/execution issues.
# ============================================================
QUESTION_BANK_DDL = f"""
create table if not exists public.question_bank_v2 (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  source text not null check (source in ('teacher','ai_generated')),
  created_by text,

  assignment_name text not null,
  question_label text not null,
  max_marks int not null check (max_marks > 0),
  topic text,
  sub_topic text,
  sub_topic_raw text,
  skill text,
  difficulty text,
  tags jsonb,

  question_text text,
  question_image_path text,

  markscheme_text text,
  markscheme_image_path text,

  question_type text not null default 'single',
  journey_json jsonb,
  subject_site text not null default '{SUBJECT_SITE}',
  track_ok text not null default 'both',

  is_active boolean not null default true
);

create unique index if not exists uq_question_bank_subject_source_assignment_label
  on public.question_bank_v2 (subject_site, source, assignment_name, question_label);

create index if not exists idx_question_bank_assignment
  on public.question_bank_v2 (assignment_name);

create index if not exists idx_question_bank_source
  on public.question_bank_v2 (source);

create index if not exists idx_question_bank_active
  on public.question_bank_v2 (is_active);
""".strip()


QUESTION_BANK_ALTER_DDL = """
alter table public.question_bank_v2
  add column if not exists sub_topic_raw text;

create unique index if not exists uq_question_bank_subject_source_assignment_label
  on public.question_bank_v2 (subject_site, source, assignment_name, question_label);
""".strip()


@st.cache_resource
def _ensure_question_bank_table_cached(_fp: str) -> None:
    eng = get_db_engine()
    if eng is None:
        raise RuntimeError("Database engine not configured.")
    with eng.begin() as conn:
        _exec_sql_many(conn, QUESTION_BANK_DDL)
        _exec_sql_many(conn, QUESTION_BANK_ALTER_DDL)
    LOGGER.info("Question bank table ready", extra={"ctx": {"component": "db", "table": "question_bank_v2"}})


def ensure_question_bank_table():
    if st.session_state.get("bank_table_ready", False):
        return
    fp = (_safe_secret("DATABASE_URL", "") or "")[:40]
    try:
        _ensure_question_bank_table_cached(fp)
        st.session_state["bank_table_ready"] = True
        st.session_state["db_last_error"] = ""
    except Exception as e:
        st.session_state["db_last_error"] = f"Question Bank Table Error: {type(e).__name__}: {e}"
        st.session_state["bank_table_ready"] = False
        LOGGER.error("Question bank table ensure failed", extra={"ctx": {"component": "db", "error": type(e).__name__}})


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
                  max_marks, question_type,
                  topic, sub_topic, sub_topic_raw, skill, difficulty,
                  tags, question_text,
                  subject_site, track_ok, is_active
                from public.question_bank_v2
                {where}
                order by created_at desc
                limit :limit
            """),
            conn,
            params={"limit": int(limit), "subject_site": subject_site},
        )
    return df



def load_question_bank_df(limit: int = 5000, include_inactive: bool = False) -> pd.DataFrame:
    fp = (_safe_secret("DATABASE_URL", "") or "")[:40]
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
            text("select * from public.question_bank_v2 where id = :id and subject_site = :subject_site limit 1"),
            {"id": int(qid), "subject_site": SUBJECT_SITE}
        ).mappings().first()
    return dict(row) if row else {}



def load_question_by_id(qid: int) -> Dict[str, Any]:
    fp = (_safe_secret("DATABASE_URL", "") or "")[:40]
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
    topic: Optional[str] = None,
    sub_topic: Optional[str] = None,
    sub_topic_raw: Optional[str] = None,
    skill: Optional[str] = None,
    difficulty: Optional[str] = None,
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
    insert into public.question_bank_v2
      (source, created_by, subject_site, track_ok, assignment_name, question_label, max_marks,
       topic, sub_topic, sub_topic_raw, skill, difficulty,
       question_type, journey_json, tags,
       question_text, question_image_path,
       markscheme_text, markscheme_image_path,
       is_active, updated_at)
    values
      (:source, :created_by, :subject_site, :track_ok, :assignment_name, :question_label, :max_marks,
       :topic, :sub_topic, :sub_topic_raw, :skill, :difficulty,
       :question_type, CAST(:journey_json AS jsonb),
       CAST(:tags AS jsonb),
       :question_text, :question_image_path,
       :markscheme_text, :markscheme_image_path,
       true, now())
    on conflict (subject_site, source, assignment_name, question_label) do update set
       created_by = excluded.created_by,
       subject_site = excluded.subject_site,
       track_ok = excluded.track_ok,
       max_marks = excluded.max_marks,
       topic = excluded.topic,
       sub_topic = excluded.sub_topic,
       sub_topic_raw = excluded.sub_topic_raw,
       skill = excluded.skill,
       difficulty = excluded.difficulty,
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
                "topic": (topic or "").strip() or None,
                "sub_topic": (sub_topic or "").strip() or None,
                "sub_topic_raw": (sub_topic_raw or sub_topic or "").strip() or None,
                "skill": (skill or "").strip() or None,
                "difficulty": (difficulty or "").strip() or None,
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
                text("delete from public.question_bank_v2 where id = :id and subject_site = :subject_site"),
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
