import json
import logging
from typing import Optional

import pandas as pd
import streamlit as st
from sqlalchemy import text

from config import SUBJECT_SITE, _safe_secret
from db import _exec_sql_many, get_db_engine


LOGGER = logging.getLogger("panphy")
ATTEMPTS_TABLE = "attempts_v1"
AI_TIMINGS_TABLE = "ai_timings_v1"


def ensure_attempts_table() -> None:
    if st.session_state.get("db_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return

    ddl_create = f"""
    create table if not exists public.{ATTEMPTS_TABLE} (
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
    alter table public.{ATTEMPTS_TABLE}
      add column if not exists question_bank_id bigint;
    alter table public.{ATTEMPTS_TABLE}
      add column if not exists step_index int;

    alter table public.{ATTEMPTS_TABLE}
      add column if not exists readback_type text;
    alter table public.{ATTEMPTS_TABLE}
      add column if not exists readback_markdown text;
    alter table public.{ATTEMPTS_TABLE}
      add column if not exists readback_warnings jsonb;
    alter table public.{ATTEMPTS_TABLE}
      add column if not exists subject_site text not null default '{SUBJECT_SITE}';
    alter table public.{ATTEMPTS_TABLE}
      add column if not exists track text not null default 'combined';
    """

    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, ddl_create)
            _exec_sql_many(conn, ddl_alter)
        st.session_state["db_last_error"] = ""
        st.session_state["db_table_ready"] = True
        LOGGER.info("Attempts table ready", extra={"ctx": {"component": "db", "table": ATTEMPTS_TABLE}})
    except Exception as exc:
        st.session_state["db_last_error"] = f"Table Creation Error: {type(exc).__name__}"
        st.session_state["db_table_ready"] = False
        LOGGER.exception(
            "Attempts table ensure failed",
            extra={"ctx": {"component": "db", "error": type(exc).__name__}},
        )


def ensure_ai_timings_table() -> None:
    if st.session_state.get("db_ai_timings_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return

    ddl_create = f"""
    create table if not exists public.{AI_TIMINGS_TABLE} (
      id bigserial primary key,
      created_at timestamptz not null default now(),
      subject_site text not null default '{SUBJECT_SITE}',
      timing_type text not null,
      duration_seconds double precision not null,
      success boolean not null default true
    );

    create index if not exists idx_ai_timings_subject_type_created
      on public.{AI_TIMINGS_TABLE} (subject_site, timing_type, created_at desc);
    """
    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, ddl_create)
        st.session_state["db_ai_timings_ready"] = True
        st.session_state["db_last_error"] = ""
        LOGGER.info("AI timings table ready", extra={"ctx": {"component": "db", "table": AI_TIMINGS_TABLE}})
    except Exception as exc:
        st.session_state["db_last_error"] = f"Table Creation Error: {type(exc).__name__}"
        st.session_state["db_ai_timings_ready"] = False
        LOGGER.exception(
            "AI timings table ensure failed",
            extra={"ctx": {"component": "db", "error": type(exc).__name__}},
        )


def insert_ai_timing(timing_type: str, duration_seconds: float, success: bool = True) -> None:
    eng = get_db_engine()
    if eng is None:
        return
    ensure_ai_timings_table()
    timing_type = (timing_type or "").strip().lower()
    if not timing_type:
        return
    query = f"""
        insert into public.{AI_TIMINGS_TABLE}
        (subject_site, timing_type, duration_seconds, success)
        values
        (:subject_site, :timing_type, :duration_seconds, :success)
    """
    try:
        with eng.begin() as conn:
            conn.execute(
                text(query),
                {
                    "subject_site": SUBJECT_SITE,
                    "timing_type": timing_type,
                    "duration_seconds": float(duration_seconds),
                    "success": bool(success),
                },
            )
        st.session_state["db_last_error"] = ""
    except Exception as exc:
        st.session_state["db_last_error"] = f"Insert Error: {type(exc).__name__}"
        LOGGER.exception("insert_ai_timing failed", extra={"ctx": {"component": "db", "error": type(exc).__name__}})


@st.cache_data(ttl=120)
def load_ai_timing_average_cached(
    _fp: str,
    subject_site: str,
    timing_type: str,
    min_samples: int = 5,
) -> tuple[Optional[float], int]:
    eng = get_db_engine()
    if eng is None:
        return None, 0
    ensure_ai_timings_table()
    subject_site = (subject_site or "").strip().lower() or SUBJECT_SITE
    timing_type = (timing_type or "").strip().lower()
    if not timing_type:
        return None, 0
    query = text(
        f"""
        select avg(duration_seconds) as avg_s, count(*) as n
        from public.{AI_TIMINGS_TABLE}
        where subject_site = :subject_site
          and timing_type = :timing_type
          and success = true
        """
    )
    with eng.connect() as conn:
        row = conn.execute(
            query,
            {"subject_site": subject_site, "timing_type": timing_type},
        ).fetchone()
    if not row:
        return None, 0
    avg_s = float(row[0]) if row[0] is not None else None
    count = int(row[1] or 0)
    if avg_s is None or count < max(1, int(min_samples)):
        return None, count
    return avg_s, count


def insert_attempt(
    student_id: str,
    question_key: str,
    report: dict,
    mode: str,
    question_bank_id: Optional[int] = None,
    step_index: Optional[int] = None,
) -> None:
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

    query = f"""
        insert into public.{ATTEMPTS_TABLE}
        (subject_site, track, student_id, question_key, question_bank_id, step_index, mode, marks_awarded, max_marks, summary, feedback_points, next_steps,
         readback_type, readback_markdown, readback_warnings)
        values
        (:subject_site, :track, :student_id, :question_key, :question_bank_id, :step_index, :mode, :marks_awarded, :max_marks, :summary,
         CAST(:feedback_points AS jsonb), CAST(:next_steps AS jsonb),
         :readback_type, :readback_markdown, CAST(:readback_warnings AS jsonb))
    """
    try:
        with eng.begin() as conn:
            conn.execute(
                text(query),
                {
                    "subject_site": SUBJECT_SITE,
                    "track": st.session_state.get("track", "combined"),
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
                    "readback_warnings": rb_warn,
                },
            )
        st.session_state["db_last_error"] = ""
    except Exception as exc:
        st.session_state["db_last_error"] = f"Insert Error: {type(exc).__name__}"
        LOGGER.exception("insert_attempt failed", extra={"ctx": {"component": "db", "error": type(exc).__name__}})


@st.cache_data(ttl=20)
def load_attempts_df_cached(_fp: str, subject_site: str, limit: int = 5000) -> pd.DataFrame:
    eng = get_db_engine()
    if eng is None:
        return pd.DataFrame()
    ensure_attempts_table()
    subject_site = (subject_site or "").strip().lower() or SUBJECT_SITE
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                f"""
                select id, created_at, student_id, question_key, question_bank_id, step_index, mode,
                       marks_awarded, max_marks, readback_type
                from public.{ATTEMPTS_TABLE}
                where subject_site = :subject_site
                order by created_at desc
                limit :limit
            """
            ),
            conn,
            params={"limit": int(limit), "subject_site": subject_site},
        )
    if not df.empty:
        df["marks_awarded"] = pd.to_numeric(df["marks_awarded"], errors="coerce").fillna(0).astype(int)
        df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce").fillna(0).astype(int)
    return df


def load_attempts_df(limit: int = 5000) -> pd.DataFrame:
    fp = (_safe_secret("DATABASE_URL", "") or "")[:40]
    subject_site = (SUBJECT_SITE or "").strip().lower()
    try:
        return load_attempts_df_cached(fp, subject_site=subject_site, limit=limit)
    except Exception as exc:
        st.session_state["db_last_error"] = f"Load Error: {type(exc).__name__}"
        LOGGER.exception("load_attempts_df failed", extra={"ctx": {"component": "db", "error": type(exc).__name__}})
        return pd.DataFrame()


def delete_attempt_by_id(attempt_id: int) -> bool:
    eng = get_db_engine()
    if eng is None:
        return False
    ensure_attempts_table()
    try:
        with eng.begin() as conn:
            res = conn.execute(
                text(f"delete from public.{ATTEMPTS_TABLE} where id = :id and subject_site = :subject_site"),
                {"id": int(attempt_id), "subject_site": SUBJECT_SITE},
            )
        st.session_state["db_last_error"] = ""
        try:
            load_attempts_df_cached.clear()
        except Exception:
            pass
        return res.rowcount > 0
    except Exception as exc:
        st.session_state["db_last_error"] = f"Delete Attempt Error: {type(exc).__name__}"
        LOGGER.exception(
            "delete_attempt_by_id failed",
            extra={"ctx": {"component": "db", "error": type(exc).__name__}},
        )
        return False
