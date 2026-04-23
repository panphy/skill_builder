import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Tuple
from zoneinfo import ZoneInfo

import streamlit as st
from sqlalchemy import text

from config import SUBJECT_SITE
from db import _exec_sql_many, get_db_engine


LOGGER = logging.getLogger("panphy")
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW_SECONDS = 60 * 60


RATE_LIMITS_DDL = f"""
create table if not exists public.rate_limits (
  subject_site text not null default '{SUBJECT_SITE}',
  student_id text not null,
  submission_count int not null default 0,
  window_start_time timestamptz not null default now(),
  tokens double precision not null default {float(RATE_LIMIT_MAX)},
  last_refill_at timestamptz not null default now(),
  primary key (subject_site, student_id)
);
create index if not exists idx_rate_limits_window_start_time
  on public.rate_limits (window_start_time);
create index if not exists idx_rate_limits_last_refill_at
  on public.rate_limits (last_refill_at);
""".strip()


def _refill_tokens(tokens: float, elapsed_seconds: float, capacity: int, window_seconds: int) -> float:
    capacity = max(1, int(capacity))
    window_seconds = max(1, int(window_seconds))
    current = min(float(capacity), max(0.0, float(tokens)))
    elapsed = max(0.0, float(elapsed_seconds))
    return min(float(capacity), current + elapsed * (float(capacity) / float(window_seconds)))


def _seconds_until_next_token(tokens: float, capacity: int, window_seconds: int) -> int:
    if tokens >= 1.0:
        return 0
    capacity = max(1, int(capacity))
    window_seconds = max(1, int(window_seconds))
    refill_rate = float(capacity) / float(window_seconds)
    return max(1, int(math.ceil((1.0 - max(0.0, tokens)) / refill_rate)))


def ensure_rate_limits_table() -> None:
    if st.session_state.get("rate_limits_table_ready", False):
        return
    eng = get_db_engine()
    if eng is None:
        return

    capacity = float(RATE_LIMIT_MAX)
    ddl_migrate = f"""
    alter table public.rate_limits
      add column if not exists subject_site text not null default '{SUBJECT_SITE}';
    alter table public.rate_limits
      add column if not exists tokens double precision;
    alter table public.rate_limits
      add column if not exists last_refill_at timestamptz;

    update public.rate_limits
      set subject_site = '{SUBJECT_SITE}'
      where subject_site is null or subject_site = '';
    update public.rate_limits
      set tokens = greatest(0::double precision, least({capacity}, {capacity} - coalesce(submission_count, 0)::double precision))
      where tokens is null;
    update public.rate_limits
      set last_refill_at = coalesce(window_start_time, now())
      where last_refill_at is null;

    alter table public.rate_limits
      alter column tokens set default {capacity};
    alter table public.rate_limits
      alter column tokens set not null;
    alter table public.rate_limits
      alter column last_refill_at set default now();
    alter table public.rate_limits
      alter column last_refill_at set not null;
    alter table public.rate_limits
      drop constraint if exists rate_limits_pkey;
    alter table public.rate_limits
      add primary key (subject_site, student_id);
    """
    try:
        with eng.begin() as conn:
            _exec_sql_many(conn, RATE_LIMITS_DDL)
            _exec_sql_many(conn, ddl_migrate)
        st.session_state["rate_limits_table_ready"] = True
        LOGGER.info("Rate limits table ready", extra={"ctx": {"component": "db", "table": "rate_limits"}})
    except Exception as exc:
        st.session_state["db_last_error"] = f"Rate Limits Table Error: {type(exc).__name__}"
        st.session_state["rate_limits_table_ready"] = False
        LOGGER.exception(
            "Rate limits table ensure failed",
            extra={"ctx": {"component": "db", "error": type(exc).__name__}},
        )


def _effective_student_id(student_id: str) -> str:
    sid = (student_id or "").strip()
    if sid:
        return sid
    return f"anon_{st.session_state['anon_id']}"


def _format_reset_time(dt_utc: datetime) -> str:
    try:
        local = dt_utc.astimezone(ZoneInfo("Europe/London"))
        return local.strftime("%H:%M on %d %b %Y")
    except Exception:
        return dt_utc.strftime("%H:%M UTC on %d %b %Y")


def _coerce_utc(value, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return fallback


def _check_rate_limit_db(student_id: str) -> Tuple[bool, int, str]:
    if st.session_state.get("is_teacher", False):
        return True, RATE_LIMIT_MAX, ""

    eng = get_db_engine()
    if eng is None:
        return True, RATE_LIMIT_MAX, ""

    ensure_rate_limits_table()

    sid = _effective_student_id(student_id)
    now_utc = datetime.now(timezone.utc)

    try:
        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    insert into public.rate_limits
                    (subject_site, student_id, submission_count, window_start_time, tokens, last_refill_at)
                    values (:subject_site, :sid, 0, :now_utc, :capacity, :now_utc)
                    on conflict (subject_site, student_id) do nothing
                    """
                ),
                {
                    "subject_site": SUBJECT_SITE,
                    "sid": sid,
                    "now_utc": now_utc,
                    "capacity": float(RATE_LIMIT_MAX),
                },
            )

            row = conn.execute(
                text(
                    """
                    select subject_site, student_id, submission_count, window_start_time, tokens, last_refill_at
                    from public.rate_limits
                    where subject_site = :subject_site
                      and student_id = :sid
                    for update
                    """
                ),
                {"subject_site": SUBJECT_SITE, "sid": sid},
            ).mappings().first()

            if not row:
                return True, RATE_LIMIT_MAX, ""

            last_refill = _coerce_utc(row["last_refill_at"], now_utc)
            legacy_count = int(row["submission_count"] or 0)
            stored_tokens = row["tokens"]
            if stored_tokens is None:
                stored_tokens = max(0.0, float(RATE_LIMIT_MAX - legacy_count))

            elapsed = (now_utc - last_refill).total_seconds()
            tokens = _refill_tokens(stored_tokens, elapsed, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0

            remaining = max(0, int(math.floor(tokens)))
            legacy_submission_count = max(0, RATE_LIMIT_MAX - remaining)
            conn.execute(
                text(
                    """
                    update public.rate_limits
                    set tokens = :tokens,
                        last_refill_at = :now_utc,
                        submission_count = :submission_count
                    where subject_site = :subject_site
                      and student_id = :sid
                    """
                ),
                {
                    "tokens": float(tokens),
                    "now_utc": now_utc,
                    "submission_count": int(legacy_submission_count),
                    "subject_site": SUBJECT_SITE,
                    "sid": sid,
                },
            )

        reset_seconds = _seconds_until_next_token(tokens, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS)
        reset_str = _format_reset_time(now_utc + timedelta(seconds=reset_seconds))
        return allowed, remaining, reset_str
    except Exception as exc:
        st.session_state["db_last_error"] = f"Rate Limit Error: {type(exc).__name__}"
        LOGGER.exception("Rate limit check failed", extra={"ctx": {"component": "db", "error": type(exc).__name__}})
        return True, RATE_LIMIT_MAX, ""
