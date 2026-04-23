import logging
import os
from logging.handlers import RotatingFileHandler

import streamlit as st

from ai_feedback import get_gpt_feedback_from_bank
from ai_generation import AI_READY, MODEL_NAME
from ai_progress import _run_ai_with_progress
from attempts import delete_attempt_by_id, ensure_attempts_table, insert_attempt, load_attempts_df
from canvas_utils import canvas_has_ink, data_url_to_image_data, preprocess_canvas_image
from db import db_ready, get_db_driver_type, insert_question_bank_row
from image_utils import _compress_bytes_to_limit, _encode_image_bytes, validate_image_file
from markdown_rendering import normalize_markdown_math, render_md_box, render_report
from rate_limiter import RATE_LIMIT_MAX, _check_rate_limit_db, _effective_student_id
from session_state import init_session_state
from storage import (
    bytes_to_pil,
    cached_download_from_storage,
    safe_bytes_to_pil,
    slugify,
    supabase_ready,
    upload_to_storage,
)
from track_state import (
    TRACK_DEFAULT,
    TRACK_PARAM,
    init_track_state,
    persist_track_to_browser,
    set_query_param,
)
from ui_student import render_student_page
from ui_teacher import render_teacher_page


PANPHY_LOGO_URL = "https://panphy.github.io/assets/panphy.png"
PANPHY_FAVICON_URL = "https://panphy.github.io/assets/favicon.png"

CANVAS_BG_HEX = "#ffffff"
TEXTAREA_HEIGHT_DEFAULT = 420
TEXTAREA_HEIGHT_EXPANDED = 640
CANVAS_HEIGHT_DEFAULT = 640
CANVAS_HEIGHT_EXPANDED = 860

QUESTION_MAX_MB = 5.0
MARKSCHEME_MAX_MB = 5.0
CANVAS_MAX_MB = 2.0


class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict) and ctx:
            keys = sorted(ctx.keys())
            kv = " ".join([f"[{key}={ctx[key]}]" for key in keys if ctx[key] is not None and ctx[key] != ""])
            if kv:
                return f"{base} {kv}"
        return base


class SessionIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            sid = st.session_state.get("session_id", "")
        except Exception:
            sid = ""
        if sid:
            ctx = getattr(record, "ctx", None)
            if isinstance(ctx, dict):
                ctx.setdefault("sid", sid)
            else:
                record.ctx = {"sid": sid}
        return True


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
    logger.addFilter(SessionIDFilter())
    logger.info("Logging configured", extra={"ctx": {"component": "startup"}})
    return logger


st.set_page_config(
    page_title="PanPhy Skill Builder",
    page_icon=PANPHY_FAVICON_URL,
    layout="wide",
)

init_session_state()
LOGGER = setup_logging()

try:
    from components.panphy_stylus_canvas import stylus_canvas
except Exception:
    LOGGER.exception("Failed to import stylus_canvas; canvas features disabled.")
    stylus_canvas = None


def _stylus_canvas_available() -> bool:
    return stylus_canvas is not None


def _render_badge(label: str, *, color: str, icon: str | None = None) -> None:
    if hasattr(st, "badge"):
        st.badge(label, color=color, icon=icon)
    else:
        md_color = color if color in {"red", "orange", "yellow", "blue", "green", "violet", "gray", "grey"} else "blue"
        st.markdown(f":{md_color}-badge[{label}]")


def _render_sidebar() -> str:
    nav = st.sidebar.radio(
        "Navigate",
        ["Student", "Teacher Dashboard", "Question Bank"],
        index=0,
        key="nav_page",
    )

    track_label = st.sidebar.selectbox(
        "Track",
        ["Combined", "Separate"],
        index=0 if st.session_state.get("track", TRACK_DEFAULT) == "combined" else 1,
        key="sidebar_track_label",
        help="Combined hides Separate-only topics/questions. Separate shows everything.",
    )
    selected_track = "combined" if track_label == "Combined" else "separate"
    if selected_track != st.session_state.get("track", TRACK_DEFAULT):
        st.session_state["track"] = selected_track
        set_query_param(**{TRACK_PARAM: selected_track})
    persist_track_to_browser(st.session_state.get("track", TRACK_DEFAULT))

    with st.sidebar:
        if st.session_state.get("track", TRACK_DEFAULT) == "combined":
            _render_badge("COMBINED", color="orange")
        else:
            _render_badge("SEPARATE", color="primary")
        st.caption("The badge shows whether COMBINED or SEPARATED Physics selected.")

    return nav


def _render_header() -> None:
    header_left, _header_mid, header_right = st.columns([3, 2, 1])
    track = st.session_state.get("track", TRACK_DEFAULT)

    with header_right:
        if track == "combined":
            _render_badge("COMBINED", color="orange", icon=":material/merge_type:")
        else:
            _render_badge("SEPARATE", color="primary", icon=":material/call_split:")

    with header_left:
        logo_col, title_col = st.columns([1, 5])
        with logo_col:
            st.markdown(
                f'<a href="https://panphy.github.io/" target="_blank" rel="noopener noreferrer">'
                f'<img src="{PANPHY_LOGO_URL}" width="64" alt="PanPhy logo"/></a>',
                unsafe_allow_html=True,
            )
        with title_col:
            st.title("PanPhy Skill Builder")
            st.caption(f"Powered by OpenAI {MODEL_NAME}")

    with header_right:
        issues = []
        if not AI_READY:
            issues.append("AI model not connected.")
        if not db_ready():
            issues.append("Database not connected.")
        if issues:
            st.caption("⚠️ System status")
            for message in issues:
                st.caption(message)


def _build_ui_helpers() -> dict:
    return {
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


def main() -> None:
    init_track_state()
    nav = _render_sidebar()
    _render_header()

    helpers = _build_ui_helpers()
    if nav == "Student":
        render_student_page(helpers)
    elif nav in ("Teacher Dashboard", "Question Bank"):
        render_teacher_page(nav, helpers)

    st.divider()
    st.markdown(
        "<div style='text-align: center;'>"
        "© 2026 PanPhy Projects<br>"
        "<a href='mailto:panphylabs@icloud.com'>Contact Me</a> • "
        "<a href='https://buymeacoffee.com/panphy'>Support My Projects</a>"
        "</div>",
        unsafe_allow_html=True,
    )


main()
