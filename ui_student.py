import io
import json
import math
from typing import Any, Dict

import pandas as pd
import streamlit as st
from PIL import Image

from ai_generation import AI_READY, JOURNEY_CHECKPOINT_EVERY
from config import (
    clean_sub_topic_label,
    SUBJECT_SITE,
    _safe_secret,
)
from db import (
    load_question_bank_distinct_difficulties,
    load_question_bank_distinct_skills,
    load_question_bank_distinct_sub_topics,
    load_question_bank_distinct_topics,
    load_question_bank_page,
    load_question_by_id,
)


def render_student_page(helpers: dict):
    CANVAS_BG_HEX = helpers["CANVAS_BG_HEX"]
    CANVAS_HEIGHT_DEFAULT = helpers["CANVAS_HEIGHT_DEFAULT"]
    CANVAS_HEIGHT_EXPANDED = helpers["CANVAS_HEIGHT_EXPANDED"]
    CANVAS_MAX_MB = helpers["CANVAS_MAX_MB"]
    RATE_LIMIT_MAX = helpers["RATE_LIMIT_MAX"]
    TEXTAREA_HEIGHT_DEFAULT = helpers["TEXTAREA_HEIGHT_DEFAULT"]
    TEXTAREA_HEIGHT_EXPANDED = helpers["TEXTAREA_HEIGHT_EXPANDED"]
    _check_rate_limit_db = helpers["_check_rate_limit_db"]
    _compress_bytes_to_limit = helpers["_compress_bytes_to_limit"]
    _effective_student_id = helpers["_effective_student_id"]
    _encode_image_bytes = helpers["_encode_image_bytes"]
    _run_ai_with_progress = helpers["_run_ai_with_progress"]
    _stylus_canvas_available = helpers["_stylus_canvas_available"]
    cached_download_from_storage = helpers["cached_download_from_storage"]
    canvas_has_ink = helpers["canvas_has_ink"]
    data_url_to_image_data = helpers["data_url_to_image_data"]
    db_ready = helpers["db_ready"]
    get_gpt_feedback_from_bank = helpers["get_gpt_feedback_from_bank"]
    insert_attempt = helpers["insert_attempt"]
    normalize_markdown_math = helpers["normalize_markdown_math"]
    preprocess_canvas_image = helpers["preprocess_canvas_image"]
    render_report = helpers["render_report"]
    safe_bytes_to_pil = helpers["safe_bytes_to_pil"]
    stylus_canvas = helpers["stylus_canvas"]
    validate_image_file = helpers["validate_image_file"]
    bytes_to_pil = helpers["bytes_to_pil"]

    st.divider()
    st.subheader("Student Workspace")
    st.caption("Choose a question set, confirm the filters, then answer and submit for feedback.")

    source_options = ["AI Practice", "Teacher Uploads", "All"]
    expand_by_default = False

    def _display_classification(value: Any, fallback: str = "Uncategorized") -> str:
        if pd.isna(value):
            return fallback
        text = str(value).strip()
        return text if text else fallback

    track = st.session_state.get("track", "combined")

    def _display_sub_topic(value: Any, raw_value: Any = None, fallback: str = "Uncategorized") -> str:
        primary = _display_classification(value, "")
        if primary:
            return clean_sub_topic_label(primary, track)
        secondary = _display_classification(raw_value, fallback)
        return clean_sub_topic_label(secondary, track)

    def _reset_student_filters() -> None:
        st.session_state["student_source"] = "AI Practice"
        st.session_state["student_topic_filter"] = "All"
        st.session_state["student_sub_topic_filter"] = "All"
        st.session_state["student_skill_filter"] = "All"
        st.session_state["student_difficulty_filter"] = "All"
        st.session_state["student_page_index"] = 0
        st.session_state["student_page_number"] = 1
        st.session_state["selected_qid"] = None
        st.session_state["cached_q_row"] = None

    def _render_filter_chips(chips: list[str]) -> None:
        if not chips:
            st.caption("No active filters. Showing everything for this source.")
            return
        st.markdown("**Active filters:** " + ", ".join(chips))

    def _advance_question(
        choice_key: str | None,
        question_sequence: list[int],
        label_by_id: dict[int, str],
        current_qid: int | None,
    ) -> None:
        if not choice_key or not question_sequence or current_qid is None:
            return
        try:
            current_index = question_sequence.index(current_qid)
        except ValueError:
            return
        next_index = (current_index + 1) % len(question_sequence)
        next_id = question_sequence[next_index]
        next_label = label_by_id.get(next_id)
        if next_label:
            st.session_state[choice_key] = next_label
            st.session_state["selected_qid"] = None

    def _retreat_question(
        choice_key: str | None,
        question_sequence: list[int],
        label_by_id: dict[int, str],
        current_qid: int | None,
    ) -> None:
        if not choice_key or not question_sequence or current_qid is None:
            return
        try:
            current_index = question_sequence.index(current_qid)
        except ValueError:
            return
        prev_index = (current_index - 1) % len(question_sequence)
        prev_id = question_sequence[prev_index]
        prev_label = label_by_id.get(prev_id)
        if prev_label:
            st.session_state[choice_key] = prev_label
            st.session_state["selected_qid"] = None

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
            st.info(
                "Teachers: add DATABASE_URL in secrets, then upload questions in 📚 Question Bank → Upload scans."
            )
        else:
            source_map = {
                "AI Practice": "ai_generated",
                "Teacher Uploads": "teacher",
                "All": None,
            }
            source_filter = source_map.get(source, None)
            topic_options = load_question_bank_distinct_topics(source=source_filter)
            if not topic_options:
                st.info(
                    "No questions in the database yet. Teachers can generate AI practice questions or upload scans in "
                    "📚 Question Bank → Upload scans."
                )
            else:
                with st.container(border=True):
                    header_cols = st.columns([3, 1])
                    with header_cols[0]:
                        st.markdown("### Filters")
                        st.caption("Refine the question list, then confirm the active filters below.")
                    with header_cols[1]:
                        st.button(
                            "Reset filters",
                            use_container_width=True,
                            on_click=_reset_student_filters,
                            key="student_filters_reset",
                        )

                    topics = ["All"] + sorted(topic_options, key=lambda value: _display_classification(value).lower())
                    if st.session_state.get("student_topic_filter") not in topics:
                        st.session_state["student_topic_filter"] = "All"
                    topic_cols = st.columns(2)
                    with topic_cols[0]:
                        topic_filter = st.selectbox(
                            "Step 1: Topic group",
                            topics,
                            key="student_topic_filter",
                            format_func=lambda value: "All" if value == "All" else _display_classification(value),
                        )

                    sub_topic_options = load_question_bank_distinct_sub_topics(
                        source=source_filter,
                        topic=None if topic_filter == "All" else topic_filter,
                    )
                    sub_topics = ["All"] + sorted(
                        sub_topic_options,
                        key=lambda value: clean_sub_topic_label(value, track).lower(),
                    )
                    if st.session_state.get("student_sub_topic_filter") not in sub_topics:
                        st.session_state["student_sub_topic_filter"] = "All"
                    with topic_cols[1]:
                        sub_topic_filter = st.selectbox(
                            "Step 2: Topic",
                            sub_topics,
                            key="student_sub_topic_filter",
                            format_func=lambda value: "All" if value == "All" else _display_sub_topic(value, value),
                        )

                    skill_cols = st.columns(2)
                    skill_options = load_question_bank_distinct_skills(
                        source=source_filter,
                        topic=None if topic_filter == "All" else topic_filter,
                        sub_topic=None if sub_topic_filter == "All" else sub_topic_filter,
                    )
                    skills = ["All"] + sorted(skill_options, key=lambda value: _display_classification(value).lower())
                    if st.session_state.get("student_skill_filter") not in skills:
                        st.session_state["student_skill_filter"] = "All"
                    with skill_cols[0]:
                        skill_filter = st.selectbox(
                            "Step 3: Skill",
                            skills,
                            key="student_skill_filter",
                            format_func=lambda value: "All" if value == "All" else _display_classification(value),
                        )

                    difficulty_options = load_question_bank_distinct_difficulties(
                        source=source_filter,
                        topic=None if topic_filter == "All" else topic_filter,
                        sub_topic=None if sub_topic_filter == "All" else sub_topic_filter,
                        skill=None if skill_filter == "All" else skill_filter,
                    )
                    difficulties = ["All"] + sorted(
                        difficulty_options,
                        key=lambda value: _display_classification(value).lower(),
                    )
                    if st.session_state.get("student_difficulty_filter") not in difficulties:
                        st.session_state["student_difficulty_filter"] = "All"
                    with skill_cols[1]:
                        difficulty_filter = st.selectbox(
                            "Step 4: Difficulty",
                            difficulties,
                            key="student_difficulty_filter",
                            format_func=lambda value: "All" if value == "All" else _display_classification(value),
                        )

                    active_filters = []
                    if source != "All":
                        active_filters.append(f"Source: {source}")
                    if topic_filter != "All":
                        active_filters.append(f"Topic group: {_display_classification(topic_filter)}")
                    if sub_topic_filter != "All":
                        active_filters.append(f"Topic: {_display_sub_topic(sub_topic_filter, sub_topic_filter)}")
                    if skill_filter != "All":
                        active_filters.append(f"Skill: {_display_classification(skill_filter)}")
                    if difficulty_filter != "All":
                        active_filters.append(f"Difficulty: {_display_classification(difficulty_filter)}")

                    st.markdown("**Active filters**")
                    _render_filter_chips(active_filters)

                filter_state = (
                    source_filter,
                    topic_filter,
                    sub_topic_filter,
                    skill_filter,
                    difficulty_filter,
                )
                if st.session_state.get("student_filter_state") != filter_state:
                    st.session_state["student_filter_state"] = filter_state
                    st.session_state["student_page_index"] = 0
                    st.session_state["student_page_number"] = 1

                page_cols = st.columns([1.2, 1.4, 2])
                with page_cols[0]:
                    page_size = st.selectbox(
                        "Questions per page",
                        [10, 25, 50, 100],
                        key="student_page_size",
                        help="Adjust how many questions show up at once.",
                    )
                page_index = int(st.session_state.get("student_page_index", 0) or 0)
                page_data = load_question_bank_page(
                    source=source_filter,
                    topic=None if topic_filter == "All" else topic_filter,
                    sub_topic=None if sub_topic_filter == "All" else sub_topic_filter,
                    skill=None if skill_filter == "All" else skill_filter,
                    difficulty=None if difficulty_filter == "All" else difficulty_filter,
                    page_size=page_size,
                    page_index=page_index,
                )
                total_questions = int(page_data.get("total", 0) or 0)
                with st.container(border=True):
                    st.markdown("### Question status")
                    st.caption(
                        f"Source: {source} • Topic group: "
                        f"{_display_classification(topic_filter) if topic_filter != 'All' else 'All topics'} • "
                        f"Difficulty: {_display_classification(difficulty_filter) if difficulty_filter != 'All' else 'All levels'}"
                    )
                    st.caption(f"Questions available for this selection: {total_questions}")
                max_page_index = max(0, math.ceil(total_questions / page_size) - 1)
                if page_index > max_page_index:
                    page_index = max_page_index
                    st.session_state["student_page_index"] = page_index
                    st.session_state["student_page_number"] = page_index + 1
                    page_data = load_question_bank_page(
                        source=source_filter,
                        topic=None if topic_filter == "All" else topic_filter,
                        sub_topic=None if sub_topic_filter == "All" else sub_topic_filter,
                        skill=None if skill_filter == "All" else skill_filter,
                        difficulty=None if difficulty_filter == "All" else difficulty_filter,
                        page_size=page_size,
                        page_index=page_index,
                    )
                with page_cols[1]:
                    max_page_number = max_page_index + 1 if total_questions else 1
                    st.session_state.setdefault("student_page_number", page_index + 1)
                    page_number = st.number_input(
                        "Page number",
                        min_value=1,
                        max_value=max_page_number,
                        step=1,
                        key="student_page_number",
                        help="Jump to a specific page of results.",
                        disabled=max_page_number <= 1,
                    )
                    st.session_state["student_page_index"] = int(page_number) - 1
                with page_cols[2]:
                    if total_questions:
                        start_index = page_index * page_size + 1
                        end_index = min(total_questions, (page_index + 1) * page_size)
                        st.caption(
                            f"Showing {start_index}-{end_index} of {total_questions} questions "
                            f"(page {page_index + 1} of {max_page_number})"
                        )
                    else:
                        st.caption("No questions match this filter set yet.")

                df_filtered = page_data.get("df", pd.DataFrame()).copy()
                if df_filtered.empty:
                    empty_choice = "No questions yet for this selection."
                    st.selectbox(
                        "Question",
                        [empty_choice],
                        key=f"student_question_choice::empty::{source_filter}::{topic_filter}::{sub_topic_filter}::{skill_filter}::{difficulty_filter}",
                        disabled=True,
                    )
                    st.info("No questions yet for this selection. Try a different topic or check back later.")
                else:
                    df_filtered["topic_display"] = df_filtered["topic"].apply(_display_classification)
                    df_filtered["sub_topic_display"] = df_filtered.apply(
                        lambda row: _display_sub_topic(row.get("sub_topic"), row.get("sub_topic_raw")),
                        axis=1,
                    )
                    df_filtered["skill_display"] = df_filtered["skill"].apply(_display_classification)
                    df_filtered["difficulty_display"] = df_filtered["difficulty"].apply(_display_classification)
                    df_filtered["label"] = df_filtered.apply(
                        lambda r: (
                            f"{r['topic_display']} / {r['sub_topic_display']} / {r['skill_display']} / {r['difficulty_display']}"
                            f" | {r['question_label']} ({int(r['max_marks'])} marks) [{r.get('question_type','single')}] [id {int(r['id'])}]"
                        ),
                        axis=1,
                    )
                    choices = df_filtered["label"].tolist()
                    labels_map = dict(zip(df_filtered["label"], df_filtered["id"]))
                    id_sequence = [int(x) for x in df_filtered["id"].tolist()]
                    label_by_id = {int(row_id): label for row_id, label in zip(df_filtered["id"], df_filtered["label"])}

                    choice_key = (
                        f"student_question_choice::{source_filter}::{topic_filter}::{sub_topic_filter}"
                        f"::{skill_filter}::{difficulty_filter}::page_{page_index}"
                    )
                    if st.session_state.get(choice_key) not in choices:
                        st.session_state[choice_key] = choices[0]

                    choice = st.selectbox("Question", choices, key=choice_key)
                    chosen_id = int(labels_map.get(choice, 0)) if choice else 0
                    st.session_state["student_question_sequence"] = id_sequence
                    st.session_state["student_question_label_by_id"] = label_by_id
                    st.session_state["student_question_choice_key"] = choice_key

                    if chosen_id:
                        if st.session_state.get("selected_qid") != chosen_id:
                            st.session_state["selected_qid"] = chosen_id

                            q_row = load_question_by_id(chosen_id)
                            st.session_state["cached_q_row"] = q_row

                            st.session_state["cached_q_path"] = (q_row.get("question_image_path") or "").strip()
                            st.session_state["cached_ms_path"] = (q_row.get("markscheme_image_path") or "").strip()

                            q_path = (st.session_state.get("cached_q_path") or "").strip()
                            if q_path:
                                fp = (_safe_secret("SUPABASE_URL", "") or "")[:40]
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
        topic_label = _display_classification(_qr.get("topic"))
        sub_topic_label = _display_sub_topic(_qr.get("sub_topic"), _qr.get("sub_topic_raw"))
        skill_label = _display_classification(_qr.get("skill"))
        difficulty_label = _display_classification(_qr.get("difficulty"))
        st.caption(
            "Selected: "
            f"{topic_label} / {sub_topic_label} / {skill_label} / {difficulty_label} | {_qr.get('question_label', '')}"
        )

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
            st.subheader("Question")
            st.info("Select a question above to begin.")
        elif q_type != "journey":
            st.subheader("Question")
            max_marks = int(q_row.get("max_marks", 1))
            q_text = (q_row.get("question_text") or "").strip()

            with st.container(border=True):
                if question_img is not None:
                    st.image(question_img, caption="Question image", use_container_width=True)
                if q_text:
                    st.markdown(normalize_markdown_math(q_text))
                if (question_img is None) and (not q_text):
                    st.warning("This question has no question text or image.")
            question_sequence = st.session_state.get("student_question_sequence") or []
            label_by_id = st.session_state.get("student_question_label_by_id") or {}
            choice_key = st.session_state.get("student_question_choice_key")
            if qid and choice_key and len(question_sequence) > 1:
                try:
                    current_index = question_sequence.index(qid)
                except ValueError:
                    current_index = -1
                nav_disabled = current_index == -1
                nav_cols = st.columns([1, 1])
                with nav_cols[0]:
                    st.button(
                        "Previous Question",
                        disabled=nav_disabled,
                        use_container_width=True,
                        key="student_prev_question_btn",
                        on_click=_retreat_question,
                        args=(choice_key, question_sequence, label_by_id, qid),
                    )
                with nav_cols[1]:
                    st.button(
                        "Next Question",
                        disabled=nav_disabled,
                        use_container_width=True,
                        key="student_next_question_btn",
                        on_click=_advance_question,
                        args=(choice_key, question_sequence, label_by_id, qid),
                    )
            st.caption(f"Max Marks: {max_marks}")

            st.write("")
            st.markdown("**Answer in the box below.**")
            mode_row = st.columns([0.88, 0.12])
            with mode_row[0]:
                mode_single = st.radio(
                    "Answer mode",
                    ["Type answer", "Write answer"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="answer_mode_single",
                )
            with mode_row[1]:
                if str(mode_single).startswith("Type"):
                    text_expanded = bool(st.session_state.get("text_expanded_single", False))
                    if st.button(
                        "▲" if text_expanded else "▼",
                        help=("Collapse working area" if text_expanded else "Expand working area"),
                        key="text_expand_btn_single",
                    ):
                        st.session_state["text_expanded_single"] = not text_expanded
                        text_expanded = not text_expanded
                else:
                    canvas_expanded = bool(st.session_state.get("canvas_expanded_single", False))
                    if st.button(
                        "▲" if canvas_expanded else "▼",
                        help=("Collapse working area" if canvas_expanded else "Expand working area"),
                        key="canvas_expand_btn_single",
                    ):
                        st.session_state["canvas_expanded_single"] = not canvas_expanded
                        canvas_expanded = not canvas_expanded

            if str(mode_single).startswith("Type"):
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
                        st.toast("Please type an answer first.")
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
                            st.session_state["text_expanded_single"] = False

                            def task():
                                ms_path = (st.session_state.get("cached_ms_path") or q_row.get("markscheme_image_path") or "").strip()
                                ms_img = None
                                if ms_path:
                                    fp = (_safe_secret("SUPABASE_URL", "") or "")[:40]
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
                    f"panphy_canvas_h_{SUBJECT_SITE}_single_expanded_v2"
                    if canvas_expanded
                    else f"panphy_canvas_h_{SUBJECT_SITE}_single_v2"
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
                    undo_clicked = tool_row[2].button("Undo", use_container_width=True, key="canvas_undo_single")
                    clear_clicked = tool_row[3].button("Clear", use_container_width=True, key="canvas_clear_single")
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
                        min_height=CANVAS_HEIGHT_DEFAULT,
                        max_height=CANVAS_HEIGHT_EXPANDED,
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
                    clear_clicked = tool_row[1].button("Clear", use_container_width=True, key="canvas_clear_single")
                    if clear_clicked:
                        st.session_state["feedback"] = None
                        st.session_state["last_canvas_image_data_single"] = None
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
                        st.toast("Canvas is blank. Write your answer first, then press Submit.")
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

                    def task():
                        return get_gpt_feedback_from_bank(
                            student_answer=img_for_ai,
                            q_row=q_row,
                            is_student_image=True,
                            question_img=question_img,
                            markscheme_img=None,
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
            journey_obj = journey_obj or {}
            steps = journey_obj.get("steps", [])
            if not isinstance(steps, list):
                steps = []
            total_steps = len(steps)
            step_i = int(st.session_state.get("journey_step_index", 0) or 0)
            step_i = max(0, min(step_i, max(0, total_steps - 1)))
            st.session_state["journey_step_index"] = step_i

            st.subheader("Topic Journey")
            if total_steps == 0:
                st.warning("This journey has no steps.")
            else:
                st.caption(f"Step {step_i + 1} of {total_steps}")

                step_obj = steps[step_i] if step_i < total_steps else {}
                step_q_row = {
                    "question_text": step_obj.get("question_text", ""),
                    "markscheme_text": step_obj.get("markscheme_text", ""),
                    "max_marks": step_obj.get("max_marks", 1),
                    "question_type": "single",
                }

                step_qtext = (step_q_row.get("question_text") or "").strip()
                step_marks = int(step_q_row.get("max_marks", 1))

                if st.session_state.get("journey_show_steps_preview", False):
                    st.markdown("**All steps preview**")
                    with st.expander("Show all steps"):
                        for i, s in enumerate(steps):
                            st.markdown(f"**Step {i+1}:** {s.get('objective','').strip()}")
                            st.markdown(normalize_markdown_math(s.get("question_text", "")))
                            st.markdown(normalize_markdown_math(s.get("markscheme_text", "")))
                            st.divider()

                with st.container(border=True):
                    st.markdown(normalize_markdown_math(step_qtext or ""))
                st.caption(f"Max Marks: {step_marks}")

                question_sequence = st.session_state.get("student_question_sequence") or []
                label_by_id = st.session_state.get("student_question_label_by_id") or {}
                choice_key = st.session_state.get("student_question_choice_key")
                if qid and choice_key and len(question_sequence) > 1:
                    try:
                        current_index = question_sequence.index(qid)
                    except ValueError:
                        current_index = -1
                    nav_disabled = current_index == -1
                    nav_cols = st.columns([1, 1])
                    with nav_cols[0]:
                        st.button(
                            "Previous Question",
                            disabled=nav_disabled,
                            use_container_width=True,
                            key="student_prev_question_btn_journey",
                            on_click=_retreat_question,
                            args=(choice_key, question_sequence, label_by_id, qid),
                        )
                    with nav_cols[1]:
                        st.button(
                            "Next Question",
                            disabled=nav_disabled,
                            use_container_width=True,
                            key="student_next_question_btn_journey",
                            on_click=_advance_question,
                            args=(choice_key, question_sequence, label_by_id, qid),
                        )

                # Answer mode for journey steps
                mode_row = st.columns([0.88, 0.12])
                with mode_row[0]:
                    mode_journey = st.radio(
                        "Answer mode",
                        ["Type answer", "Write answer"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key="answer_mode_journey",
                    )
                with mode_row[1]:
                    if str(mode_journey).startswith("Type"):
                        text_expanded = bool(st.session_state.get("text_expanded_journey", False))
                        if st.button(
                            "▲" if text_expanded else "▼",
                            help=("Collapse working area" if text_expanded else "Expand working area"),
                            key="text_expand_btn_journey",
                        ):
                            st.session_state["text_expanded_journey"] = not text_expanded
                            text_expanded = not text_expanded
                    else:
                        canvas_expanded = bool(st.session_state.get("canvas_expanded_journey", False))
                        if st.button(
                            "▲" if canvas_expanded else "▼",
                            help=("Collapse working area" if canvas_expanded else "Expand working area"),
                            key="canvas_expand_btn_journey",
                        ):
                            st.session_state["canvas_expanded_journey"] = not canvas_expanded
                            canvas_expanded = not canvas_expanded

                def _update_checkpoint_notes(reports_list, idx, total_steps):
                    cp_every = journey_obj.get("checkpoint_every", JOURNEY_CHECKPOINT_EVERY) or JOURNEY_CHECKPOINT_EVERY
                    cp_every = max(1, int(cp_every))
                    # Always store at 0 and final step
                    is_checkpoint = (idx % cp_every == 0) or (idx == total_steps - 1)
                    if not is_checkpoint:
                        return
                    rep = reports_list[idx] if idx < len(reports_list) else None
                    if not rep:
                        return
                    notes = st.session_state.get("journey_checkpoint_notes", {}) or {}
                    note_md = rep.get("summary") or ""
                    fb = rep.get("feedback_points") or []
                    if fb:
                        note_md += "\n\n" + "\n".join([f"- {x}" for x in fb[:6]])
                    notes[str(idx)] = note_md.strip()
                    st.session_state["journey_checkpoint_notes"] = notes

                if str(mode_journey).startswith("Type"):
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
                            st.toast("Please type an answer first.")
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
                        f"panphy_canvas_h_{SUBJECT_SITE}_journey_expanded_v2"
                        if canvas_expanded
                        else f"panphy_canvas_h_{SUBJECT_SITE}_journey_v2"
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
                        undo_clicked = tool_row[2].button("Undo", use_container_width=True, key="canvas_undo_journey")
                        clear_clicked = tool_row[3].button("Clear", use_container_width=True, key="canvas_clear_journey")
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
                            min_height=CANVAS_HEIGHT_DEFAULT,
                            max_height=CANVAS_HEIGHT_EXPANDED,
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
                        clear_clicked = tool_row[1].button("Clear", use_container_width=True, key="canvas_clear_journey")
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
                            st.toast("Canvas is blank. Write your answer first, then press Submit.")
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
        st.subheader("Feedback")
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
