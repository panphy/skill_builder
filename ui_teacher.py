import json
import secrets as pysecrets
from typing import Optional

import numpy as np
import streamlit as st

from ai_generation import AI_READY, JOURNEY_CHECKPOINT_EVERY, generate_practice_question_with_ai, generate_topic_journey_with_ai
from config import (
    DIFFICULTIES,
    QUESTION_TYPES,
    SKILLS,
    SUBJECT_SITE,
    get_topic_groups_for_track,
    get_topic_names_for_track,
    get_topic_track_ok,
)
from db import (
    delete_question_bank_by_id,
    ensure_question_bank_table,
    load_question_bank_df,
    load_question_by_id,
)


def render_teacher_page(nav_label: str, helpers: dict):
    db_ready = helpers["db_ready"]
    get_db_driver_type = helpers["get_db_driver_type"]
    ensure_attempts_table = helpers["ensure_attempts_table"]
    load_attempts_df = helpers["load_attempts_df"]
    delete_attempt_by_id = helpers["delete_attempt_by_id"]
    supabase_ready = helpers["supabase_ready"]
    cached_download_from_storage = helpers["cached_download_from_storage"]
    safe_bytes_to_pil = helpers["safe_bytes_to_pil"]
    normalize_markdown_math = helpers["normalize_markdown_math"]
    render_md_box = helpers["render_md_box"]
    slugify = helpers["slugify"]
    validate_image_file = helpers["validate_image_file"]
    _compress_bytes_to_limit = helpers["_compress_bytes_to_limit"]
    upload_to_storage = helpers["upload_to_storage"]
    _run_ai_with_progress = helpers["_run_ai_with_progress"]
    insert_question_bank_row = helpers["insert_question_bank_row"]
    QUESTION_MAX_MB = helpers["QUESTION_MAX_MB"]
    MARKSCHEME_MAX_MB = helpers["MARKSCHEME_MAX_MB"]
    track = st.session_state.get("track", "combined")

    def _index_for(options, value):
        if not options:
            return 0
        try:
            return options.index(value)
        except ValueError:
            return 0

    def _validate_classification_inputs(
        topic_value,
        sub_topic_value,
        skill_value,
        difficulty_value,
        topic_options,
        sub_topic_options,
        skill_options,
        difficulty_options,
    ):
        if not topic_value or topic_value not in topic_options:
            st.error("Please select a valid topic.")
            return False
        if not sub_topic_value or sub_topic_value not in sub_topic_options:
            st.error("Please select a valid sub-topic.")
            return False
        if not skill_value or skill_value not in skill_options:
            st.error("Please select a valid skill.")
            return False
        if not difficulty_value or difficulty_value not in difficulty_options:
            st.error("Please select a valid difficulty.")
            return False
        return True

    if nav_label == "🔒 Teacher Dashboard":
        st.divider()
        st.subheader("🔒 Teacher Dashboard")

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
                    st.dataframe(by_student, width='stretch')

                    st.write("### By topic (overall %)")
                    by_topic = (
                        df.groupby("question_key")[["marks_awarded", "max_marks"]]
                        .sum()
                        .assign(percent=lambda x: (100 * x["marks_awarded"] / x["max_marks"].replace(0, np.nan)).round(1))
                        .sort_values("percent", ascending=False)
                    )
                    st.dataframe(by_topic, width='stretch')

                    st.write("### Recent attempts")
                    st.dataframe(df.head(50), width='stretch')

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
                            delete_status: Optional[str] = None
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
                                width='stretch',
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
        return

    st.divider()
    st.subheader("📚 Question Bank")

    # Default track eligibility tag for any question you SAVE (AI drafts, edited, uploads).
    _tt_label = st.selectbox(
        "Default eligibility for saved questions",
        ["Both (Combined + Separate)", "Separate only"],
        index=0,
        key="teacher_track_ok_label",
        help="Sets the default visibility for newly saved items. Choose 'Separate only' to hide them from Combined students.",
    )
    st.session_state["teacher_track_ok"] = "both" if _tt_label.startswith("Both") else "separate_only"


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
                    else:
                        df_f = df_f.iloc[0:0]
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

                        with st.expander("Select an entry to preview", expanded=False):
                            pick = st.selectbox("Question entry", options, key="bank_preview_pick")
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
                                        st.image(q_img, width='stretch')
                                    if q_text:
                                        st.markdown(normalize_markdown_math(q_text))
                                    if (q_img is None) and (not q_text):
                                        st.caption("No question text/image (image-only teacher uploads are supported).")

                            with pv2:
                                st.markdown("**Mark scheme (teacher only)**")
                                with st.container(border=True):
                                    if ms_img is not None:
                                        st.image(ms_img, width='stretch')
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
                                width='stretch',
                                disabled=not (confirm_delete_q and delete_picks),
                                key="delete_bank_btn",
                            ):
                                st.session_state["bank_delete_requested"] = True

                            if st.session_state.get("bank_delete_requested"):
                                delete_ok = True
                                if delete_picks:
                                    for pick_label in delete_picks:
                                        try:
                                            qid = int(df_f.loc[df_f["label"] == pick_label, "id"].iloc[0])
                                        except Exception:
                                            qid = None
                                        if qid is not None:
                                            delete_ok = delete_question_bank_by_id(qid) and delete_ok
                                if delete_ok and delete_picks:
                                    st.success("Deleted selected entries.")
                                    st.session_state["bank_delete_reset"] = True
                                    st.session_state["bank_delete_requested"] = False
                                    st.rerun()
                                elif delete_picks:
                                    st.error("Delete failed. Check database errors below.")
                                    st.session_state["bank_delete_requested"] = False
                                else:
                                    st.warning("Select at least one entry.")
                                    st.session_state["bank_delete_requested"] = False

                st.divider()

            # -------------------------
            # AI generator
            # -------------------------
            with tab_ai:
                st.write("## 🤖 AI Generator")
                st.caption("Generate and review before saving to the bank.")

                c1, c2, c3, c4 = st.columns([3, 2, 2, 1])

                with c1:
                    topic = st.selectbox(
                        "Topic",
                        get_topic_names_for_track(track),
                        key="gen_topic",
                    )
                with c2:
                    qtype = st.selectbox("Question type", QUESTION_TYPES, key="gen_qtype")
                with c3:
                    difficulty = st.selectbox("Difficulty", DIFFICULTIES, key="gen_difficulty")
                with c4:
                    marks = st.number_input("Marks", min_value=1, max_value=12, value=4, step=1, key="gen_marks")

                extra = st.text_area(
                    "Optional extra instructions (for the AI)",
                    height=80,
                    placeholder="e.g. include one tricky unit conversion, use g=9.8",
                    key="gen_extra",
                )

                col_gen1, col_gen2 = st.columns([1, 1])
                with col_gen1:
                    gen_clicked = st.button("Generate draft", type="primary", width='stretch', disabled=not AI_READY, key="gen_btn")
                with col_gen2:
                    if st.button("Clear draft", width='stretch', key="gen_clear_btn"):
                        st.session_state["draft_question"] = None
                        st.session_state["draft_warning"] = None
                        st.rerun()

                if gen_clicked:
                    with st.spinner("Generating question..."):
                        try:
                            data = generate_practice_question_with_ai(
                                topic_text=topic,
                                difficulty=difficulty,
                                qtype=qtype,
                                marks=int(marks),
                                extra_instructions=extra,
                            )

                            draft = {
                                "topic": data.get("topic", topic),
                                "sub_topic": data.get("sub_topic"),
                                "skill": data.get("skill"),
                                "difficulty": data.get("difficulty", difficulty),
                                "question_type": "single",
                                "question_text": data.get("question_text", ""),
                                "markscheme_text": data.get("markscheme_text", ""),
                                "max_marks": data.get("max_marks", int(marks)),
                                "tags": data.get("tags", []),
                                "warnings": data.get("warnings", []),
                            }

                            st.session_state["draft_question"] = draft
                            st.session_state["draft_warning"] = None
                        except Exception as e:
                            st.session_state["draft_question"] = None
                            st.session_state["draft_warning"] = str(e)

                draft = st.session_state.get("draft_question")
                if st.session_state.get("draft_warning"):
                    st.error(f"AI generation failed: {st.session_state['draft_warning']}")

                if draft:
                    if draft.get("warnings"):
                        st.warning("Draft warnings:\n\n" + "\n".join([f"- {w}" for w in draft.get("warnings", [])]))

                    st.write("### ✅ Vet and edit")
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        assignment_name = st.text_input("Assignment name", value="AI Practice", key="draft_assignment")
                        question_label = st.text_input("Question label", value=f"{slugify(topic)[:24]}-{pysecrets.token_hex(3)}", key="draft_label")
                        tags_str = st.text_input("Tags (comma separated)", value=", ".join(draft.get("tags", [])), key="draft_tags")
                    with c2:
                        save_clicked = st.button("Save to Question Bank", type="primary", width='stretch', key="draft_save_btn")

                    topic_options = get_topic_names_for_track(track)
                    sub_topic_options = get_topic_groups_for_track(track)
                    skill_options = list(SKILLS)
                    difficulty_options = list(DIFFICULTIES)

                    tc1, tc2 = st.columns(2)
                    with tc1:
                        topic_val = st.selectbox(
                            "Topic",
                            topic_options,
                            index=_index_for(topic_options, draft.get("topic")),
                            key="draft_topic",
                        )
                        skill_val = st.selectbox(
                            "Skill",
                            skill_options,
                            index=_index_for(skill_options, draft.get("skill")),
                            key="draft_skill",
                        )
                    with tc2:
                        sub_topic_val = st.selectbox(
                            "Sub-topic",
                            sub_topic_options,
                            index=_index_for(sub_topic_options, draft.get("sub_topic")),
                            key="draft_sub_topic",
                        )
                        difficulty_val = st.selectbox(
                            "Difficulty",
                            difficulty_options,
                            index=_index_for(difficulty_options, draft.get("difficulty")),
                            key="draft_difficulty",
                        )

                    q_text = st.text_area("Question text", value=draft.get("question_text", ""), height=200, key="draft_q")
                    ms_text = st.text_area("Mark scheme", value=draft.get("markscheme_text", ""), height=240, key="draft_ms")
                    max_marks = st.number_input("Max marks", min_value=1, max_value=12, value=int(draft.get("max_marks", 4) or 4), step=1, key="draft_mm")

                    render_md_box("Preview: Question", q_text, empty_text="No question text.")
                    render_md_box("Preview: Mark scheme", ms_text, empty_text="No mark scheme.")

                    if save_clicked:
                        if not assignment_name.strip() or not question_label.strip():
                            st.error("Assignment name and question label are required.")
                        elif not _validate_classification_inputs(
                            topic_val,
                            sub_topic_val,
                            skill_val,
                            difficulty_val,
                            topic_options,
                            sub_topic_options,
                            skill_options,
                            difficulty_options,
                        ):
                            pass
                        else:
                            tags = [t.strip() for t in (tags_str or "").split(",") if t.strip()]
                            ok = insert_question_bank_row(
                                source="ai_generated",
                                created_by="teacher",
                                subject_site=SUBJECT_SITE,
                                track_ok=st.session_state.get("teacher_track_ok", "both"),
                                assignment_name=assignment_name.strip(),
                                question_label=question_label.strip(),
                                max_marks=int(max_marks),
                                tags=tags,
                                topic=topic_val,
                                sub_topic=sub_topic_val,
                                skill=skill_val,
                                difficulty=difficulty_val,
                                question_text=(q_text or "").strip(),
                                markscheme_text=(ms_text or "").strip(),
                                question_image_path=None,
                                markscheme_image_path=None,
                            )
                            if ok:
                                st.success("Approved and saved. Students can now access this under AI Practice.")
                            else:
                                st.error("Save failed. Check database errors below.")

                st.divider()

                st.write("### 🧭 Topic Journey")
                st.caption("Create a multi-step journey (fixed at 10 minutes / 5 steps).")

                jc1, jc2 = st.columns([3, 1])

                # --- Left column: inputs ---
                with jc1:
                    st.multiselect(
                        "Select one topic (you may pick several, but only the first is used to generate)",
                        get_topic_names_for_track(track),
                        key="journey_topics_selected",
                        max_selections=1,
                    )
                    sel_topics = list(st.session_state.get("journey_topics_selected", []) or [])
                    if sel_topics:
                        st.markdown("**Selected topic:**")
                        st.markdown(f"- {sel_topics[0]}")
                    else:
                        st.info("Choose a topic to build a journey.")

                    # Fixed journey size: 10 minutes, 5 steps
                    j_duration = 10
                    st.caption("Journey length is fixed: 10 minutes, 5 steps.")

                    j_assignment = st.text_input("Assignment name for saving", value="Topic Journey", key="jour_assignment")
                    j_tags = st.text_input("Tags (comma separated)", value="", key="jour_tags")
                    j_extra_instr = st.text_area(
                        "Optional constraints for the AI",
                        height=80,
                        placeholder="e.g. Include one tricky unit conversion. Use g = 9.8 N/kg. Require a final answer with units.",
                        key="jour_extra"
                    )

                # --- Right column: actions ---
                with jc2:
                    gen_j = st.button(
                        "Generate journey draft",
                        type="primary",
                        width='stretch',
                        disabled=not AI_READY,
                        key="jour_gen_btn",
                    )

                    if st.button("Clear journey draft", width='stretch', key="jour_clear_btn"):
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
                                    extra_instructions=j_extra_instr or "",
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
                        if st.button("Explain error", key="jour_explain_error", width='stretch'):
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
                        save_j = st.button("Save Topic Journey to bank", type="primary", width='stretch', key="jour_save_btn")
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
                st.caption("Optional question text supports Markdown and LaTeX.")

                with st.form("upload_q_form", clear_on_submit=True):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        assignment_name = st.text_input("Assignment name", placeholder="e.g. AQA Paper 1 (Electricity)", key="up_assignment")
                        question_label = st.text_input("Question label", placeholder="e.g. Q3b", key="up_label")
                    with c2:
                        max_marks_in = st.number_input("Max marks", min_value=1, max_value=50, value=3, step=1, key="up_marks")

                    topic_options = get_topic_names_for_track(track)
                    sub_topic_options = get_topic_groups_for_track(track)
                    skill_options = list(SKILLS)
                    difficulty_options = list(DIFFICULTIES)

                    uc1, uc2 = st.columns(2)
                    with uc1:
                        topic_val = st.selectbox(
                            "Topic",
                            topic_options,
                            index=0,
                            key="up_topic",
                        )
                        skill_val = st.selectbox(
                            "Skill",
                            skill_options,
                            index=0,
                            key="up_skill",
                        )
                    with uc2:
                        sub_topic_val = st.selectbox(
                            "Sub-topic",
                            sub_topic_options,
                            index=0,
                            key="up_sub_topic",
                        )
                        difficulty_val = st.selectbox(
                            "Difficulty",
                            difficulty_options,
                            index=0,
                            key="up_difficulty",
                        )

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
                    elif not _validate_classification_inputs(
                        topic_val,
                        sub_topic_val,
                        skill_val,
                        difficulty_val,
                        topic_options,
                        sub_topic_options,
                        skill_options,
                        difficulty_options,
                    ):
                        pass
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
                                topic=topic_val,
                                sub_topic=sub_topic_val,
                                skill=skill_val,
                                difficulty=difficulty_val,
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
