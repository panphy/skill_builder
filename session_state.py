import secrets as pysecrets

import streamlit as st


def _ss_init(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


def init_session_state() -> None:
    _ss_init("canvas_key", 0)
    _ss_init("feedback", None)
    _ss_init("student_answer_text_single", "")
    _ss_init("student_answer_text_journey", "")
    _ss_init("anon_id", pysecrets.token_hex(4))
    _ss_init("session_id", pysecrets.token_hex(6))
    _ss_init("db_last_error", "")
    _ss_init("db_table_ready", False)
    _ss_init("db_ai_timings_ready", False)
    _ss_init("bank_table_ready", False)
    _ss_init("rate_limits_table_ready", False)
    _ss_init("is_teacher", False)

    _ss_init("last_canvas_image_data", None)
    _ss_init("last_canvas_image_data_single", None)
    _ss_init("last_canvas_image_data_journey", None)
    _ss_init("last_canvas_data_url_single", None)
    _ss_init("last_canvas_data_url_journey", None)
    _ss_init("stylus_only_enabled", True)
    _ss_init("canvas_cmd_nonce_single", 0)
    _ss_init("canvas_cmd_nonce_journey", 0)

    _ss_init("selected_qid", None)
    _ss_init("cached_q_row", None)
    _ss_init("cached_question_img", None)
    _ss_init("cached_q_path", None)
    _ss_init("cached_ms_path", None)

    _ss_init("ai_draft", None)

    _ss_init("journey_step_index", 0)
    _ss_init("journey_step_reports", [])
    _ss_init("journey_checkpoint_notes", {})
    _ss_init("journey_active_id", None)
    _ss_init("journey_json_cache", None)
    _ss_init("journey_draft", None)
    _ss_init("journey_topics_selected", [])
    _ss_init("journey_gen_error_details", None)
    _ss_init("journey_show_error", False)
