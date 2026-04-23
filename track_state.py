import json

import streamlit as st

from config import SUBJECT_SITE


TRACK_PARAM = "track"
TRACK_DEFAULT = "combined"
TRACK_ALLOWED = {"combined", "separate"}
TRACK_STORAGE_KEY = f"panphy_track_{SUBJECT_SITE}"


def _get_query_param(key: str) -> str:
    try:
        value = st.query_params.get(key)
        if isinstance(value, list):
            return (value[0] or "").strip()
        return (value or "").strip()
    except Exception:
        qp = st.experimental_get_query_params()
        value = qp.get(key, [""])[0]
        return (value or "").strip()


def set_query_param(**kwargs) -> None:
    try:
        for key, value in kwargs.items():
            st.query_params[key] = value
    except Exception:
        st.experimental_set_query_params(**kwargs)


def _inject_track_restore_script() -> None:
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
        unsafe_allow_html=True,
    )


def persist_track_to_browser(track_value: str) -> None:
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
        unsafe_allow_html=True,
    )


def init_track_state() -> None:
    if "track_init_done" not in st.session_state:
        st.session_state["track_init_done"] = True
        _inject_track_restore_script()

    qp_track = _get_query_param(TRACK_PARAM).lower()
    if qp_track not in TRACK_ALLOWED:
        qp_track = TRACK_DEFAULT

    if st.session_state.get("track") not in TRACK_ALLOWED:
        st.session_state["track"] = qp_track

    if _get_query_param(TRACK_PARAM).lower() != st.session_state["track"]:
        set_query_param(**{TRACK_PARAM: st.session_state["track"]})
