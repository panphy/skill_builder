import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
import gspread
from google.oauth2.service_account import Credentials
import json, re
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Physics Examiner Pro", layout="wide")

# =========================
# CONFIG + CONSTANTS
# =========================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

QUESTIONS = {
    "Q1: Forces": {
        "question": "5kg box, 20N force, 4N friction. Acceleration?",
        "marks": 3,
        "mark_scheme": "1. F=16N. 2. F=ma. 3. a=3.2m/s².",
    },
    "Q2: Refraction": {
        "question": "Draw ray diagram: air to glass block.",
        "marks": 2,
        "mark_scheme": "1. Bends to normal. 2. Labels.",
    },
}

# =========================
# GOOGLE SHEETS AUTH
# =========================

def _extract_sheet_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", s)
    return m.group(1) if m else s

def _coerce_service_account_info(raw):
    """
    Accepts either:
      - dict (recommended Streamlit secrets TOML style)
      - JSON string

    Repairs common private_key newline issues:
      - literal newlines inside the JSON string value
      - \\n sequences that need to become real newlines
    """
    if isinstance(raw, dict):
        info = dict(raw)

    elif isinstance(raw, str):
        s = raw.strip()

        # First attempt: strict JSON
        try:
            info = json.loads(s)
        except json.JSONDecodeError:
            # Second attempt: permissive parsing
            try:
                info = json.loads(s, strict=False)
            except json.JSONDecodeError:
                # Third attempt: only escape literal newlines inside "private_key"
                m = re.search(r'("private_key"\s*:\s*")(.*?)(\")', s, flags=re.S)
                if not m:
                    raise

                pk_raw = m.group(2)
                pk_fixed_for_json = pk_raw.replace("\r", "").replace("\n", "\\n")
                s_fixed = s[:m.start(2)] + pk_fixed_for_json + s[m.end(2):]
                info = json.loads(s_fixed)
    else:
        raise TypeError(f"service_account_info must be dict or str, got {type(raw)}")

    # Normalize private_key to real newlines for google-auth
    if "private_key" in info and isinstance(info["private_key"], str):
        info["private_key"] = info["private_key"].replace("\\n", "\n").replace("\r", "")

    return info

@st.cache_resource(show_spinner=False)
def get_gspread_client():
    """
    Robust gspread client creation from Streamlit Secrets.

    Supports either:
      A) st.secrets["connections"]["gsheets"]["service_account_info"] as dict
      B) same key as JSON string
      C) storing individual SA fields under [connections.gsheets] (fallback)
    """
    try:
        gs = st.secrets["connections"]["gsheets"]

        if "service_account_info" in gs:
            info = _coerce_service_account_info(gs["service_account_info"])
        else:
            # Fallback: build dict from individual keys
            info = {k: gs[k] for k in gs.keys() if k != "spreadsheet"}
            if "private_key" in info and isinstance(info["private_key"], str):
                info["private_key"] = info["private_key"].replace("\\n", "\n").replace("\r", "")

        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return gspread.authorize(creds)

    except Exception as e:
        st.error(f"❌ Google Sheets auth failed: {e}")
        st.info(
            "Checklist:\n"
            "1) Your spreadsheet must be shared with the service account client_email (Editor).\n"
            "2) The private_key must retain line breaks (either \\n or real newlines).\n"
            "3) If using a JSON string in secrets, ensure it is valid JSON."
        )
        return None

# Initialize clients
gc = get_gspread_client()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================
# DATA WRITE
# =========================

def save_to_cloud(name, set_name, q_name, score, max_m, summary):
    if not gc:
        return False

    try:
        sheet_ref = st.secrets["connections"]["gsheets"]["spreadsheet"]
        s_id = _extract_sheet_id(sheet_ref)

        ws = gc.open_by_key(s_id).get_worksheet(0)
        ws.append_row(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                name,
                set_name,
                q_name,
                int(score),
                int(max_m),
                str(summary),
            ],
            value_input_option="USER_ENTERED",
        )
        return True

    except Exception as e:
        st.error(f"Save error: {e}")
        st.info(
            "If this looks like a permission error, share the spreadsheet with the service account client_email "
            "and grant Editor access."
        )
        return False

# =========================
# SESSION STATE
# =========================

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state:
    st.session_state["feedback"] = None

# =========================
# UI
# =========================

t1, t2 = st.tabs(["✍️ Student Portal", "📊 Teacher Dashboard"])

with t1:
    st.title("⚛️ Physics Examiner")

    with st.expander("👤 Identity", expanded=True):
        c1, c2, c3 = st.columns(3)
        first = c1.text_input("First")
        last = c2.text_input("Last")
        cl_set = c3.selectbox("Set", ["11Y/Ph1", "11X/Ph2", "Teacher Test"])

        name = f"{first.strip()} {last.strip()}".strip()
        if not name:
            st.warning("Please enter your first and last name before submitting.")

    col_l, col_r = st.columns(2)

    with col_l:
        q_key = st.selectbox("Question", list(QUESTIONS.keys()))
        q_data = QUESTIONS[q_key]
        st.info(q_data["question"])

        mode = st.radio("Mode", ["Type", "Draw"], horizontal=True)

        if mode == "Type":
            ans = st.text_area("Working:")

            if st.button("Submit", use_container_width=True, disabled=(not name)):
                with st.spinner("Marking..."):
                    try:
                        res = client.chat.completions.create(
                            model="gpt-5-nano",
                            messages=[
                                {
                                    "role": "user",
                                    "content": (
                                        "You are a GCSE Physics examiner.\n"
                                        f"Mark the student's answer out of {q_data['marks']} using the mark scheme.\n\n"
                                        f"Question: {q_data['question']}\n"
                                        f"Mark scheme: {q_data['mark_scheme']}\n\n"
                                        f"Student answer: {ans}\n\n"
                                        "Return JSON only with keys: score (int), summary (string)."
                                    ),
                                }
                            ],
                            response_format={"type": "json_object"},
                        )

                        raw = res.choices[0].message.content or "{}"
                        data = json.loads(raw)

                        # Guardrails
                        score = int(data.get("score", 0))
                        score = max(0, min(score, int(q_data["marks"])))
                        summary = str(data.get("summary", "")).strip()

                        st.session_state["feedback"] = {"score": score, "summary": summary}

                        ok = save_to_cloud(name, cl_set, q_key, score, q_data["marks"], summary)
                        if ok:
                            st.success("Saved to cloud.")
                        else:
                            st.warning("Not saved to cloud.")

                    except Exception as e:
                        st.error(f"Marking error: {e}")

        else:
            tool = st.toggle("Eraser")
            canvas = st_canvas(
                stroke_width=20 if tool else 2,
                stroke_color="#f8f9fa" if tool else "#000000",
                background_color="#f8f9fa",
                height=300,
                width=400,
                key=f"c_{st.session_state['canvas_key']}",
            )

            if st.button("Submit Drawing", use_container_width=True, disabled=(not name)):
                st.info("Drawing submission received. Add vision logic here when ready.")
                # You can still write a placeholder row if you want:
                ok = save_to_cloud(name, cl_set, q_key, 0, q_data["marks"], "Drawing submitted (vision marking not enabled).")
                if ok:
                    st.success("Saved to cloud.")
                else:
                    st.warning("Not saved to cloud.")

    with col_r:
        st.subheader("Feedback")
        if st.session_state["feedback"]:
            f = st.session_state["feedback"]
            st.metric("Score", f.get("score", 0), delta=None)
            st.write(f.get("summary", ""))
        else:
            st.caption("Submit an answer to see feedback here.")

with t2:
    st.subheader("Teacher Dashboard")

    pw = st.text_input("Password", type="password")
    if pw == "Newton2025":
        if not gc:
            st.error("Google Sheets is not authenticated, so the dashboard cannot load data.")
        else:
            try:
                sheet_ref = st.secrets["connections"]["gsheets"]["spreadsheet"]
                s_id = _extract_sheet_id(sheet_ref)

                ws = gc.open_by_key(s_id).get_worksheet(0)
                rows = ws.get_all_records()

                df = pd.DataFrame(rows)
                if df.empty:
                    st.info("No records yet.")
                else:
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Dashboard load error: {e}")
                st.info(
                    "If this is a permission error, share the spreadsheet with the service account client_email "
                    "and grant Editor access."
                )
    elif pw:
        st.error("Incorrect password.")