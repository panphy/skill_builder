import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
import io, base64, json, pandas as pd, numpy as np
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="AI Physics Examiner (Pro)", page_icon="⚛️", layout="wide")

# --- CUSTOM GOOGLE CONNECTION ---
@st.cache_resource
def get_gspread_client():
    try:
        # Pulls the exact JSON string from your secrets
        info = json.loads(st.secrets["connections"]["gsheets"]["service_account_info"])
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        return None

def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize
gc = get_gspread_client()
client = get_openai_client()

# --- SESSION STATE ---
if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state: st.session_state["feedback"] = None

# --- DATA ---
QUESTIONS = {
    "Q1: Forces": {"question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate acceleration.", "marks": 3, "mark_scheme": "1. Resultant=16N. 2. F=ma. 3. a=3.2m/s²."},
    "Q2: Refraction": {"question": "Draw a ray diagram: air to glass block.", "marks": 2, "mark_scheme": "1. Bends toward normal. 2. Correct labels."}
}
CLASS_SETS = ["11Y/Ph1", "11X/Ph2", "Teacher Test"]

# --- STORAGE LOGIC (REWRITTEN) ---
def save_to_cloud(name, set_name, q_name, score, max_m, summary):
    if gc is None:
        st.error("Cannot save: Google Client not initialized.")
        return False
    try:
        # Open by the ID found in your secrets
        sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        # Extract ID from URL if necessary
        sheet_id = sheet_url.split("/d/")[1].split("/")[0] if "/d/" in sheet_url else sheet_url
        
        spreadsheet = gc.open_by_key(sheet_id)
        worksheet = spreadsheet.get_worksheet(0) # First tab
        
        new_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            name, set_name, q_name, int(score), int(max_m), str(summary)
        ]
        worksheet.append_row(new_row)
        return True
    except Exception as e:
        st.error(f"Cloud Save Failed: {e}")
        return False

# --- GPT LOGIC ---
def get_gpt_feedback(answer, q_data, is_image=False):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "marking_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score_awarded": {"type": "integer"},
                    "summary": {"type": "string"}
                },
                "required": ["score_awarded", "summary"],
                "additionalProperties": False
            }
        }
    }
    
    messages = [{"role": "system", "content": f"Mark strictly. Scheme: {q_data['mark_scheme']}"}]
    if is_image:
        buffered = io.BytesIO()
        answer.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        messages.append({"role": "user", "content": [{"type": "text", "text": "Mark drawing."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]})
    else:
        messages.append({"role": "user", "content": f"Student Answer: {answer}"})

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        max_completion_tokens=3000,
        reasoning_effort="minimal",
        response_format=schema
    )
    return json.loads(response.choices[0].message.content)

# --- UI ---
tab1, tab2 = st.tabs(["✍️ Student Portal", "📊 Teacher Dashboard"])

with tab1:
    st.title("⚛️ Physics Examiner")
    
    with st.container(border=True):
        st.subheader("1. Details")
        c1, c2, c3 = st.columns(3)
        fname = c1.text_input("First Name")
        lname = c2.text_input("Last Name")
        student_set = c3.selectbox("Set", CLASS_SETS)

    col_l, col_r = st.columns(2)
    with col_l:
        q_key = st.selectbox("Question", list(QUESTIONS.keys()))
        q_data = QUESTIONS[q_key]
        st.info(q_data['question'])
        
        mode = st.radio("Mode", ["Type", "Draw"], horizontal=True)
        if mode == "Type":
            ans = st.text_area("Answer:")
            if st.button("Submit"):
                with st.spinner("Marking..."):
                    res = get_gpt_feedback(ans, q_data)
                    st.session_state["feedback"] = res
                    save_to_cloud(f"{fname} {lname}", student_set, q_key, res['score_awarded'], q_data['marks'], res['summary'])
        else:
            # Drawing tools
            t1, t2 = st.columns(2)
            with t1: tool = st.toggle("Eraser")
            with t2: 
                if st.button("Clear"):
                    st.session_state["canvas_key"] += 1
                    st.rerun()
            
            canvas = st_canvas(
                stroke_width=20 if tool else 2, stroke_color="#f8f9fa" if tool else "#000",
                background_color="#f8f9fa", height=300, width=450, key=f"c_{st.session_state['canvas_key']}"
            )
            if st.button("Submit Drawing"):
                raw = Image.fromarray(canvas.image_data.astype('uint8'))
                white_bg = Image.new("RGB", raw.size, (255, 255, 255))
                white_bg.paste(raw, mask=raw.split()[3])
                with st.spinner("Analyzing..."):
                    res = get_gpt_feedback(white_bg, q_data, is_image=True)
                    st.session_state["feedback"] = res
                    save_to_cloud(f"{fname} {lname}", student_set, q_key, res['score_awarded'], q_data['marks'], res['summary'])

    with col_r:
        if st.session_state["feedback"]:
            res = st.session_state["feedback"]
            st.metric("Score", f"{res['score_awarded']} / {q_data['marks']}")
            st.success(res['summary'])

with tab2:
    if st.text_input("Password", type="password") == "Newton2025":
        if gc:
            try:
                sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                sheet_id = sheet_url.split("/d/")[1].split("/")[0] if "/d/" in sheet_url else sheet_url
                data = gc.open_by_key(sheet_id).get_worksheet(0).get_all_records()
                st.dataframe(pd.DataFrame(data))
            except Exception as e:
                st.error(f"Dashboard Error: {e}")
