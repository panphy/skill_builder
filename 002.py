import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
import io, base64, json, pandas as pd, numpy as np
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Physics Examiner Pro", layout="wide")

# --- AUTHENTICATION HELPER ---
@st.cache_resource
def get_gspread_client():
    """Fixes 'Invalid control character' and 'dictionary length' errors."""
    try:
        raw_info = st.secrets["connections"]["gsheets"]["service_account_info"]
        
        # Parse based on whether secrets are stored as a string or TOML table
        if isinstance(raw_info, str):
            info = json.loads(raw_info, strict=False)
        else:
            info = dict(raw_info)
        
        # Sanitize the private key for Google RSA
        if "private_key" in info:
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Authentication Failed: {e}")
        return None

# Initialize Clients
gc = get_gspread_client()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- DATA ---
QUESTIONS = {
    "Q1: Forces": {
        "question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate acceleration.", 
        "marks": 3, 
        "mark_scheme": "1. Resultant Force = 16N. 2. Use F=ma. 3. a = 3.2 m/s²."
    },
    "Q2: Refraction": {
        "question": "Draw a ray diagram showing light traveling from air into a glass block.", 
        "marks": 2, 
        "mark_scheme": "1. Ray bends towards the normal in glass. 2. Correct labels for Incident and Refracted rays."
    }
}

# --- IMPROVED SAVE LOGIC ---
def save_to_cloud(name, set_name, q_name, score, max_m, summary):
    if not gc: 
        st.error("GSpread client not initialized.")
        return False
    try:
        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        # Improved ID extraction
        if "/d/" in url:
            s_id = url.split("/d/")[1].split("/")[0]
        else:
            s_id = url
            
        sheet = gc.open_by_key(s_id).get_worksheet(0)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        # Ensure values are simple types for JSON serialisation
        row = [timestamp, str(name), str(set_name), str(q_name), int(score), int(max_m), str(summary)]
        
        sheet.append_row(row)
        return True
    except gspread.exceptions.APIError as e:
        st.error(f"⚠️ Google API Error: {e}. Check if you shared the sheet with the Service Account email!")
        return False
    except Exception as e:
        st.error(f"⚠️ Save error: {e}")
        return False

# --- UI LOGIC ---
if "feedback" not in st.session_state: st.session_state["feedback"] = None

t1, t2 = st.tabs(["✍️ Student Portal", "📊 Teacher Dashboard"])

with t1:
    st.title("⚛️ Physics Examiner Pro")
    
    with st.expander("👤 Student Identity", expanded=True):
        c1, c2, c3 = st.columns(3)
        f_name = c1.text_input('First Name')
        l_name = c2.text_input('Last Name')
        cl_set = c3.selectbox("Set", ["11Y/Ph1", "11X/Ph2", "Teacher Test"])
        full_name = f"{f_name} {l_name}"

    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        q_key = st.selectbox("Select Question", list(QUESTIONS.keys()))
        q_data = QUESTIONS[q_key]
        st.info(f"**Question:** {q_data['question']}")
        
        mode = st.radio("Method", ["Type Calculations", "Draw Diagram"], horizontal=True)
        
        if mode == "Type Calculations":
            ans = st.text_area("Show your working:", height=150)
            if st.button("Submit Work", use_container_width=True):
                with st.spinner("AI Marking..."):
                    res = client.chat.completions.create(
                        model="gpt-4o", 
                        messages=[{"role": "user", "content": f"Mark: {ans}. Scheme: {q_data['mark_scheme']}. Total Marks: {q_data['marks']}. JSON format only: {{'score': int, 'summary': str}}"}],
                        response_format={"type": "json_object"}
                    )
                    data = json.loads(res.choices[0].message.content)
                    st.session_state["feedback"] = data
                    save_to_cloud(full_name, cl_set, q_key, data.get('score', 0), q_data['marks'], data.get('summary', ''))

        else:
            canvas = st_canvas(stroke_width=3, stroke_color="#000", background_color="#fff", height=300, width=500, drawing_mode="freedraw", key="canvas_main")
            if st.button("Submit Drawing", use_container_width=True):
                if canvas.image_data is not None:
                    with st.spinner("Analyzing Diagram..."):
                        img = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA').convert('RGB')
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()

                        res = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"Mark this GCSE diagram. Scheme: {q_data['mark_scheme']}. Total Marks: {q_data['marks']}. Return JSON: {{'score': int, 'summary': str}}"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                                ]
                            }],
                            response_format={"type": "json_object"}
                        )
                        data = json.loads(res.choices[0].message.content)
                        st.session_state["feedback"] = data
                        save_to_cloud(full_name, cl_set, q_key, data.get('score', 0), q_data['marks'], data.get('summary', ''))

    with col_r:
        st.subheader("Results")
        if st.session_state["feedback"]:
            f = st.session_state["feedback"]
            st.metric("Score", f"{f.get('score', 0)} / {q_data['marks']}")
            st.write(f.get('summary', ''))

with t2:
    if st.text_input("Teacher Password", type="password") == "Newton2025":
        if gc:
            try:
                url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                s_id = url.split("/d/")[1].split("/")[0] if "/d/" in url else url
                rows = gc.open_by_key(s_id).get_worksheet(0).get_all_records()
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"Could not load dashboard: {e}")
