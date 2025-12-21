import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
import io, base64, json, pandas as pd, numpy as np
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Physics Examiner Pro", layout="wide")

# --- AUTHENTICATION HELPER ---
@st.cache_resource
def get_gspread_client():
    """
    Handles the robust parsing of Service Account JSON from st.secrets.
    Ensures the private_key is correctly formatted to avoid 'Invalid control character'.
    """
    try:
        # Access the dictionary directly from Streamlit secrets
        info = dict(st.secrets["connections"]["gsheets"]["service_account_info"])
        
        # FIX: Replace literal backslash-n with actual newlines in the private key
        if "private_key" in info:
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        creds = Credentials.from_service_account_info(info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Authentication Failed: {e}")
        return None

# Initialize Clients
gc = get_gspread_client()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- DATA & CONSTANTS ---
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

# --- CORE FUNCTIONS ---
def save_to_cloud(name, set_name, q_name, score, max_m, summary):
    """Appends student results to the linked Google Sheet."""
    if not gc:
        return False
    try:
        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        # Extract ID from URL
        s_id = url.split("/d/")[1].split("/")[0] if "/d/" in url else url
        sheet = gc.open_by_key(s_id).get_worksheet(0)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        sheet.append_row([timestamp, name, set_name, q_name, int(score), int(max_m), str(summary)])
        return True
    except Exception as e:
        st.error(f"⚠️ Save error: {e}")
        return False

# --- SESSION STATE ---
if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state: st.session_state["feedback"] = None

# --- UI LAYOUT ---
t1, t2 = st.tabs(["✍️ Student Portal", "📊 Teacher Dashboard"])

with t1:
    st.title("⚛️ Physics Examiner Pro")
    
    with st.expander("👤 Student Identity", expanded=True):
        c1, c2, c3 = st.columns(3)
        f_name = c1.text_input("First Name")
        l_name = c2.text_input("Last Name")
        cl_set = c3.selectbox("Physics Set", ["11Y/Ph1", "11X/Ph2", "Teacher Test"])
        full_name = f"{f_name} {l_name}"

    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        q_key = st.selectbox("Select Question", list(QUESTIONS.keys()))
        q_data = QUESTIONS[q_key]
        
        st.info(f"**Question:** {q_data['question']}")
        
        mode = st.radio("Response Method", ["Type Calculations", "Draw Diagram"], horizontal=True)
        
        if mode == "Type Calculations":
            ans = st.text_area("Show your working here:", height=200)
            if st.button("Submit Work", use_container_width=True):
                with st.spinner("Marking..."):
                    prompt = f"""Mark this GCSE Physics answer. 
                    Question: {q_data['question']}
                    Student Answer: {ans}
                    Mark Scheme: {q_data['mark_scheme']}
                    Total Marks Available: {q_data['marks']}
                    
                    Return ONLY a JSON object: {{"score": int, "summary": "string feedback"}}"""
                    
                    res = client.chat.completions.create(
                        model="gpt-4-turbo-preview", # GPT-5-Nano placeholder
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}
                    )
                    
                    feedback = json.loads(res.choices[0].message.content)
                    st.session_state["feedback"] = feedback
                    
                    # Save to Google Sheets
                    success = save_to_cloud(full_name, cl_set, q_key, feedback.get('score', 0), q_data['marks'], feedback.get('summary', ''))
                    if success: st.success("Results saved to Teacher Dashboard.")

        else:
            st.write("Draw your diagram below:")
            tool = st.toggle("Eraser Mode")
            canvas = st_canvas(
                stroke_width=15 if tool else 3,
                stroke_color="#000000",
                background_color="#ffffff",
                height=300,
                width=500,
                drawing_mode="freedraw",
                key=f"c_{st.session_state['canvas_key']}"
            )
            if st.button("Submit Drawing", use_container_width=True):
                st.warning("Vision analysis logic pending. Drawing received!")

    with col_r:
        st.subheader("Results & Feedback")
        if st.session_state["feedback"]:
            f = st.session_state["feedback"]
            st.metric("Score", f"{f.get('score', 0)} / {q_data['marks']}")
            st.markdown(f"**Examiner Note:**\n\n {f.get('summary', '')}")
        else:
            st.write("Submit your work to see feedback.")

with t2:
    st.header("📊 Teacher Results Dashboard")
    pwd = st.text_input("Enter Teacher Password", type="password")
    
    if pwd == "Newton2025":
        if gc:
            try:
                url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                s_id = url.split("/d/")[1].split("/")[0] if "/d/" in url else url
                rows = gc.open_by_key(s_id).get_worksheet(0).get_all_records()
                
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    
                    # Simple Analytics
                    st.divider()
                    st.subheader("Quick Stats")
                    avg_score = df['score'].mean() if 'score' in df.columns else 0
                    st.write(f"Class Average: **{avg_score:.1f} marks**")
                else:
                    st.info("No records found in the spreadsheet.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")
        else:
            st.error("Authentication not configured.")
