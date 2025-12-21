import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from streamlit_gsheets import GSheetsConnection
import io, base64, json, pandas as pd, numpy as np
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="AI Physics Examiner", page_icon="⚛️", layout="wide")

# --- CONNECTIONS ---
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

try:
    client = get_openai_client()
    # Explicitly connect to Google Sheets using the secrets
    conn = st.connection("gsheets", type=GSheetsConnection)
    AI_READY = True
except Exception as e:
    st.sidebar.error(f"Connection Error: {e}")
    AI_READY = False

# --- SESSION STATE ---
if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state: st.session_state["feedback"] = None

# --- DATA ---
QUESTIONS = {
    "Q1: Forces": {"question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate acceleration.", "marks": 3, "mark_scheme": "1. Resultant=16N. 2. F=ma. 3. a=3.2m/s²."},
    "Q2: Refraction": {"question": "Draw a ray diagram: light air to glass.", "marks": 2, "mark_scheme": "1. Bends toward normal. 2. Correct labels."}
}
CLASS_SETS = ["11Y/Ph1", "11X/Ph2", "10A/Ph1", "Teacher Test"]

# --- STORAGE LOGIC ---
def save_to_cloud(name, set_name, q_name, score, max_m, summary):
    try:
        # We read the sheet, add a row, and write it back
        df = conn.read(ttl=0)
        new_data = pd.DataFrame([{
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Student Name": name,
            "Class Set": set_name,
            "Question": q_name,
            "Score": score,
            "Max Marks": max_m,
            "Feedback Summary": summary
        }])
        updated_df = pd.concat([df, new_data], ignore_index=True)
        conn.update(data=updated_df)
        return True
    except Exception as e:
        st.error(f"Cloud Save Failed: {e}")
        return False

def get_gpt_feedback(answer, q_data, is_image=False):
    # Enforce structured JSON output to avoid KeyErrors
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "marking_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "summary": {"type": "string"}
                },
                "required": ["score", "summary"],
                "additionalProperties": False
            }
        }
    }

    messages = [{"role": "system", "content": f"Mark strictly. Scheme: {q_data['mark_scheme']}"}]
    if is_image:
        buffered = io.BytesIO()
        answer.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        messages.append({"role": "user", "content": [{"type": "text", "text": "Mark this drawing."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]})
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
    st.title("⚛️ AI Physics Examiner")
    
    # 1. Identity Section
    with st.container(border=True):
        st.subheader("1. Enter Your Details")
        c1, c2, c3 = st.columns(3)
        fname = c1.text_input("First Name")
        lname = c2.text_input("Last Name")
        student_set = c3.selectbox("Class Set", CLASS_SETS)

    # 2. Work Section
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("2. Complete the Task")
        q_key = st.selectbox("Select Question", list(QUESTIONS.keys()))
        q_data = QUESTIONS[q_key]
        st.info(f"**Question:** {q_data['question']}")
        
        mode = st.radio("Answer Method:", ["⌨️ Type Answer", "✍️ Drawing/Handwriting"], horizontal=True)
        
        if mode == "⌨️ Type Answer":
            ans_text = st.text_area("Your working out:", height=200)
            if st.button("Submit Typed Answer"):
                if not (fname and lname): st.warning("Please enter your name first!")
                else:
                    with st.spinner("Marking..."):
                        res = get_gpt_feedback(ans_text, q_data)
                        st.session_state["feedback"] = res
                        save_to_cloud(f"{fname} {lname}", student_set, q_key, res['score'], q_data['marks'], res['summary'])
        
        else: # DRAWING MODE
            # TOOLBAR for Pen/Eraser/Clear
            tcol1, tcol2, tcol3 = st.columns([1, 1, 1])
            with tcol1: tool = st.toggle("🧼 Eraser", False)
            with tcol2: 
                if st.button("🗑️ Clear Canvas"):
                    st.session_state["canvas_key"] += 1
                    st.rerun()
            
            # Canvas logic
            stroke_color = "#f8f9fa" if tool else "#000000"
            stroke_width = 20 if tool else 2
            
            canvas = st_canvas(
                stroke_width=stroke_width, stroke_color=stroke_color,
                background_color="#f8f9fa", height=300, width=500,
                key=f"c_{st.session_state['canvas_key']}"
            )
            
            if st.button("Submit Drawing"):
                if not (fname and lname): st.warning("Please enter your name first!")
                else:
                    with st.spinner("Analyzing drawing..."):
                        raw = Image.fromarray(canvas.image_data.astype('uint8'))
                        white_bg = Image.new("RGB", raw.size, (255, 255, 255))
                        white_bg.paste(raw, mask=raw.split()[3])
                        res = get_gpt_feedback(white_bg, q_data, is_image=True)
                        st.session_state["feedback"] = res
                        save_to_cloud(f"{fname} {lname}", student_set, q_key, res['score'], q_data['marks'], res['summary'])

    with col_right:
        st.subheader("3. Examiner Feedback")
        if st.session_state["feedback"]:
            res = st.session_state["feedback"]
            st.metric("Score", f"{res['score']} / {q_data['marks']}")
            st.success(f"**Report:** {res['summary']}")
            if st.button("Try Again / New Question"):
                st.session_state["feedback"] = None
                st.rerun()
        else:
            st.info("Submit your work to see your score and feedback.")

with tab2:
    st.title("👨‍🏫 Teacher Dashboard")
    pwd = st.text_input("Access Password", type="password")
    if pwd == "Newton2025":
        try:
            # ttl=0 forces the app to fetch fresh data from Google Sheets
            df = conn.read(ttl=0)
            if not df.empty:
                st.write(f"Total Submissions: {len(df)}")
                st.dataframe(df, use_container_width=True)
                # Export Button
                st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
            else:
                st.info("The sheet is empty. Waiting for student submissions.")
        except Exception as e:
            st.error(f"Could not read Google Sheet: {e}")
            st.info("Check: Have you shared the sheet with the Service Account email?")

# --- SIDEBAR DIAGNOSTICS ---
with st.sidebar:
    st.header("⚙️ System Status")
    if AI_READY: st.success("✅ AI Brain Connected")
    else: st.error("❌ AI Brain Disconnected")
    
    if st.button("Test Google Sheets Connection"):
        try:
            test_df = conn.read(ttl=0)
            st.success(f"✅ Connected! Found {len(test_df)} rows.")
        except Exception as e:
            st.error(f"❌ Connection Failed: {e}")
