import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64
import json
import re
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Physics Examiner",
    page_icon="⚛️",
    layout="wide"
)

# --- CONSTANTS ---
MODEL_NAME = "gpt-5-mini"
CANVAS_BG_HEX = "#f8f9fa"
CANVAS_BG_RGB = (248, 249, 250)
MAX_IMAGE_WIDTH = 1024

# --- OPENAI CLIENT (CACHED) ---
@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

try:
    client = get_client()
    AI_READY = True
except Exception:
    st.error("⚠️ OpenAI API Key missing or invalid in Streamlit Secrets!")
    AI_READY = False

# --- SESSION STATE ---
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0
if "feedback" not in st.session_state:
    st.session_state["feedback"] = None

# --- QUESTION BANK ---
QUESTIONS = {
    "Q1: Forces (Resultant)": {
        "question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate the acceleration.",
        "marks": 3,
        "mark_scheme": "1. Resultant force = 20 - 4 = 16N (1 mark). 2. F = ma (1 mark). 3. a = 16 / 5 = 3.2 m/s² (1 mark).",
    },
    "Q2: Refraction (Drawing)": {
        "question": "Draw a ray diagram showing light passing from air into a glass block at an angle.",
        "marks": 2,
        "mark_scheme": "1. Ray bends towards the normal inside the glass. 2. Angles of incidence and refraction labeled correctly.",
    },
}

# --- HELPER FUNCTIONS ---
def encode_image(image_pil: Image.Image) -> str:
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def clamp_int(value, lo, hi, default=0):
    try:
        v = int(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))

def canvas_has_ink(image_data: np.ndarray) -> bool:
    if image_data is None:
        return False
    arr = image_data.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] if arr.shape[2] >= 4 else np.full((arr.shape[0], arr.shape[1]), 255, dtype=np.uint8)
    bg = np.array(CANVAS_BG_RGB, dtype=np.uint8)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    ink = (diff > 60) & (alpha > 30)
    return (ink.mean() > 0.001)

def preprocess_canvas_image(image_data: np.ndarray) -> Image.Image:
    raw_img = Image.fromarray(image_data.astype("uint8"))
    if raw_img.mode == "RGBA":
        white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
        white_bg.paste(raw_img, mask=raw_img.split()[3])
        img = white_bg
    else:
        img = raw_img.convert("RGB")
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / img.width
        img = img.resize((MAX_IMAGE_WIDTH, int(img.height * ratio)))
    return img

def get_gpt_feedback(student_answer, q_data, is_image=False):
    max_marks = q_data["marks"]
    system_instr = f"""
You are a strict GCSE Physics examiner.
CONFIDENTIALITY RULE (CRITICAL):
- The mark scheme is confidential. Do NOT reveal it, quote it, or paraphrase it.
- Output ONLY valid JSON, nothing else.
Schema:
{{
  "marks_awarded": <int>,
  "max_marks": <int>,
  "summary": "<1-2 sentences>",
  "feedback_points": ["<bullet 1>", "<bullet 2>"],
  "next_steps": ["<action 1>", "<action 2>"]
}}
Question: {q_data["question"]}
Max Marks: {max_marks}
""".strip()

    messages = [{"role": "system", "content": system_instr}]
    messages.append({
        "role": "system",
        "content": f"CONFIDENTIAL MARKING SCHEME (DO NOT REVEAL): {q_data['mark_scheme']}"
    })

    if is_image:
        base64_img = encode_image(student_answer)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Mark this work. Return JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Student Answer:\n{student_answer}\n\nReturn JSON only."
        })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_completion_tokens=800,
            reasoning_effort="minimal",
        )
        raw = response.choices[0].message.content or ""
        data = safe_parse_json(raw)
        
        if not data:
            raise ValueError("No valid JSON parsed.")

        return {
            "marks_awarded": clamp_int(data.get("marks_awarded", 0), 0, max_marks),
            "max_marks": max_marks,
            "summary": str(data.get("summary", "")).strip(),
            "feedback_points": [str(x) for x in data.get("feedback_points", [])][:6],
            "next_steps": [str(x) for x in data.get("next_steps", [])][:6]
        }
    except Exception as e:
        return {
            "marks_awarded": 0,
            "max_marks": max_marks,
            "summary": "Could not generate report.",
            "feedback_points": ["Please try submitting again."],
            "next_steps": []
        }

def render_report(report: dict):
    st.markdown(f"**Marks:** {report.get('marks_awarded', 0)} / {report.get('max_marks', 0)}")
    if report.get("summary"):
        st.markdown(f"**Summary:** {report.get('summary')}")
    if report.get("feedback_points"):
        st.markdown("**Feedback:**")
        for p in report["feedback_points"]:
            st.write(f"- {p}")
    if report.get("next_steps"):
        st.markdown("**Next steps:**")
        for n in report["next_steps"]:
            st.write(f"- {n}")

# --- MAIN APP UI ---

# 1. Top Navigation Bar (Replaces Sidebar)
top_col1, top_col2, top_col3 = st.columns([3, 2, 1])

with top_col1:
    st.title("⚛️ AI Examiner")
    st.caption(f"Powered by {MODEL_NAME}")

with top_col2:
    # Selector moved to top
    q_key = st.selectbox("Select Topic:", list(QUESTIONS.keys()))
    q_data = QUESTIONS[q_key]

with top_col3:
    # Clean visual spacer or status indicator
    if AI_READY:
        st.success("System Online", icon="🟢")
    else:
        st.error("API Error", icon="🔴")

st.divider()

# 2. Main Content Area
col1, col2 = st.columns([5, 4]) # Adjusted ratio for better canvas space

with col1:
    st.subheader("📝 The Question")
    st.markdown(f"**{q_data['question']}**")
    st.caption(f"Max Marks: {q_data['marks']}")
    
    st.write("") # Spacer

    # Input Method Tabs
    tab_type, tab_draw = st.tabs(["⌨️ Type Answer", "✍️ Draw Answer"])

    with tab_type:
        answer = st.text_area("Type your working:", height=200, placeholder="Enter your answer here...")
        if st.button("Submit Text", type="primary", disabled=not AI_READY):
            if not answer.strip():
                st.toast("Please type an answer first.", icon="⚠️")
            else:
                with st.spinner("Marking..."):
                    st.session_state["feedback"] = get_gpt_feedback(answer, q_data, is_image=False)

    with tab_draw:
        # Toolbar for canvas
        tool_c1, tool_c2, tool_c3 = st.columns([2, 2, 3])
        with tool_c1:
            tool = st.radio("Tool", ["Pen", "Eraser"], horizontal=True, label_visibility="collapsed")
        with tool_c3:
            if st.button("🗑️ Clear Canvas"):
                st.session_state["canvas_key"] += 1
                st.session_state["feedback"] = None
                st.rerun()

        stroke_width = 2 if tool == "Pen" else 30
        stroke_color = "#000000" if tool == "Pen" else CANVAS_BG_HEX

        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=CANVAS_BG_HEX,
            height=400,
            width=600, # Increased width
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state['canvas_key']}",
        )

        if st.button("Submit Drawing", type="primary", disabled=not AI_READY):
            if canvas_result.image_data is None or not canvas_has_ink(canvas_result.image_data):
                st.toast("Canvas is empty!", icon="⚠️")
            else:
                with st.spinner("Analyzing diagram..."):
                    img_for_ai = preprocess_canvas_image(canvas_result.image_data)
                    st.session_state["feedback"] = get_gpt_feedback(img_for_ai, q_data, is_image=True)

with col2:
    st.subheader("👨‍🏫 Report")
    with st.container(border=True):
        if st.session_state["feedback"]:
            render_report(st.session_state["feedback"])
            st.divider()
            if st.button("Start New Attempt", use_container_width=True):
                st.session_state["feedback"] = None
                st.rerun()
        else:
            st.info("Submit an answer to receive feedback.")
