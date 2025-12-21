import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="OpenAI Physics Examiner", page_icon="📝", layout="wide")

# --- INITIALIZE OPENAI CLIENT ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    AI_READY = True
except Exception:
    st.error("⚠️ OpenAI API Key missing in Secrets!")
    AI_READY = False

# --- QUESTION BANK ---
QUESTIONS = {
    "Q1: Forces": {
        "question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate the acceleration.",
        "marks": 3,
        "mark_scheme": "1. Resultant force = 20 - 4 = 16N (1 mark). 2. F = ma (1 mark). 3. a = 16 / 5 = 3.2 m/s² (1 mark)."
    },
    "Q2: Refraction": {
        "question": "Draw a ray diagram showing light passing from air into a glass block.",
        "marks": 2,
        "mark_scheme": "1. Ray bends towards the normal inside the glass. 2. Angles of incidence and refraction labeled correctly."
    }
}

# --- HELPER: ENCODE IMAGE FOR GPT-5.2 ---
def encode_image(image_pil):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_gpt_feedback(student_answer, q_data, is_image=False):
    # Using GPT-5.2 for frontier reasoning and visual marking
    model_name = "gpt-5.2" 
    
    system_instr = f"You are a GCSE Physics Examiner. Mark strictly.\nQuestion: {q_data['question']}\nScheme: {q_data['mark_scheme']}\nMax Marks: {q_data['marks']}"
    
    messages = [{"role": "system", "content": system_instr}]
    
    if is_image:
        base64_img = encode_image(student_answer)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Mark this handwritten/drawn answer."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": f"Student Answer: {student_answer}"})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Examiner Error: {e}"

# --- APP UI ---
st.title("🚀 OpenAI Examiner (GPT-5.2)")
st.sidebar.header("Select Task")
q_key = st.sidebar.selectbox("Question", list(QUESTIONS.keys()))
q_data = QUESTIONS[q_key]

st.info(f"**Question:** {q_data['question']} ({q_data['marks']} marks)")

mode = st.radio("Mode:", ["Type", "Draw/Write"], horizontal=True)

if mode == "Type":
    answer = st.text_area("Your answer:")
    if st.button("Submit") and AI_READY:
        st.markdown(get_gpt_feedback(answer, q_data))
else:
    # Drawing Canvas
    canvas_result = st_canvas(stroke_width=2, stroke_color="#000", background_color="#eee", height=300, width=600, key="canvas")
    if st.button("Submit Drawing") and AI_READY:
        if canvas_result.image_data is not None:
            # Prepare Image
            raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'))
            st.markdown(get_gpt_feedback(raw_img, q_data, is_image=True))
