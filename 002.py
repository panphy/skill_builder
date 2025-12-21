import streamlit as st
from openai import OpenAI
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="OpenAI Physics Examiner", page_icon="🚀", layout="wide")

# --- INITIALIZE OPENAI CLIENT ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    AI_READY = True
except Exception:
    st.error("⚠️ OpenAI API Key missing in Streamlit Secrets!")
    AI_READY = False

# --- QUESTION BANK ---
QUESTIONS = {
    "Q1: Forces (Resultant)": {
        "question": "A 5kg box is pushed with a 20N force. Friction is 4N. Calculate the acceleration.",
        "marks": 3,
        "mark_scheme": "1. Resultant force = 20 - 4 = 16N (1 mark). 2. F = ma (1 mark). 3. a = 16 / 5 = 3.2 m/s² (1 mark)."
    },
    "Q2: Refraction (Drawing)": {
        "question": "Draw a ray diagram showing light passing from air into a glass block at an angle.",
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
    # GPT-5.2 is the flagship; GPT-5.2-instant is the fast fallback
    model_name = "gpt-5.2" 
    
    system_instr = f"""You are a strict GCSE Physics Examiner. 
    Mark strictly according to the mark scheme. 
    Question: {q_data['question']}
    Mark Scheme: {q_data['mark_scheme']}
    Max Marks: {q_data['marks']}"""
    
    messages = [{"role": "system", "content": system_instr}]
    
    if is_image:
        base64_img = encode_image(student_answer)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Mark this handwritten/drawn physics answer strictly."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": f"Student Answer: {student_answer}"})

    try:
        # FIX: max_tokens replaced with max_completion_tokens for GPT-5 series
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=500 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Examiner Error: {str(e)}"

# --- APP UI ---
st.title("🚀 OpenAI Examiner (GPT-5.2)")
st.sidebar.header("Select Task")
q_key = st.sidebar.selectbox("Question Topic", list(QUESTIONS.keys()))
q_data = QUESTIONS[q_key]

st.info(f"**Question:** {q_data['question']} \n\n**Marks available:** {q_data['marks']}")

mode = st.radio("How will you answer?", ["⌨️ Type", "✍️ Handwriting/Drawing"], horizontal=True)

if mode == "Type":
    answer = st.text_area("Type your working and final answer here:", height=150)
    if st.button("Submit Answer") and AI_READY:
        with st.spinner("GPT-5.2 is marking..."):
            st.markdown(get_gpt_feedback(answer, q_data))
else:
    st.write("Draw/Write your working below:")
    # Digital Canvas for Handwriting
    canvas_result = st_canvas(
        stroke_width=2, 
        stroke_color="#000", 
        background_color="#f8f9fa", 
        height=350, 
        width=700, 
        key="canvas"
    )
    if st.button("Submit Drawing") and AI_READY:
        if canvas_result.image_data is not None:
            with st.spinner("Analyzing handwriting..."):
                raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                # Replace transparency for AI clarity
                white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
                white_bg.paste(raw_img, mask=raw_img.split()[3]) 
                st.markdown(get_gpt_feedback(white_bg, q_data, is_image=True))
        else:
            st.warning("Please write something on the canvas first.")
