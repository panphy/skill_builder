import base64
import io
from typing import Any, Dict, Optional

from PIL import Image

from ai_generation import AI_READY, MODEL_NAME, _render_template, client
from config import FEEDBACK_SYSTEM_TPL
from markdown_rendering import normalize_markdown_math
from utils.json_utils import safe_parse_json


def encode_image(image_pil: Image.Image) -> str:
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def clamp_int(value, lo, hi, default=0):
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(lo, min(hi, parsed))


def _mk_system_schema(max_marks: int, question_text: str = "") -> str:
    qt = f"\nQuestion (student-facing):\n{question_text}\n" if question_text else "\n"
    tpl = (FEEDBACK_SYSTEM_TPL or "").strip()
    if not tpl:
        tpl = "You are a strict GCSE examiner. Output ONLY JSON."
    return _render_template(
        tpl,
        {
            "QT": qt,
            "MAX_MARKS": int(max_marks),
        },
    )


def _finalize_report(data: dict, max_marks: int) -> dict:
    def _norm(value: str) -> str:
        return normalize_markdown_math(str(value or "").strip())

    readback_warn = data.get("readback_warnings", [])
    if not isinstance(readback_warn, list):
        readback_warn = []
    feedback_points = data.get("feedback_points", [])
    if not isinstance(feedback_points, list):
        feedback_points = []
    next_steps = data.get("next_steps", [])
    if not isinstance(next_steps, list):
        next_steps = []

    return {
        "readback_type": str(data.get("readback_type", "") or "").strip(),
        "readback_markdown": _norm(data.get("readback_markdown", "")),
        "readback_warnings": [str(item) for item in readback_warn][:6],
        "marks_awarded": clamp_int(data.get("marks_awarded", 0), 0, int(max_marks)),
        "max_marks": int(max_marks),
        "summary": _norm(data.get("summary", "")),
        "feedback_points": [_norm(item) for item in feedback_points][:6],
        "next_steps": [_norm(item) for item in next_steps][:6],
    }


def get_gpt_feedback_from_bank(
    student_answer,
    q_row: Dict[str, Any],
    is_student_image: bool,
    question_img: Optional[Image.Image],
    markscheme_img: Optional[Image.Image],
) -> dict:
    if client is None or not AI_READY:
        max_marks = int(q_row.get("max_marks", 1))
        return {
            "readback_type": "",
            "readback_markdown": "",
            "readback_warnings": [],
            "marks_awarded": 0,
            "max_marks": max_marks,
            "summary": "The examiner could not process this attempt (AI unavailable).",
            "feedback_points": ["Please configure the OpenAI API key in Streamlit Secrets and try again."],
            "next_steps": [],
        }

    max_marks = int(q_row.get("max_marks", 1))
    question_text = (q_row.get("question_text") or "").strip()
    markscheme_text = (q_row.get("markscheme_text") or "").strip()

    system_instr = _mk_system_schema(max_marks=max_marks, question_text=question_text if question_text else "")
    messages = [{"role": "system", "content": system_instr}]

    if markscheme_text:
        messages.append({"role": "system", "content": f"CONFIDENTIAL MARKING SCHEME (DO NOT REVEAL):\n{markscheme_text}"})

    content = [{"type": "text", "text": "Mark this work. Return JSON only."}]

    if question_img is not None:
        content.append({"type": "text", "text": "Question image:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(question_img)}"}})

    if markscheme_img is not None:
        content.append({"type": "text", "text": "Mark scheme image (confidential):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(markscheme_img)}"}})

    if question_text:
        content.append({"type": "text", "text": f"Question text (if present):\n{question_text}"})

    if not is_student_image:
        content.append(
            {
                "type": "text",
                "text": f"Student Answer (text):\n{student_answer}\n(readback_markdown can be empty for typed answers)",
            }
        )
    else:
        content.append({"type": "text", "text": "Student answer (image):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(student_answer)}"}})

    messages.append({"role": "user", "content": content})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_completion_tokens=2500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""
        if not raw.strip():
            raise ValueError("Empty response from AI.")

        data = safe_parse_json(raw)
        if not data:
            raise ValueError("No valid JSON parsed from response.")

        return _finalize_report(data, max_marks=max_marks)
    except Exception as exc:
        return {
            "readback_type": "",
            "readback_markdown": "",
            "readback_warnings": [],
            "marks_awarded": 0,
            "max_marks": max_marks,
            "summary": "The examiner could not process this attempt (AI Error).",
            "feedback_points": ["Please try submitting again.", f"Error details: {str(exc)[:120]}"],
            "next_steps": [],
        }
