import re
from typing import Dict

import streamlit as st


_MD_TOKEN_CODEBLOCK = re.compile(r"```.*?```", re.DOTALL)
_MD_TOKEN_INLINECODE = re.compile(r"`[^`\n]+`")
_MD_TOKEN_MATHBLOCK = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_MD_TOKEN_MATHINLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def _protect_segments(pattern: re.Pattern, text_in: str, store: Dict[str, str], prefix: str) -> str:
    def _repl(match):
        key = f"@@{prefix}{len(store)}@@"
        store[key] = match.group(0)
        return key

    return pattern.sub(_repl, text_in)


def _restore_segments(text_in: str, store: Dict[str, str]) -> str:
    out = text_in
    for key in sorted(store.keys(), key=lambda value: -len(value)):
        out = out.replace(key, store[key])
    return out


def normalize_markdown_math(md_text: str) -> str:
    text = md_text or ""
    if not text.strip():
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\\\\([A-Za-z])", r"\\\1", text)
    text = re.sub(r"\\\\([\\[\\]{}()^_])", r"\\\1", text)

    protected: Dict[str, str] = {}
    text = _protect_segments(_MD_TOKEN_CODEBLOCK, text, protected, "CB")
    text = _protect_segments(_MD_TOKEN_INLINECODE, text, protected, "IC")

    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    def _normalize_unit_escapes(math_text: str) -> str:
        out = math_text
        out = re.sub(r"\\m\s*\^\s*\{?\s*-?1\s*\}?", r"\\,m^{-1}", out)
        out = re.sub(r"\\m\b", r"\\,m", out)
        out = re.sub(r"\\s\b", r"\\,s", out)
        out = re.sub(r"\\kg\b", r"\\,kg", out)
        return out

    def _normalize_units_in_math(text_in: str) -> str:
        def _fix_block(match: re.Match) -> str:
            return f"$${_normalize_unit_escapes(match.group(1))}$$"

        def _fix_inline(match: re.Match) -> str:
            return f"${_normalize_unit_escapes(match.group(1))}$"

        out = re.sub(r"\$\$(.*?)\$\$", _fix_block, text_in, flags=re.DOTALL)
        return re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _fix_inline, out, flags=re.DOTALL)

    text = _normalize_units_in_math(text)
    text = _protect_segments(_MD_TOKEN_MATHBLOCK, text, protected, "MB")
    text = _protect_segments(_MD_TOKEN_MATHINLINE, text, protected, "MI")

    token_pat = re.compile(
        r"(?<!\$)(?<!\\)(?<![A-Za-z0-9])"
        r"([A-Za-z0-9]+(?:/[A-Za-z0-9]+)*"
        r"(?:\s*(?:\^\s*[-+]?\d+|_\{[^}]+\}|_[A-Za-z0-9]+)))"
        r"(?![A-Za-z0-9])"
    )

    def _wrap(match: re.Match) -> str:
        expr = re.sub(r"\s*(\^|_)\s*", r"\1", match.group(1)).strip()
        return f"${expr}$"

    text = token_pat.sub(_wrap, text)
    return _restore_segments(text, protected)


def render_md_box(title: str, md_text: str, caption: str = "", empty_text: str = "") -> None:
    st.markdown(f"**{title}**")
    with st.container(border=True):
        txt = normalize_markdown_math((md_text or "").strip())
        if txt:
            st.markdown(txt)
        else:
            st.caption(empty_text or "No content.")
    if caption:
        st.caption(caption)


def render_report(report: dict) -> None:
    readback_md = (report.get("readback_markdown") or "").strip()
    if readback_md:
        st.markdown("**AI readback (what it thinks you wrote/drew):**")
        with st.container(border=True):
            st.markdown(normalize_markdown_math(readback_md))

        rb_warn = report.get("readback_warnings", [])
        if rb_warn:
            st.caption("Readback notes:")
            for warning in rb_warn[:6]:
                st.markdown(normalize_markdown_math(f"- {warning}"))
        st.divider()

    st.markdown(f"**Marks:** {int(report.get('marks_awarded', 0))} / {int(report.get('max_marks', 0))}")
    if report.get("summary"):
        st.markdown(normalize_markdown_math(f"**Summary:** {report.get('summary')}"))
    if report.get("feedback_points"):
        st.markdown("**Feedback:**")
        for point in report["feedback_points"]:
            st.markdown(normalize_markdown_math(f"- {point}"))
    if report.get("next_steps"):
        st.markdown("**Next steps:**")
        for step in report["next_steps"]:
            st.markdown(normalize_markdown_math(f"- {step}"))
