from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import streamlit as st


@dataclass(frozen=True)
class StatusTone:
    name: str
    icon: str


STATUS_TONES = {
    "info": StatusTone("info", "ℹ️"),
    "success": StatusTone("success", "✅"),
    "warning": StatusTone("warning", "⚠️"),
    "error": StatusTone("error", "🚨"),
}


def _tone(kind: str) -> StatusTone:
    return STATUS_TONES.get(kind, STATUS_TONES["info"])


def render_page_header(title: str, subtitle: str | None = None, tag: str | None = None) -> None:
    tag_html = f"<span class='pp-pill'>{tag}</span>" if tag else ""
    subtitle_html = f"<p class='pp-subtitle'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="pp-hero">
          <div>
            <h1>{title} {tag_html}</h1>
            {subtitle_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str | None = None, eyebrow: str | None = None) -> None:
    eyebrow_html = f"<span class='pp-eyebrow'>{eyebrow}</span>" if eyebrow else ""
    description_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        f"""
        <div class="pp-section">
          {eyebrow_html}
          <h2>{title}</h2>
          {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(title: str, body: str, *, kind: str = "info") -> None:
    tone = _tone(kind)
    st.markdown(
        f"""
        <div class="pp-callout pp-callout-{tone.name}">
          <strong>{tone.icon} {title}</strong>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str, *, icon: str = "🧭") -> None:
    st.markdown(
        f"""
        <div class="pp-card pp-empty">
          <div class="pp-empty-icon">{icon}</div>
          <div>
            <h3>{title}</h3>
            <p>{body}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_panel(title: str, body: str, *, kind: str = "info") -> None:
    tone = _tone(kind)
    st.markdown(
        f"""
        <div class="pp-card pp-status pp-status-{tone.name}">
          <h4>{tone.icon} {title}</h4>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(steps: Iterable[str], active_index: int = 0) -> None:
    items = []
    for idx, step in enumerate(steps):
        state = "active" if idx == active_index else "done" if idx < active_index else ""
        items.append(
            f"""
            <div class="pp-step {state}">
              <span class="pp-step-index">{idx + 1}</span>
              <span class="pp-step-label">{step}</span>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="pp-stepper">
          {"".join(items)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skeleton(rows: int = 3, height_px: int = 14) -> None:
    skeletons = "".join(
        [f"<div class='pp-skeleton' style='height:{height_px}px'></div>" for _ in range(max(rows, 1))]
    )
    st.markdown(
        f"""
        <div class="pp-skeleton-stack">
          {skeletons}
        </div>
        """,
        unsafe_allow_html=True,
    )
