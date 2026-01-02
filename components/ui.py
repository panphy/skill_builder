from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
<style>
:root {
  --pp-radius: 16px;
  --pp-border: 1px solid rgba(15, 23, 42, 0.08);
  --pp-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}

.pp-hero {
  background: linear-gradient(135deg, rgba(91, 140, 255, 0.15), rgba(62, 205, 175, 0.18));
  border-radius: calc(var(--pp-radius) + 6px);
  padding: 28px;
  border: var(--pp-border);
}

.pp-hero h1 {
  margin: 0 0 8px 0;
  font-size: 30px;
  font-weight: 700;
}

.pp-muted {
  color: rgba(15, 23, 42, 0.7);
}

.pp-card {
  border: var(--pp-border);
  border-radius: var(--pp-radius);
  padding: 18px;
  background: #ffffff;
  box-shadow: var(--pp-shadow);
}

.pp-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 4px 12px;
  background: rgba(91, 140, 255, 0.12);
  color: #2448b9;
  font-size: 12px;
  font-weight: 600;
}

.pp-skeleton {
  position: relative;
  overflow: hidden;
  border-radius: 10px;
  background: rgba(148, 163, 184, 0.15);
  height: 14px;
  margin: 8px 0;
}

.pp-skeleton::after {
  content: "";
  position: absolute;
  inset: 0;
  transform: translateX(-100%);
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.65), transparent);
  animation: pp-skeleton 1.4s infinite;
}

@keyframes pp-skeleton {
  100% { transform: translateX(100%); }
}

.pp-callout {
  border-radius: var(--pp-radius);
  border: 1px dashed rgba(15, 23, 42, 0.2);
  padding: 14px 16px;
  background: rgba(248, 250, 252, 0.8);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badge: Optional[str] = None) -> None:
    badge_html = f"<span class='pp-chip'>{badge}</span>" if badge else ""
    st.markdown(
        f"""
<div class="pp-hero">
  {badge_html}
  <h1>{title}</h1>
  <div class="pp-muted">{subtitle}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str, caption: Optional[str] = None) -> None:
    caption_html = f"<div class='pp-muted' style='margin-top:8px'>{caption}</div>" if caption else ""
    st.markdown(
        f"""
<div class="pp-card">
  <div style="font-weight:700; font-size:16px; margin-bottom:6px;">{title}</div>
  <div class="pp-muted">{body}</div>
  {caption_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="pp-card" style="text-align:center;">
  <div style="font-size:26px; margin-bottom:6px;">{icon}</div>
  <div style="font-weight:700; font-size:16px;">{title}</div>
  <div class="pp-muted" style="margin-top:6px;">{body}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_skeleton(lines: int = 3) -> None:
    for _ in range(lines):
        st.markdown("<div class='pp-skeleton'></div>", unsafe_allow_html=True)


def render_callout(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="pp-callout">
  <div style="font-weight:700; margin-bottom:4px;">{title}</div>
  <div class="pp-muted">{body}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_bullets(items: Iterable[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")
