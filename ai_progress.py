import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from attempts import insert_ai_timing, load_ai_timing_average_cached
from config import SUBJECT_SITE, _safe_secret
from db import db_ready


def _resolve_timing_type(ctx: dict) -> Optional[str]:
    mode = str((ctx or {}).get("mode", "") or "").strip().lower()
    if not mode:
        return None
    if mode in {"text", "writing"} or mode.startswith("journey_"):
        return "ai_marking"
    if mode == "ai_generator":
        return "single_question_generation"
    if mode == "topic_journey":
        return "topic_journey_generation"
    return None


def _run_ai_with_progress(
    task_fn,
    ctx: dict,
    typical_range: str,
    est_seconds: float,
    subtitle: str | None = None,
    step_index: int | None = None,
    total_steps: int | None = None,
) -> dict:
    overlay = st.empty()
    start_t = time.monotonic()
    timing_type = _resolve_timing_type(ctx)
    effective_est_seconds = est_seconds
    estimate_label = typical_range
    if timing_type and db_ready():
        avg_s, _count = load_ai_timing_average_cached(
            _safe_secret("DATABASE_URL", "") or "",
            SUBJECT_SITE,
            timing_type,
        )
        if avg_s is not None:
            effective_est_seconds = avg_s
            estimate_label = f"{avg_s:.0f} seconds (avg)"

    def _render_overlay(subtitle_text: str, percent: int) -> None:
        step_label = ""
        step_percent = 0
        if total_steps and total_steps > 0 and step_index:
            step_label = f"Question {step_index} of {total_steps}"
            step_percent = min(100, max(0, int((step_index / total_steps) * 100)))

        popup_script = f"""
<script>
(function() {{
    var doc = window.parent.document;
    var overlay = doc.getElementById('ai-loading-overlay');

    var percent = {percent};
    var subtitle = "{subtitle_text}";
    var stepLabel = "{step_label}";
    var stepPercent = {step_percent};
    var estimateLabel = "{estimate_label}";

    if (overlay) {{
        var progressFill = overlay.querySelector('.ai-popup-progress-fill');
        var percentText = overlay.querySelector('.ai-popup-percent');
        var subtitleEl = overlay.querySelector('.ai-popup-subtitle');
        var estimateEl = overlay.querySelector('.ai-popup-estimate');
        var stepContainer = overlay.querySelector('.ai-popup-step-container');

        if (progressFill) progressFill.style.width = percent + '%';
        if (percentText) percentText.textContent = percent + '%';
        if (estimateEl) estimateEl.textContent = 'Estimate: ' + estimateLabel;

        if (subtitleEl) {{
            if (subtitle) {{
                subtitleEl.textContent = subtitle;
                subtitleEl.style.display = 'block';
            }} else {{
                subtitleEl.style.display = 'none';
            }}
        }}

        if (stepContainer) {{
            if (stepLabel) {{
                var stepLabelEl = stepContainer.querySelector('.ai-popup-step-label');
                var stepFill = stepContainer.querySelector('.ai-popup-progress-fill');
                if (stepLabelEl) stepLabelEl.textContent = stepLabel;
                if (stepFill) stepFill.style.width = stepPercent + '%';
                stepContainer.style.display = 'block';
            }} else {{
                stepContainer.style.display = 'none';
            }}
        }}
        return;
    }}

    var styleId = 'ai-loading-styles';
    if (!doc.getElementById(styleId)) {{
        var style = doc.createElement('style');
        style.id = styleId;
        style.textContent = `
            #ai-loading-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.6);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 999999;
                backdrop-filter: blur(4px);
            }}
            .ai-loading-popup {{
                background: white;
                border-radius: 16px;
                padding: 32px 40px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                text-align: center;
                max-width: 420px;
                width: 90%;
                animation: ai-popup-appear 0.3s ease-out;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            @keyframes ai-popup-appear {{
                from {{ opacity: 0; transform: scale(0.9) translateY(-10px); }}
                to {{ opacity: 1; transform: scale(1) translateY(0); }}
            }}
            .ai-popup-spinner {{
                width: 48px;
                height: 48px;
                border: 4px solid #e0e0e0;
                border-top: 4px solid #4A90D9;
                border-radius: 50%;
                animation: ai-spin 1s linear infinite;
                margin: 0 auto 20px auto;
            }}
            @keyframes ai-spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .ai-popup-title {{ font-size: 20px; font-weight: 600; color: #1a1a1a; margin-bottom: 8px; }}
            .ai-popup-subtitle {{ font-size: 14px; color: #666; margin-bottom: 16px; }}
            .ai-popup-progress-container {{ margin: 20px 0; }}
            .ai-popup-progress-bar {{ width: 100%; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; margin-bottom: 8px; }}
            .ai-popup-progress-fill {{ height: 100%; background: linear-gradient(90deg, #4A90D9, #6BA3E0); border-radius: 4px; transition: width 0.3s ease; }}
            .ai-popup-percent {{ font-size: 14px; color: #333; font-weight: 500; }}
            .ai-popup-estimate {{ font-size: 13px; color: #888; margin-top: 4px; }}
            .ai-popup-note {{ font-size: 12px; color: #999; margin-top: 16px; font-style: italic; }}
            .ai-popup-step-container {{ margin-top: 16px; }}
            .ai-popup-step-label {{ font-size: 13px; color: #666; margin-bottom: 8px; }}
            @media (prefers-color-scheme: dark) {{
                .ai-loading-popup {{ background: #2d2d2d; }}
                .ai-popup-title {{ color: #f0f0f0; }}
                .ai-popup-subtitle {{ color: #aaa; }}
                .ai-popup-progress-bar {{ background: #404040; }}
                .ai-popup-percent {{ color: #e0e0e0; }}
                .ai-popup-estimate {{ color: #888; }}
                .ai-popup-note {{ color: #777; }}
                .ai-popup-step-label {{ color: #aaa; }}
            }}
        `;
        doc.head.appendChild(style);
    }}

    overlay = doc.createElement('div');
    overlay.id = 'ai-loading-overlay';
    overlay.innerHTML = `
        <div class="ai-loading-popup">
            <div class="ai-popup-spinner"></div>
            <div class="ai-popup-title">AI is working</div>
            <div class="ai-popup-subtitle" style="display: ${{subtitle ? 'block' : 'none'}};">${{subtitle}}</div>
            <div class="ai-popup-progress-container">
                <div class="ai-popup-progress-bar"><div class="ai-popup-progress-fill" style="width: ${{percent}}%;"></div></div>
                <div class="ai-popup-percent">${{percent}}%</div>
                <div class="ai-popup-estimate">Estimate: ${{estimateLabel}}</div>
            </div>
            <div class="ai-popup-step-container" style="display: ${{stepLabel ? 'block' : 'none'}};">
                <div class="ai-popup-step-label">${{stepLabel}}</div>
                <div class="ai-popup-progress-bar"><div class="ai-popup-progress-fill" style="width: ${{stepPercent}}%;"></div></div>
            </div>
            <div class="ai-popup-note">May take longer for complex tasks</div>
        </div>
    `;
    doc.body.appendChild(overlay);
}})();
</script>
"""

        with overlay.container():
            components.html(popup_script, height=0, scrolling=False)

    def _calc_percent(elapsed_s: float, done: bool = False) -> int:
        if done:
            return 100
        if effective_est_seconds <= 0:
            return 0
        return min(95, max(0, int((elapsed_s / effective_est_seconds) * 100)))

    _render_overlay(subtitle or "", 0)

    report = None
    success = False
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task_fn)
            while not future.done():
                elapsed = time.monotonic() - start_t
                _render_overlay(subtitle or "", _calc_percent(elapsed))
                time.sleep(0.35)

            report = future.result()
            success = True

        _render_overlay("Done. Updating the page...", _calc_percent(time.monotonic() - start_t, done=True))
        time.sleep(0.08)
        return report
    finally:
        if timing_type and success and report is not None:
            insert_ai_timing(timing_type, time.monotonic() - start_t, success=True)
        overlay.empty()
        cleanup_script = """
<script>
(function() {
    var doc = window.parent.document;
    var overlay = doc.getElementById('ai-loading-overlay');
    if (overlay) overlay.remove();
})();
</script>
"""
        components.html(cleanup_script, height=0, scrolling=False)
