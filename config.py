import json
import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


def _safe_secret(key: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# ============================================================
# SUBJECT CONTENT (topics + prompts)
# ============================================================
# For multi-subject scaling: keep the core app identical, and store subject-specific
# topic lists and AI prompt packs under: subjects/<subject_site>/.
#
# Recommended deployment pattern for separate subject sites:
#   - set SUBJECT_SITE in Streamlit Secrets (or environment variable)
#   - e.g. SUBJECT_SITE="physics"
SUBJECT_SITE = _safe_secret("SUBJECT_SITE") or os.getenv("SUBJECT_SITE", "physics")
SUBJECT_SITE = (SUBJECT_SITE or "physics").strip().lower()


@st.cache_data(show_spinner=False)
def _load_subject_pack(subject_site: str) -> dict:
    base = Path(__file__).resolve().parent
    subj_dir = base / "subjects" / subject_site

    topics_path = subj_dir / "topics.json"
    prompts_path = subj_dir / "prompts.json"
    settings_path = subj_dir / "settings.json"
    equations_path = subj_dir / "equations.json"

    if not topics_path.exists():
        raise FileNotFoundError(f"Missing topics file: {topics_path}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing prompts file: {prompts_path}")

    topics = json.loads(topics_path.read_text(encoding="utf-8"))
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    equations = json.loads(equations_path.read_text(encoding="utf-8")) if equations_path.exists() else {}

    return {"topics": topics, "prompts": prompts, "settings": settings, "equations": equations}


try:
    SUBJECT_PACK = _load_subject_pack(SUBJECT_SITE)
except Exception as _e:
    st.error(f"❌ Subject pack failed to load for SUBJECT_SITE='{SUBJECT_SITE}'.\n\n{type(_e).__name__}: {_e}")
    st.stop()

SUBJECT_SETTINGS = SUBJECT_PACK.get("settings", {}) or {}
SUBJECT_TOPICS_RAW = SUBJECT_PACK.get("topics", {}) or {}
SUBJECT_PROMPTS = SUBJECT_PACK.get("prompts", {}) or {}
SUBJECT_EQUATIONS = SUBJECT_PACK.get("equations", {}) or {}

# Topics for dropdowns (student + teacher)
TOPICS_CATALOG = SUBJECT_TOPICS_RAW.get("topics", [])

def _iter_topics_for_track(track: str) -> List[Dict[str, Any]]:
    track = (track or "").strip().lower()
    topics: List[Dict[str, Any]] = []
    for t in TOPICS_CATALOG:
        track_ok = str(t.get("track_ok", "both")).strip().lower() or "both"
        if track == "combined" and track_ok == "separate_only":
            continue
        topics.append(t)
    return topics


def get_topic_names_for_track(track: str) -> List[str]:
    names: List[str] = []
    for t in _iter_topics_for_track(track):
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        names.append(name)
    return names


def get_topic_groups_for_track(track: str) -> List[str]:
    groups: List[str] = []
    for t in _iter_topics_for_track(track):
        group = str(t.get("group", "")).strip()
        if not group:
            continue
        groups.append(group)
    return sorted(set(groups))


def get_topic_group_names_for_track(track: str) -> List[str]:
    return get_topic_groups_for_track(track)


def get_sub_topic_names_for_group(track: str, group: str) -> List[str]:
    group_norm = (group or "").strip().lower()
    if not group_norm:
        return []
    names: List[str] = []
    for t in _iter_topics_for_track(track):
        group_val = str(t.get("group", "")).strip().lower()
        if group_val != group_norm:
            continue
        name = str(t.get("name", "")).strip()
        if name:
            names.append(name)
    return sorted(names, key=lambda value: clean_sub_topic_label(value, track).lower())


def get_all_topic_group_names() -> List[str]:
    groups: List[str] = []
    for t in TOPICS_CATALOG:
        group = str(t.get("group", "")).strip()
        if not group:
            continue
        groups.append(group)
    return sorted(set(groups))


def clean_sub_topic_label(name: str, track: str | None = None) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    raw_lower = raw.lower()
    if track:
        groups = get_topic_group_names_for_track(track)
    else:
        groups = get_all_topic_group_names()
    for group in groups:
        group = str(group or "").strip()
        if not group:
            continue
        prefix = f"{group}:"
        if raw_lower.startswith(prefix.lower()):
            return raw[len(prefix):].lstrip()
    return raw


def get_topic_group_for_name(topic_name: str) -> str | None:
    name_norm = (topic_name or "").strip().lower()
    if not name_norm:
        return None
    for t in TOPICS_CATALOG:
        nm = str(t.get("name", "")).strip().lower()
        if nm == name_norm:
            return str(t.get("group", "")).strip() or None
    return None



def get_topic_track_ok(topic_name: str) -> str:
    """Return track eligibility for a topic name in TOPICS_CATALOG: 'both' or 'separate_only'."""
    name_norm = (topic_name or "").strip().lower()
    if not name_norm:
        return "both"
    for t in TOPICS_CATALOG:
        nm = str(t.get("name", "")).strip().lower()
        if nm == name_norm:
            tok = str(t.get("track_ok", "both") or "both").strip().lower()
            return tok if tok in ("both", "separate_only") else "both"
    return "both"


# UI option lists (can be overridden per subject via settings.json)
QUESTION_TYPES = SUBJECT_SETTINGS.get("question_types") or ["Calculation", "Explanation", "Practical/Methods", "Graph/Analysis", "Mixed"]
DIFFICULTIES = SUBJECT_SETTINGS.get("difficulties") or ["Easy", "Medium", "Hard"]
SKILLS = SUBJECT_SETTINGS.get("skills") or [
    "Recall",
    "Calculation",
    "Explanation",
    "Practical/Methods",
    "Graph/Analysis",
    "Mixed",
]

# Prompt components (loaded from prompts.json)
GCSE_ONLY_GUARDRAILS = str(SUBJECT_PROMPTS.get("gcse_only_guardrails", "") or "").strip()
MARKDOWN_LATEX_RULES = str(SUBJECT_PROMPTS.get("markdown_latex_rules", "") or "").strip()



def _build_equation_guardrails(eq_pack: dict) -> str:
    """Build a compact, prompt-friendly guardrail block from subjects/<site>/equations.json."""
    if not isinstance(eq_pack, dict):
        return ""
    notes = eq_pack.get("notation_rules") or []
    eqs = eq_pack.get("key_equations") or []
    forb = eq_pack.get("forbidden_notes") or []
    out: List[str] = []
    if notes:
        out.append("EQUATION SHEET / NOTATION (from subject pack):")
        for line in notes[:12]:
            s = str(line).strip()
            if s:
                out.append(f"- {s}")
    if eqs:
        out.append("Key equations (use these forms and symbols):")
        for e in eqs[:18]:
            if isinstance(e, dict):
                name = str(e.get("name", "")).strip()
                latex = str(e.get("latex", "")).strip()
                if latex:
                    if name:
                        out.append(f"- {name}: {latex}")
                    else:
                        out.append(f"- {latex}")
            else:
                s = str(e).strip()
                if s:
                    out.append(f"- {s}")
    if forb:
        out.append("Explicitly NOT in AQA GCSE scope for this app:")
        for line in forb[:12]:
            s = str(line).strip()
            if s:
                out.append(f"- {s}")
    return "\n".join(out).strip()


EQUATION_GUARDRAILS = _build_equation_guardrails(SUBJECT_EQUATIONS)
if EQUATION_GUARDRAILS:
    GCSE_ONLY_GUARDRAILS = (GCSE_ONLY_GUARDRAILS + "\n\n" + EQUATION_GUARDRAILS).strip()


# Prompt templates
QGEN_SYSTEM_TPL = str(SUBJECT_PROMPTS.get("qgen_system", "") or "")
QGEN_USER_TPL = str(SUBJECT_PROMPTS.get("qgen_user", "") or "")
QGEN_REPAIR_PREFIX_TPL = str(SUBJECT_PROMPTS.get("qgen_repair_prefix", "") or "")

JOURNEY_SYSTEM_TPL = str(SUBJECT_PROMPTS.get("journey_system", "") or "")
JOURNEY_USER_TPL = str(SUBJECT_PROMPTS.get("journey_user", "") or "")
JOURNEY_REPAIR_PREFIX_TPL = str(SUBJECT_PROMPTS.get("journey_repair_prefix", "") or "")

FEEDBACK_SYSTEM_TPL = str(SUBJECT_PROMPTS.get("feedback_system", "") or "")
