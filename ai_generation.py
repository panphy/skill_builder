import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

from config import (
    GCSE_ONLY_GUARDRAILS,
    MARKDOWN_LATEX_RULES,
    QGEN_REPAIR_PREFIX_TPL,
    QGEN_SYSTEM_TPL,
    QGEN_USER_TPL,
    JOURNEY_REPAIR_PREFIX_TPL,
    JOURNEY_SYSTEM_TPL,
    JOURNEY_USER_TPL,
    DIFFICULTIES,
    SKILLS,
    SUBJECT_EQUATIONS,
    get_topic_group_for_name,
    get_topic_group_names_for_track,
    get_sub_topic_names_for_group,
)

LOGGER = logging.getLogger("panphy")

MODEL_NAME = "gpt-5-mini"
TRACK_DEFAULT = "combined"

ATOMS_ISOTOPES_SUB_TOPIC = "Atomic structure: Atoms and isotopes"
ATOMS_ISOTOPES_PHYSICS_ONLY = (
    "For 'Atoms and isotopes' (GCSE Physics), keep it physics-only: protons, neutrons, "
    "electrons, isotopes, atomic number/mass number, nuclear charge, and atomic models. "
    "Do NOT ask about electron shell arrangements, electron configuration, chemical bonding, "
    "the periodic table, or chemical properties."
)


def _append_instruction(extra_instructions: str, instruction: str) -> str:
    extra_instructions = (extra_instructions or "").strip()
    if not instruction:
        return extra_instructions
    if extra_instructions:
        return f"{extra_instructions}\n{instruction}"
    return instruction


@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


client: Optional[OpenAI] = None
try:
    client = get_client()
    AI_READY = True
except Exception as e:
    client = None
    st.error("⚠️ OpenAI API Key missing or invalid in Streamlit Secrets!")
    AI_READY = False
    LOGGER.error("OpenAI client init failed", extra={"ctx": {"component": "openai", "error": type(e).__name__}})


# ============================================================
# --- JSON / PROMPT UTILS ---
# ============================================================

def safe_parse_json(text_str: str):
    try:
        return json.loads(text_str)
    except Exception:
        pass

    try:
        # Scan for the first balanced-brace JSON object to avoid greedy regex capture.
        s = text_str or ""
        start = s.find("{")
        if start != -1:
            depth = 0
            for idx in range(start, len(s)):
                ch = s[idx]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(s[start : idx + 1])
    except Exception:
        pass
    return None



def _render_template(tpl: str, mapping: Dict[str, Any]) -> str:
    # Simple token replacement. Tokens look like: <<TOKEN_NAME>>
    out = str(tpl or "")
    for k, v in (mapping or {}).items():
        out = out.replace(f"<<{k}>>", str(v))
    return out


# ============================================================
# --- EQUATION VALIDATION HELPERS ---
# ============================================================

def _get_equation_whitelist(eq_pack: dict) -> set:
    whitelist: set = set()
    if not isinstance(eq_pack, dict):
        return whitelist
    for eq in (eq_pack.get("key_equations") or []):
        if isinstance(eq, dict):
            latex = str(eq.get("latex", "") or "").strip()
            if latex:
                normalized = _normalize_equation_text(latex)
                if normalized:
                    whitelist.add(normalized)
                    whitelist.update(_derive_equation_rearrangements(normalized))
        else:
            s = str(eq).strip()
            if s:
                normalized = _normalize_equation_text(s)
                if normalized:
                    whitelist.add(normalized)
    for note in (eq_pack.get("notation_rules") or []):
        for match in _get_equation_regexes()["plain_eq"].finditer(str(note or "")):
            whitelist.add(_normalize_equation_text(match.group(1)))
    return whitelist



def _get_equation_regexes() -> Dict[str, re.Pattern]:
    return {
        "latex_block": re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
        "latex_inline": re.compile(r"\$(.+?)\$", re.DOTALL),
        "latex_paren": re.compile(r"\\\((.+?)\\\)", re.DOTALL),
        "plain_eq": re.compile(r"([A-Za-z0-9\\_\^\+\-\*/\(\)\{\}=\s]+)"),
    }



def _extract_simple_fraction(expr: str) -> Optional[Tuple[str, str]]:
    s = expr.strip()
    if not s.startswith("\\frac"):
        return None
    idx = len("\\frac")
    if idx >= len(s) or s[idx] != "{":
        return None
    idx += 1
    depth = 1
    numerator_chars: List[str] = []
    while idx < len(s) and depth > 0:
        ch = s[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                idx += 1
                break
        if depth > 0:
            numerator_chars.append(ch)
        idx += 1
    if depth != 0:
        return None
    if idx >= len(s) or s[idx] != "{":
        return None
    idx += 1
    depth = 1
    denominator_chars: List[str] = []
    while idx < len(s) and depth > 0:
        ch = s[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                idx += 1
                break
        if depth > 0:
            denominator_chars.append(ch)
        idx += 1
    if depth != 0 or idx != len(s):
        return None
    numerator = "".join(numerator_chars).strip()
    denominator = "".join(denominator_chars).strip()
    if not numerator or not denominator:
        return None
    return numerator, denominator



def _derive_equation_rearrangements(eq: str) -> set:
    if not eq or "=" not in eq:
        return set()
    lhs, rhs = eq.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs:
        return set()
    rearrangements: set = set()
    rhs_frac = _extract_simple_fraction(rhs)
    if rhs_frac:
        numerator, denominator = rhs_frac
        rearrangements.add(f"{numerator}={lhs}{denominator}")
        rearrangements.add(f"{numerator}={lhs}*{denominator}")
        rearrangements.add(f"{denominator}={numerator}/{lhs}")
    lhs_frac = _extract_simple_fraction(lhs)
    if lhs_frac:
        numerator, denominator = lhs_frac
        rearrangements.add(f"{numerator}={rhs}{denominator}")
        rearrangements.add(f"{numerator}={rhs}*{denominator}")
        rearrangements.add(f"{denominator}={numerator}/{rhs}")
    return {r for r in rearrangements if r}



def _normalize_equation_text(eq: str) -> str:
    s = str(eq or "").strip()
    if not s:
        return ""
    s = s.replace("−", "-").replace("–", "-")
    greek_map = {
        "Δ": "\\Delta",
        "λ": "\\lambda",
        "ρ": "\\rho",
        "θ": "\\theta",
        "μ": "\\mu",
    }
    for k, v in greek_map.items():
        s = s.replace(k, v)
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\s+", "", s)
    return s



def _extract_equation_candidates(text: str) -> List[str]:
    if not text:
        return []
    regexes = _get_equation_regexes()
    candidates: List[str] = []
    seen: set = set()
    for key in ("latex_block", "latex_inline", "latex_paren"):
        for match in regexes[key].finditer(text):
            cand = (match.group(1) or "").strip()
            if cand:
                norm = _normalize_equation_text(cand)
                if norm in seen:
                    continue
                seen.add(norm)
                candidates.append(cand)
    return candidates



def _find_non_whitelisted_equations(text: str) -> List[str]:
    whitelist = _get_equation_whitelist(SUBJECT_EQUATIONS)
    if not whitelist:
        return []
    candidates = _extract_equation_candidates(text)
    non_whitelisted: List[str] = []
    for cand in candidates:
        if re.match(r"^\s*TOTAL\s*=", cand, flags=re.IGNORECASE):
            continue
        norm = _normalize_equation_text(cand)
        if norm and norm not in whitelist:
            non_whitelisted.append(cand)
    if non_whitelisted:
        LOGGER.info(
            "Equation whitelist validation found non-whitelisted equations",
            extra={"ctx": {"component": "equation_whitelist", "count": len(non_whitelisted)}},
        )
    else:
        LOGGER.info(
            "Equation whitelist validation passed",
            extra={"ctx": {"component": "equation_whitelist", "count": 0}},
        )
    return non_whitelisted


# ============================================================
# --- SELF CHECK + VALIDATION ---
# ============================================================

def _self_check_equations(question_text: str, markscheme_text: str, subject_pack: dict) -> List[str]:
    eq_pack = subject_pack or {}
    key_eqs = eq_pack.get("key_equations") or []
    whitelist = [str(e.get("latex", "") or "").strip() for e in key_eqs if isinstance(e, dict)]
    notation = [str(n).strip() for n in (eq_pack.get("notation_rules") or []) if str(n).strip()]
    prompt = _render_template(
        """
You are checking GCSE Physics equations for compliance with the official equation sheet.
List every equation explicitly used in the question and mark scheme.
Then verify each equation appears on the official equation sheet list provided.

Return JSON: {"violations":[{"equation":"...","reason":"..."}]}

Official equations (LaTeX):
<<WHITELIST>>

Notation rules / canonical forms:
<<NOTATION>>

Question:
<<QUESTION>>

Mark scheme:
<<MARKSCHEME>>
""",
        {
            "WHITELIST": "\n".join([f"- {w}" for w in whitelist]) or "(none)",
            "NOTATION": "\n".join([f"- {n}" for n in notation]) or "(none)",
            "QUESTION": question_text or "",
            "MARKSCHEME": markscheme_text or "",
        },
    ).strip()

    try:
        response = client.with_options(timeout=10).chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = safe_parse_json(raw) or {}
        violations = data.get("violations", [])
        if isinstance(violations, list):
            out = []
            for v in violations:
                if isinstance(v, dict):
                    eq = str(v.get("equation", "") or "").strip()
                    reason = str(v.get("reason", "") or "").strip()
                    if eq:
                        out.append(f"{eq} ({reason})" if reason else eq)
                else:
                    s = str(v).strip()
                    if s:
                        out.append(s)
            LOGGER.info(
                "Equation self-check completed",
                extra={"ctx": {"component": "equation_self_check", "count": len(out)}},
            )
            return out
    except Exception as exc:
        LOGGER.warning(
            "Equation self-check failed; skipping",
            extra={"ctx": {"component": "equation_self_check", "error": type(exc).__name__}},
        )
    return []



def _extract_total_from_marksheme(ms: str) -> Optional[int]:
    m = re.search(r"\btotal\b\s*[:=]\s*(\d+)\b", ms or "", flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None



def _has_part_marking(ms: str) -> bool:
    s = ms or ""
    return bool(re.search(r"(\([a-z]\)|\b[a-z]\))\s.*\[\s*\d+\s*\]", s, flags=re.IGNORECASE | re.DOTALL))



def _forbidden_found(q: str, ms: str) -> List[str]:
    t = (q or "") + "\n" + (ms or "")
    bad = []
    patterns = [
        (r"\\mu_0|\bmu0\b|\bμ0\b", "Uses μ0 (not GCSE)"),
        (r"\\epsilon_0|\bepsilon0\b|\bε0\b", "Uses ε0 (not GCSE)"),
        (r"\bB\s*=\s*\\mu_0\s*n\s*I\b|\bB\s*=\s*μ0\s*n\s*I\b", "Uses solenoid field equation B=μ0 n I (not GCSE)"),
        (r"\bflux\b|\bflux linkage\b|\binductance\b", "Uses flux/inductance language (not GCSE here)"),
        (r"\bFaraday\b|\bLenz\b", "Uses Faraday/Lenz law (not GCSE equation form here)"),
        (r"\bcalculus\b|\bdifferentiat|\bintegrat", "Uses calculus (not GCSE)"),            (r"\bz\s*=\s*(\\Delta|Δ)\s*\\lambda\s*/\s*\\lambda|\bz\s*=\s*(delta|Δ)\s*lambda\s*/\s*lambda|Δ\s*λ\s*/\s*λ|Δ\s*lambda\s*/\s*lambda", "Uses red-shift calculation z=Δλ/λ (not required at AQA GCSE; red-shift is qualitative)"),
        (r"\\frac\{1\}\{2\}\s*k\s*x\s*\^\s*2|\b0\.5\s*k\s*x\s*\^\s*2\b|\b1\s*/\s*2\s*k\s*x\s*\^\s*2\b|\bk\s*x\s*\^\s*2\b", "Uses elastic potential energy with x: use AQA notation Ee = 1/2 k e^2 (extension e)"),
        (r"\bF\s*=\s*k\s*x\b|\bF\s*=\s*kx\b", "Uses Hooke’s law with x: use AQA notation F = k e (extension e)"),

    ]
    # Add subject-pack forbidden patterns (equations.json), if provided
    try:
        fps = SUBJECT_EQUATIONS.get("forbidden_patterns") or []
        for fp in fps:
            if isinstance(fp, dict):
                rx = str(fp.get("regex", "") or "").strip()
                rs = str(fp.get("reason", "") or "").strip()
                if rx and rs:
                    patterns.append((rx, rs))
    except Exception:
        pass

    for pat, label in patterns:
        if re.search(pat, t, flags=re.IGNORECASE):
            bad.append(label)
    return bad



def _auto_check_warnings(question_text: str, markscheme_text: str, max_marks: int) -> List[str]:
    reasons: List[str] = []
    qtxt = str(question_text or "").strip()
    mstxt = str(markscheme_text or "").strip()

    if not qtxt:
        reasons.append("Missing question_text.")
    if not mstxt:
        reasons.append("Missing markscheme_text.")

    total = _extract_total_from_marksheme(mstxt)
    if total != int(max_marks):
        reasons.append(f"Mark scheme TOTAL must equal {int(max_marks)}.")

    if not _has_part_marking(mstxt):
        reasons.append("Mark scheme must include part-by-part marks like '(a) ... [2]'.")

    reasons.extend(_forbidden_found(qtxt, mstxt))

    non_whitelisted = _find_non_whitelisted_equations(f"{qtxt}\n{mstxt}")
    if non_whitelisted:
        offending = ", ".join(sorted(set(non_whitelisted)))
        reasons.append(f"Contains non-whitelisted equations: {offending}")

    if "$" in qtxt and "\\(" in qtxt:
        reasons.append("Use $...$ for LaTeX, avoid \\( ... \\).")

    return reasons


# ============================================================
# --- AI GENERATION ---
# ============================================================

def generate_practice_question_with_ai(
    topic_text: str,
    sub_topic_text: str | None,
    difficulty: str,
    qtype: str,
    marks: int,
    extra_instructions: str = "",
) -> Dict[str, Any]:
    track = st.session_state.get("track", TRACK_DEFAULT)
    requested_sub_topic = str(sub_topic_text or "").strip()
    extra_instructions = (extra_instructions or "").strip()
    if requested_sub_topic:
        extra_hint = f"Requested sub-topic: {requested_sub_topic}"
        extra_instructions = _append_instruction(extra_instructions, extra_hint)
    if requested_sub_topic.lower() == ATOMS_ISOTOPES_SUB_TOPIC.lower():
        extra_instructions = _append_instruction(extra_instructions, ATOMS_ISOTOPES_PHYSICS_ONLY)
    topic_options = get_topic_group_names_for_track(track)
    skill_options = list(SKILLS)
    difficulty_options = list(DIFFICULTIES)

    def _coerce_vocab(value: Any, allowed: List[str]) -> Optional[str]:
        val = str(value or "").strip()
        if not val:
            return None
        allowed_map = {str(opt).strip().lower(): str(opt).strip() for opt in allowed}
        return allowed_map.get(val.lower())

    def _validate(d: Dict[str, Any]) -> Tuple[bool, List[str]]:
        qtxt = str(d.get("question_text", "") or "")
        mstxt = str(d.get("markscheme_text", "") or "")
        reasons = _auto_check_warnings(qtxt, mstxt, int(marks))

        topic_val = _coerce_vocab(d.get("topic"), topic_options)
        if not topic_val:
            reasons.append("Missing or invalid topic group.")
        elif topic_text and topic_val.lower() != str(topic_text).strip().lower():
            reasons.append("topic must match the requested Topic group.")

        sub_topic_options = get_sub_topic_names_for_group(track, topic_val or topic_text)
        sub_topic_val = _coerce_vocab(d.get("sub_topic"), sub_topic_options)
        if not sub_topic_val:
            reasons.append("Missing or invalid sub_topic.")
        elif requested_sub_topic and sub_topic_val.lower() != requested_sub_topic.lower():
            reasons.append("sub_topic must match the requested Topic.")
        else:
            expected_group = get_topic_group_for_name(sub_topic_val)
            if expected_group and topic_val and expected_group.lower() != topic_val.lower():
                reasons.append(f"sub_topic must match the topic group '{topic_val}'.")

        skill_val = _coerce_vocab(d.get("skill"), skill_options)
        if not skill_val:
            reasons.append("Missing or invalid skill.")

        difficulty_val = _coerce_vocab(d.get("difficulty"), difficulty_options)
        if not difficulty_val:
            reasons.append("Missing or invalid difficulty.")
        elif difficulty and difficulty_val.lower() != str(difficulty).strip().lower():
            reasons.append("difficulty must match the requested Difficulty.")

        mm = d.get("max_marks", None)
        try:
            mm_int = int(mm)
        except Exception:
            mm_int = None
        if mm_int != int(marks):
            reasons.append(f"max_marks must equal {int(marks)}.")

        # Subject-pack forbidden patterns (equations.json): reject out-of-scope content early
        try:
            t_all = json.dumps(d, ensure_ascii=False)
            fps = SUBJECT_EQUATIONS.get("forbidden_patterns") or []
            for fp in fps:
                if isinstance(fp, dict):
                    rx = str(fp.get("regex", "") or "").strip()
                    rs = str(fp.get("reason", "") or "").strip()
                    if rx and rs and re.search(rx, t_all, flags=re.IGNORECASE):
                        reasons.append(f"Journey contains forbidden content: {rs}")
        except Exception:
            pass

        return (len(reasons) == 0), reasons

    def _call_model(
        repair: bool,
        reasons: Optional[List[str]] = None,
        extra_hint: str | None = None,
    ) -> Dict[str, Any]:
        system = _render_template(QGEN_SYSTEM_TPL, {
            "GCSE_ONLY_GUARDRAILS": GCSE_ONLY_GUARDRAILS,
            "MARKDOWN_LATEX_RULES": MARKDOWN_LATEX_RULES,
            "TRACK": st.session_state.get("track", TRACK_DEFAULT),
        })
        system = (system or "").strip()

        base_user = _render_template(QGEN_USER_TPL, {
            "TOPIC": (topic_text or "").strip(),
            "SUB_TOPIC": requested_sub_topic,
            "DIFFICULTY": str(difficulty),
            "QTYPE": str(qtype),
            "MARKS": int(marks),
            "EXTRA_INSTRUCTIONS": extra_instructions or "(none)",
            "TOPIC_OPTIONS": ", ".join(topic_options) or "(none)",
            "SUB_TOPIC_OPTIONS": ", ".join(get_sub_topic_names_for_group(track, topic_text)) or "(none)",
            "SKILL_OPTIONS": ", ".join(skill_options) or "(none)",
            "DIFFICULTY_OPTIONS": ", ".join(difficulty_options) or "(none)",
            "TRACK": st.session_state.get("track", TRACK_DEFAULT),
        })
        base_user = (base_user or "").strip()

        if not repair:
            user = base_user
        else:
            bullet_reasons = "\n".join([f"- {r}" for r in (reasons or [])]) or "- (unspecified)"
            user = _render_template(QGEN_REPAIR_PREFIX_TPL, {
                "BULLET_REASONS": bullet_reasons,
                "MARKS": int(marks),
            })
            user = (user or "").strip() + "\n\n" + base_user


        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=2500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = safe_parse_json(raw) or {}
        return data

    data = _call_model(repair=False)
    ok, reasons = _validate(data)
    self_check = _self_check_equations(
        str(data.get("question_text", "") or ""),
        str(data.get("markscheme_text", "") or ""),
        SUBJECT_EQUATIONS,
    )
    if self_check:
        reasons.append("Self-check found equation sheet violations: " + "; ".join(self_check))
        ok = False

    if not ok:
        data2 = _call_model(repair=True, reasons=reasons)
        ok2, reasons2 = _validate(data2)
        if ok2:
            data = data2
        else:
            data = data2 if isinstance(data2, dict) and data2 else data
            data["warnings"] = reasons2[:10]

    topic_val = _coerce_vocab(data.get("topic"), topic_options) or (topic_text or "").strip()
    sub_topic_options = get_sub_topic_names_for_group(track, topic_val)
    out = {
        "topic": topic_val,
        "sub_topic": _coerce_vocab(data.get("sub_topic"), sub_topic_options),
        "skill": _coerce_vocab(data.get("skill"), skill_options),
        "difficulty": _coerce_vocab(data.get("difficulty"), difficulty_options) or str(difficulty).strip(),
        "question_text": str(data.get("question_text", "") or "").strip(),
        "markscheme_text": str(data.get("markscheme_text", "") or "").strip(),
        "max_marks": int(marks),
        "tags": data.get("tags", []),
        "warnings": data.get("warnings", []),
    }
    if not isinstance(out["tags"], list):
        out["tags"] = []
    out["tags"] = [str(t).strip() for t in out["tags"] if str(t).strip()][:12]
    if not isinstance(out["warnings"], list):
        out["warnings"] = []
    out["warnings"] = [str(w) for w in out["warnings"]][:10]
    
    return out


# ============================================================
# TOPIC JOURNEY GENERATOR (teacher-only)
# ============================================================
DURATION_TO_STEPS = {10: 5}
JOURNEY_CHECKPOINT_EVERY = 3


def generate_topic_journey_with_ai(
    topic_plain_english: str,
    duration_minutes: int,
    extra_instructions: str = "",
) -> Dict[str, Any]:
    steps_n = DURATION_TO_STEPS.get(int(duration_minutes), 8)
    topic_plain_english = (topic_plain_english or "").strip()
    extra_instructions = (extra_instructions or "").strip()
    if ATOMS_ISOTOPES_SUB_TOPIC.lower() in topic_plain_english.lower():
        extra_instructions = _append_instruction(extra_instructions, ATOMS_ISOTOPES_PHYSICS_ONLY)
    def _validate(d: Dict[str, Any]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if not isinstance(d, dict):
            return False, ["Output is not a JSON object."]
        if str(d.get("topic", "")).strip() == "":
            reasons.append("Missing topic.")
        steps = d.get("steps", [])
        if not isinstance(steps, list) or len(steps) != steps_n:
            reasons.append(f"steps must be a list of length {steps_n}.")
            return False, reasons

        for i, stp in enumerate(steps):
            if not isinstance(stp, dict):
                reasons.append(f"Step {i+1} is not an object.")
                continue
            if not str(stp.get("objective", "")).strip():
                reasons.append(f"Step {i+1}: missing objective.")
            if not str(stp.get("question_text", "")).strip():
                reasons.append(f"Step {i+1}: missing question_text.")
            if not str(stp.get("markscheme_text", "")).strip():
                reasons.append(f"Step {i+1}: missing markscheme_text.")
            try:
                mm = int(stp.get("max_marks", 0))
            except Exception:
                mm = 0
            if mm <= 0 or mm > 12:
                reasons.append(f"Step {i+1}: max_marks must be 1-12.")
            ms = str(stp.get("markscheme_text", "") or "")
            if f"TOTAL = {mm}" not in ms:
                reasons.append(f"Step {i+1}: markscheme_text must end with 'TOTAL = {mm}'.")
            non_whitelisted = _find_non_whitelisted_equations(
                f"{stp.get('question_text', '')}\n{stp.get('markscheme_text', '')}"
            )
            if non_whitelisted:
                offending = ", ".join(sorted(set(non_whitelisted)))
                reasons.append(f"Step {i+1}: contains non-whitelisted equations: {offending}")
        return (len(reasons) == 0), reasons

    def _call_model(
        repair: bool,
        reasons: Optional[List[str]] = None,
        extra_hint: str | None = None,
    ) -> Dict[str, Any]:
        system = _render_template(JOURNEY_SYSTEM_TPL, {
            "GCSE_ONLY_GUARDRAILS": GCSE_ONLY_GUARDRAILS,
            "MARKDOWN_LATEX_RULES": MARKDOWN_LATEX_RULES,
            "TRACK": st.session_state.get("track", TRACK_DEFAULT),
        })
        system = (system or "").strip()

        base_user = _render_template(JOURNEY_USER_TPL, {
            "TOPIC_PLAIN": (topic_plain_english or "").strip(),
            "DURATION_MIN": int(duration_minutes),
            "STEPS_N": int(steps_n),
                    })
        base_user = (base_user or "").strip()
        if extra_instructions:
            base_user = base_user + "\n\nOptional constraints for the AI:\n" + extra_instructions

        if not repair:
            user = base_user
        else:
            bullet_reasons = "\n".join([f"- {r}" for r in (reasons or [])]) or "- (unspecified)"
            user = _render_template(JOURNEY_REPAIR_PREFIX_TPL, {
                "BULLET_REASONS": bullet_reasons,
                "STEPS_N": int(steps_n),
            })
            user = (user or "").strip() + "\n\n" + base_user
            if extra_hint:
                user = user + "\n\nExtra constraints:\n" + extra_hint.strip()


        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return safe_parse_json(raw) or {}

    data = _call_model(repair=False)
    ok, reasons = _validate(data)
    steps_for_check = data.get("steps", []) if isinstance(data, dict) else []
    if isinstance(steps_for_check, list):
        for idx, stp in enumerate(steps_for_check):
            if not isinstance(stp, dict):
                continue
            self_check = _self_check_equations(
                str(stp.get("question_text", "") or ""),
                str(stp.get("markscheme_text", "") or ""),
                SUBJECT_EQUATIONS,
            )
            if self_check:
                reasons.append(
                    f"Step {idx+1}: self-check found equation sheet violations: " + "; ".join(self_check)
                )
                ok = False
    if not ok:
        data2 = _call_model(repair=True, reasons=reasons)
        ok2, reasons2 = _validate(data2)
        if ok2:
            data = data2
        else:
            if isinstance(data2, dict):
                steps_candidate = data2.get("steps", None)
                steps_list = steps_candidate if isinstance(steps_candidate, list) else []
            else:
                steps_list = []
            needs_steps = (
                "steps must be a list" in " ".join(reasons2).lower()
                or len(steps_list) == 0
            )
            if needs_steps:
                extra_hint = (
                    "Return a complete steps list of exactly the requested length. "
                    "Keep each step concise (1-2 sentences per question/markscheme) to avoid truncation."
                )
                data3 = _call_model(repair=True, reasons=reasons2, extra_hint=extra_hint)
                ok3, reasons3 = _validate(data3)
                if ok3:
                    data = data3
                else:
                    data = data3 if isinstance(data3, dict) and data3 else data2
                    data["warnings"] = reasons3[:12]
            else:
                data = data2 if isinstance(data2, dict) and data2 else data
                data["warnings"] = reasons2[:12]

    # Final clean-up / normalization (display-time normalization will still run)
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    steps = steps[:steps_n]
    for stp in steps:
        if isinstance(stp, dict):
            stp["objective"] = str(stp.get("objective", "") or "").strip()
            stp["question_text"] = str(stp.get("question_text", "") or "").strip()
            stp["markscheme_text"] = str(stp.get("markscheme_text", "") or "").strip()
            try:
                stp["max_marks"] = int(stp.get("max_marks", 1))
            except Exception:
                stp["max_marks"] = 1
            if not isinstance(stp.get("misconceptions", []), list):
                stp["misconceptions"] = []

    data["steps"] = steps
    data["topic"] = (data.get("topic") or topic_plain_english).strip()
    data["checkpoint_every"] = int(data.get("checkpoint_every", JOURNEY_CHECKPOINT_EVERY) or JOURNEY_CHECKPOINT_EVERY)
    if data["checkpoint_every"] <= 0:
        data["checkpoint_every"] = JOURNEY_CHECKPOINT_EVERY

    return data
