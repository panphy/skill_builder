"""Shared JSON parsing helpers.

Usage note:
    Import `safe_parse_json` from this module so JSON handling stays centralized.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional


def safe_parse_json(text_str: str) -> Optional[Any]:
    """Best-effort extractor for a JSON object from an LLM response."""
    s = (text_str or "").strip()
    if not s:
        return None

    # 1) Direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    def _prefer(obj_list: Iterable[Any]) -> Optional[Any]:
        """Prefer dicts containing a 'steps' list; else first dict."""
        objs = list(obj_list)
        for o in objs:
            if isinstance(o, dict) and isinstance(o.get("steps"), list) and len(o.get("steps")) > 0:
                return o
        for o in objs:
            if isinstance(o, dict):
                return o
        return objs[0] if objs else None

    # 2) Fenced json blocks
    objs = []
    for m in re.finditer(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE):
        try:
            objs.append(json.loads(m.group(1)))
        except Exception:
            continue
    picked = _prefer(objs)
    if picked is not None:
        return picked

    # 3) Balanced braces extraction: collect all top-level {...} objects
    candidates = []
    depth = 0
    in_str = False
    esc = False
    start = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(s[start : i + 1])
                    start = None

    parsed = []
    for cand in candidates:
        try:
            parsed.append(json.loads(cand))
        except Exception:
            continue

    picked = _prefer(parsed)
    if picked is not None:
        return picked

    return None
