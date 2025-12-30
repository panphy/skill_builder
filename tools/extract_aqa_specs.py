#!/usr/bin/env python3
"""
AQA GCSE subject-content -> "Spec Pack" JSON extractor

What it does
- Downloads the official AQA specification subject-content pages (HTML)
- Extracts the topic hierarchy (codes like 4.1.1.2) and titles
- Optionally extracts the "Content" and "Key opportunities for skills development" statements
- Writes JSON files compatible with PanPhy Skill Builder's "Spec Pack" loader

Important notes
- AQA specification text is copyrighted by AQA. This tool is intended for your internal teaching use.
- Be polite: this script rate-limits requests and supports caching.

Usage (recommended)
1) Create folders in your repo:
   specs/
   scripts/
2) Put this file at scripts/extract_aqa_specs.py
3) Install deps locally:
   pip install requests beautifulsoup4 lxml
4) Run:
   python scripts/extract_aqa_specs.py --out specs --cache .cache/aqa --include-content

You can also run a single subject:
   python scripts/extract_aqa_specs.py --out specs --subject aqa_gcse_physics

"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag


# -----------------------------
# Config
# -----------------------------
AQA_BASE = "https://www.aqa.org.uk"

SUBJECTS: Dict[str, Dict[str, str]] = {
    # Separate sciences
    "aqa_gcse_biology": {
        "tier": "higher",
        "start_url": "https://www.aqa.org.uk/subjects/biology/gcse/biology-8461/specification/subject-content",
        "mode": "multipage",
    },
    "aqa_gcse_chemistry": {
        "tier": "higher",
        "start_url": "https://www.aqa.org.uk/subjects/chemistry/gcse/chemistry-8462/specification/subject-content",
        "mode": "multipage",
    },
    "aqa_gcse_physics": {
        "tier": "higher",
        "start_url": "https://www.aqa.org.uk/subjects/physics/gcse/physics-8463/specification/subject-content",
        "mode": "multipage",
    },
    # Combined Science Trilogy (8464) has a single long page per subject
    "aqa_combined_biology": {
        "tier": "higher",
        "start_url": "https://www.aqa.org.uk/subjects/science/gcse/science-8464/specification/biology-subject-content",
        "mode": "singlepage",
    },
    "aqa_combined_chemistry": {
        "tier": "higher",
        "start_url": "https://www.aqa.org.uk/subjects/science/gcse/science-8464/specification/chemistry-subject-content",
        "mode": "singlepage",
    },
    "aqa_combined_physics": {
        "tier": "higher",
        "start_url": "https://www.aqa.org.uk/subjects/science/gcse/science-8464/specification/physics-subject-content",
        "mode": "singlepage",
    },
}

DEFAULT_COMMAND_WORDS = [
    "calculate", "compare", "complete", "describe", "determine", "draw", "estimate",
    "evaluate", "explain", "give", "identify", "justify", "label", "measure",
    "name", "plot", "predict", "show", "state", "suggest"
]

DEFAULT_MARK_SCHEME_RULES = [
    "Part-by-part marks with [x] for each part",
    "Total line must match max_marks exactly",
    "Calculation questions: method mark plus accuracy/units where appropriate",
    "No mark scheme text is ever shown to students"
]

DEFAULT_MUST_AVOID = [
    "Beyond-spec content",
    "A-level/IB calculus, differentiation, integration",
    "Unnecessary advanced constants and field equations unless explicitly in the spec section",
    "Copyrighted past paper questions (must be original)"
]


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    # Fix common split typo from HTML extraction (e.g. "F or example")
    s = s.replace("F or example", "For example")
    return s


def is_probably_main_content(tag: Tag) -> bool:
    if not isinstance(tag, Tag):
        return False
    name = tag.name.lower()
    return name in {"main", "article", "div", "section"}


def get_main_container(soup: BeautifulSoup) -> Tag:
    """
    AQA pages have lots of nav/footer noise. We'll try to focus on the actual spec content.
    """
    main = soup.find("main")
    if main:
        return main

    # Fallbacks
    article = soup.find("article")
    if article:
        return article

    body = soup.find("body")
    if body:
        return body

    return soup  # type: ignore


def absolutize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if u.startswith("/"):
        return AQA_BASE + u
    return AQA_BASE + "/" + u


def code_key(code: str) -> Tuple[int, ...]:
    parts = []
    for p in (code or "").split("."):
        try:
            parts.append(int(p))
        except Exception:
            parts.append(0)
    return tuple(parts) if parts else (0,)


CODE_TITLE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*)\s*$")
HT_ONLY_RE = re.compile(r"\(HT\s*only\)", flags=re.IGNORECASE)


@dataclass
class Node:
    code: str
    title: str
    ht_only: bool
    url: str
    level: int
    children: List["Node"]


def node_to_dict(n: Node) -> Dict:
    return {
        "code": n.code,
        "title": n.title,
        "ht_only": bool(n.ht_only),
        "url": n.url,
        "children": [node_to_dict(c) for c in n.children],
    }


def flatten_tree(nodes: List[Node]) -> List[Dict]:
    out: List[Dict] = []

    def walk(n: Node):
        out.append({"code": n.code, "title": n.title, "ht_only": bool(n.ht_only)})
        for c in n.children:
            walk(c)

    for n in nodes:
        walk(n)

    return out


def build_tree(nodes: List[Node]) -> List[Node]:
    """
    Build a hierarchy based on code depth. Example:
      4.1 -> level 2
      4.1.1 -> level 3
      4.1.1.2 -> level 4
    """
    # Deduplicate by code keeping first title/url/ht_only, but preserve smallest level
    by_code: Dict[str, Node] = {}
    for n in nodes:
        if n.code in by_code:
            existing = by_code[n.code]
            # Keep earliest URL if possible, but update if missing
            if not existing.url and n.url:
                existing.url = n.url
            # If any heading says HT only, treat as HT only
            existing.ht_only = existing.ht_only or n.ht_only
            # Prefer a longer, non-empty title
            if len(n.title) > len(existing.title):
                existing.title = n.title
            # Prefer smaller level (closer to root)
            existing.level = min(existing.level, n.level)
        else:
            by_code[n.code] = Node(
                code=n.code,
                title=n.title,
                ht_only=n.ht_only,
                url=n.url,
                level=n.level,
                children=[],
            )

    # Sort by code numeric order
    ordered = sorted(by_code.values(), key=lambda x: code_key(x.code))

    roots: List[Node] = []
    stack: List[Node] = []

    for n in ordered:
        # Clear children just in case
        n.children = []

        while stack and stack[-1].level >= n.level:
            stack.pop()

        if not stack:
            roots.append(n)
        else:
            stack[-1].children.append(n)

        stack.append(n)

    return roots


# -----------------------------
# Fetching with caching + politeness
# -----------------------------
def cache_path_for(cache_dir: Path, url: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")
    return cache_dir / f"{safe}.html"


def fetch_html(url: str, *, cache_dir: Optional[Path], sleep_s: float, timeout_s: float) -> str:
    url = absolutize_url(url)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_path_for(cache_dir, url)
        if cp.exists():
            return cp.read_text(encoding="utf-8", errors="ignore")

    headers = {
        "User-Agent": "PanPhySpecExtractor/1.0 (educational use; contact: teacher)",
        "Accept": "text/html,application/xhtml+xml",
    }

    # Basic retry
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=timeout_s)
            r.raise_for_status()
            html = r.text
            if cache_dir:
                cp = cache_path_for(cache_dir, url)
                cp.write_text(html, encoding="utf-8", errors="ignore")
            time.sleep(max(0.0, sleep_s))
            return html
        except Exception as e:
            last_err = e
            time.sleep(0.6 + attempt * 0.8)

    raise RuntimeError(f"Failed to fetch {url}: {type(last_err).__name__}: {last_err}")


# -----------------------------
# Parsing
# -----------------------------
def parse_code_title(text: str) -> Optional[Tuple[str, str, bool]]:
    """
    Parses headings like:
      '4.1 Energy'
      '6.7.2.2 Fleming\\'s left-hand rule (HT only)'
    Returns (code, title, ht_only)
    """
    t = normalize_ws(text)
    m = CODE_TITLE_RE.match(t)
    if not m:
        return None
    code = m.group(1)
    title = m.group(2).strip()
    ht_only = bool(HT_ONLY_RE.search(title))
    title = HT_ONLY_RE.sub("", title).strip()
    title = title.replace("  ", " ").strip()
    return code, title, ht_only


def find_topic_links_from_subject_content_index(html: str) -> List[str]:
    """
    For separate sciences, the 'subject-content' page links out to topic pages like:
      .../subject-content/energy
    We'll collect unique links of that form.
    """
    soup = BeautifulSoup(html, "lxml")
    main = get_main_container(soup)

    urls: List[str] = []
    for a in main.find_all("a", href=True):
        href = a.get("href", "")
        if not href:
            continue
        if "/specification/subject-content/" not in href:
            continue
        # Exclude the index itself which ends exactly with '/subject-content'
        if re.search(r"/specification/subject-content/?$", href):
            continue
        abs_url = absolutize_url(href)
        urls.append(abs_url)

    # Deduplicate preserving order
    seen = set()
    out = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def iter_text_blocks(main: Tag) -> List[Tuple[str, str]]:
    """
    Flatten relevant content into ordered blocks:
    - ("h", heading_text)
    - ("t", paragraph_text)
    - ("li", list_item_text)
    """
    blocks: List[Tuple[str, str]] = []

    # AQA pages are long, but within main content, headings and paragraphs/lists are relevant.
    for el in main.descendants:
        if not isinstance(el, Tag):
            continue
        name = el.name.lower()
        if name in {"h2", "h3", "h4", "h5", "h6"}:
            txt = normalize_ws(el.get_text(" ", strip=True))
            if txt:
                blocks.append(("h", txt))
        elif name == "p":
            txt = normalize_ws(el.get_text(" ", strip=True))
            if txt:
                blocks.append(("t", txt))
        elif name in {"li"}:
            txt = normalize_ws(el.get_text(" ", strip=True))
            if txt:
                blocks.append(("li", txt))

    # De-noise some obvious repeats by compressing consecutive duplicates
    cleaned: List[Tuple[str, str]] = []
    prev = None
    for b in blocks:
        if prev == b:
            continue
        cleaned.append(b)
        prev = b
    return cleaned


def extract_nodes_and_optional_content(
    html: str,
    url: str,
    include_content: bool,
) -> Tuple[List[Node], Dict[str, Dict[str, List[str]]]]:
    """
    Returns:
      - list of Nodes (topic hierarchy candidates)
      - content_map: {code: {"content": [...], "skills": [...], "refs": [...]}}
    """
    soup = BeautifulSoup(html, "lxml")
    main = get_main_container(soup)
    blocks = iter_text_blocks(main)

    nodes: List[Node] = []
    content_map: Dict[str, Dict[str, List[str]]] = {}

    current_code: Optional[str] = None
    mode: Optional[str] = None  # "content" | "skills" | None

    def ensure_entry(code: str):
        if code not in content_map:
            content_map[code] = {"content": [], "skills": [], "refs": []}

    for kind, txt in blocks:
        if kind == "h":
            parsed = parse_code_title(txt)
            if parsed:
                code, title, ht_only = parsed
                level = len(code.split("."))
                nodes.append(Node(code=code, title=title, ht_only=ht_only, url=url, level=level, children=[]))
                current_code = code
                mode = None
                if include_content:
                    ensure_entry(code)
            continue

        # Non-heading blocks
        if not include_content or not current_code:
            continue

        low = txt.lower().strip()
        if low == "content":
            mode = "content"
            continue
        if low.startswith("key opportunities for skills development"):
            mode = "skills"
            continue

        # WS/MS/AT refs often appear as standalone blocks, store separately
        if re.match(r"^(ws|ms|at)\s*\d", txt, flags=re.IGNORECASE):
            ensure_entry(current_code)
            content_map[current_code]["refs"].append(txt)
            continue

        # Discard obvious navigation noise
        if "show index" in low or low.startswith("specification") or low.startswith("planning resources"):
            continue

        ensure_entry(current_code)
        if mode == "skills":
            content_map[current_code]["skills"].append(txt)
        else:
            # default to content if mode not set; AQA pages put overview text above "Content"
            content_map[current_code]["content"].append(txt)

    # De-dup within each list while preserving order
    if include_content:
        for code, m in content_map.items():
            for k in ["content", "skills", "refs"]:
                seen = set()
                out: List[str] = []
                for item in m.get(k, []):
                    if item in seen:
                        continue
                    seen.add(item)
                    out.append(item)
                m[k] = out

    return nodes, content_map


# -----------------------------
# JSON assembly
# -----------------------------
def make_spec_pack(
    subject_key: str,
    tier: str,
    start_url: str,
    mode: str,
    cache_dir: Optional[Path],
    sleep_s: float,
    timeout_s: float,
    include_content: bool,
) -> Dict:
    source_urls = {"subject_content": start_url}

    all_nodes: List[Node] = []
    combined_content_map: Dict[str, Dict[str, List[str]]] = {}

    if mode == "singlepage":
        html = fetch_html(start_url, cache_dir=cache_dir, sleep_s=sleep_s, timeout_s=timeout_s)
        nodes, content_map = extract_nodes_and_optional_content(html, start_url, include_content=include_content)
        all_nodes.extend(nodes)
        combined_content_map.update(content_map)

    elif mode == "multipage":
        index_html = fetch_html(start_url, cache_dir=cache_dir, sleep_s=sleep_s, timeout_s=timeout_s)
        topic_urls = find_topic_links_from_subject_content_index(index_html)

        # Always parse the index itself too (it contains top-level headings)
        urls = [start_url] + topic_urls

        for u in urls:
            html = fetch_html(u, cache_dir=cache_dir, sleep_s=sleep_s, timeout_s=timeout_s)
            nodes, content_map = extract_nodes_and_optional_content(html, u, include_content=include_content)
            all_nodes.extend(nodes)
            # Merge content_map; prefer first-seen for each code list ordering
            for code, m in content_map.items():
                if code not in combined_content_map:
                    combined_content_map[code] = m
                else:
                    for k in ["content", "skills", "refs"]:
                        existing = combined_content_map[code].get(k, [])
                        for item in m.get(k, []):
                            if item not in existing:
                                existing.append(item)
                        combined_content_map[code][k] = existing

        source_urls["topic_pages"] = topic_urls

    else:
        raise ValueError(f"Unknown mode: {mode}")

    topic_tree_nodes = build_tree(all_nodes)
    allowed_topics = flatten_tree(topic_tree_nodes)

    out = {
        "spec_version": "aqa_html_extract_v1",
        "subject_key": subject_key,
        "tier": tier,
        "generated_at_utc": utc_now_iso(),
        "source_urls": source_urls,
        "topic_tree": [node_to_dict(n) for n in topic_tree_nodes],
        "allowed_topics": allowed_topics,
        "command_words": DEFAULT_COMMAND_WORDS,
        "mark_scheme_rules": DEFAULT_MARK_SCHEME_RULES,
        "must_avoid": DEFAULT_MUST_AVOID,
    }

    if include_content:
        out["content_map"] = combined_content_map

    return out


# -----------------------------
# CLI
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Extract AQA GCSE subject content into Spec Pack JSON.")
    parser.add_argument("--out", type=str, default="specs", help="Output folder for JSON files.")
    parser.add_argument("--cache", type=str, default="", help="Optional cache folder for fetched HTML.")
    parser.add_argument("--subject", type=str, default="", help="Extract only one subject_key (e.g. aqa_gcse_physics).")
    parser.add_argument("--include-content", action="store_true", help="Include extracted content/skills statements (large).")
    parser.add_argument("--sleep", type=float, default=0.4, help="Seconds to sleep between requests (politeness).")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache) if args.cache.strip() else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    subjects = SUBJECTS
    if args.subject.strip():
        key = args.subject.strip()
        if key not in SUBJECTS:
            raise SystemExit(f"Unknown subject_key '{key}'. Known: {', '.join(sorted(SUBJECTS.keys()))}")
        subjects = {key: SUBJECTS[key]}

    for subject_key, cfg in subjects.items():
        print(f"[extract] {subject_key} ...")
        pack = make_spec_pack(
            subject_key=subject_key,
            tier=cfg["tier"],
            start_url=cfg["start_url"],
            mode=cfg["mode"],
            cache_dir=cache_dir,
            sleep_s=float(args.sleep),
            timeout_s=float(args.timeout),
            include_content=bool(args.include_content),
        )

        out_path = out_dir / f"{subject_key}.json"
        out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}  (topics: {len(pack.get('allowed_topics', []))})")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
