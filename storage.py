import io
import logging
import re
from typing import Optional

import streamlit as st
from PIL import Image

from config import _safe_secret


LOGGER = logging.getLogger("panphy")
STORAGE_BUCKET = "physics-bank"


@st.cache_resource
def get_supabase_client():
    url = (_safe_secret("SUPABASE_URL", "") or "").strip()
    key = (_safe_secret("SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
    except Exception:
        return None
    return create_client(url, key)


def supabase_ready() -> bool:
    return get_supabase_client() is not None


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "untitled"


def _clean_storage_path(path: str) -> str:
    if not isinstance(path, str):
        return ""
    cleaned = path.strip().lstrip("/")
    cleaned = cleaned.replace("\\", "/")
    return re.sub(r"/{2,}", "/", cleaned)


def upload_to_storage(path: str, file_bytes: bytes, content_type: str) -> bool:
    sb = get_supabase_client()
    if sb is None:
        st.session_state["db_last_error"] = "Supabase Storage not configured."
        return False

    cleaned_path = _clean_storage_path(path)
    if not cleaned_path:
        st.session_state["db_last_error"] = "Storage Upload Error: empty path."
        return False

    file_options = {
        "contentType": str(content_type),
        "content-type": str(content_type),
        "cacheControl": "3600",
        "cache-control": "3600",
        "upsert": "true",
        "x-upsert": "true",
    }

    try:
        try:
            res = sb.storage.from_(STORAGE_BUCKET).upload(cleaned_path, file_bytes, file_options)
        except TypeError:
            res = sb.storage.from_(STORAGE_BUCKET).upload(
                path=cleaned_path,
                file=file_bytes,
                file_options=file_options,
            )

        err = None
        if hasattr(res, "error"):
            err = getattr(res, "error")
        elif isinstance(res, dict):
            err = res.get("error")
        if err:
            raise RuntimeError(str(err))
        return True
    except Exception as exc:
        st.session_state["db_last_error"] = f"Storage Upload Error: {type(exc).__name__}"
        LOGGER.exception(
            "Storage upload failed",
            extra={
                "ctx": {
                    "component": "storage",
                    "op": "upload",
                    "path": cleaned_path,
                    "error": type(exc).__name__,
                }
            },
        )
        return False


def download_from_storage(path: str) -> bytes:
    sb = get_supabase_client()
    if sb is None:
        return b""

    cleaned_path = _clean_storage_path(path)
    if not cleaned_path:
        return b""

    try:
        res = sb.storage.from_(STORAGE_BUCKET).download(cleaned_path)

        if isinstance(res, (bytes, bytearray)):
            return bytes(res)

        if hasattr(res, "data") and res.data is not None:
            if isinstance(res.data, (bytes, bytearray)):
                return bytes(res.data)

        if hasattr(res, "content") and res.content is not None:
            if isinstance(res.content, (bytes, bytearray)):
                return bytes(res.content)

        if hasattr(res, "read"):
            try:
                out = res.read()
                if isinstance(out, (bytes, bytearray)):
                    return bytes(out)
            except Exception:
                pass

        return b""
    except Exception as exc:
        st.session_state["db_last_error"] = f"Storage Download Error: {type(exc).__name__}"
        LOGGER.exception(
            "Storage download failed",
            extra={
                "ctx": {
                    "component": "storage",
                    "op": "download",
                    "path": cleaned_path,
                    "error": type(exc).__name__,
                }
            },
        )
        return b""


@st.cache_data(ttl=300)
def cached_download_from_storage(path: str, _fp: str = "") -> bytes:
    return download_from_storage(path)


def bytes_to_pil(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    img.load()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def safe_bytes_to_pil(img_bytes: bytes) -> Optional[Image.Image]:
    if not img_bytes:
        return None
    try:
        return bytes_to_pil(img_bytes)
    except Exception as exc:
        LOGGER.error(
            "Failed to decode image bytes",
            extra={"ctx": {"component": "image", "error": type(exc).__name__}},
        )
        return None
