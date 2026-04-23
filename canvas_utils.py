import base64
import io

import numpy as np
from PIL import Image


CANVAS_BG_RGB = (255, 255, 255)
MAX_IMAGE_WIDTH = 1024


def data_url_to_image_data(data_url: str) -> np.ndarray:
    if not data_url or not isinstance(data_url, str):
        raise ValueError("Missing data URL")
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    _header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    return np.array(img)


def canvas_has_ink(image_data: np.ndarray) -> bool:
    if image_data is None:
        return False
    try:
        arr = np.asarray(image_data)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    except Exception:
        return False

    if arr.ndim != 3 or arr.shape[2] < 3:
        return False

    rgb = arr[:, :, :3]
    bg = np.array(CANVAS_BG_RGB, dtype=np.uint8)
    diff = np.max(np.abs(rgb.astype(np.int16) - bg.astype(np.int16)), axis=2)
    ink_pixels = int(np.count_nonzero(diff > 10))
    return ink_pixels >= 25


def preprocess_canvas_image(image_data: np.ndarray) -> Image.Image:
    raw_img = Image.fromarray(np.asarray(image_data).astype("uint8"))
    if raw_img.mode == "RGBA":
        white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
        white_bg.paste(raw_img, mask=raw_img.split()[3])
        img = white_bg
    else:
        img = raw_img.convert("RGB")
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / img.width
        img = img.resize((MAX_IMAGE_WIDTH, max(1, int(img.height * ratio))))
    return img
