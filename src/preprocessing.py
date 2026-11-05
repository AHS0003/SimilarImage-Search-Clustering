"""Deterministic RGB image decoding and aspect-preserving resizing.

Resizing pixels and normalizing descriptor vectors are separate operations.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    """An image cannot be decoded; do not substitute invented features."""


def read_rgb(path):
    """Decode first frame once, orient using EXIF, composite alpha on white."""
    try:
        with Image.open(path) as image:
            image.seek(0)
            image = ImageOps.exif_transpose(image)
            if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                rgba = image.convert('RGBA')
                image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba)
            rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f'Image illisible : {path}') from exc
    if not rgb.size:
        raise InvalidImageError(f'Image vide : {path}')
    return rgb


def _as_rgb(source):
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3 or not source.size or source.dtype != np.uint8:
            raise InvalidImageError('Un tableau RGB uint8 non vide est requis.')
        return source
    return read_rgb(source)


def resize_preserving_aspect(rgb, size=256):
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    target = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def letterbox(rgb, size):
    fitted = resize_preserving_aspect(rgb, size)
    h, w = fitted.shape[:2]
    top, left = (size - h) // 2, (size - w) // 2
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    valid = np.zeros((size, size), dtype=bool)
    canvas[top:top + h, left:left + w] = fitted
    valid[top:top + h, left:left + w] = True
    return canvas, valid
