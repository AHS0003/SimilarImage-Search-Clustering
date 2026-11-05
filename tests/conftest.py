"""Small deterministic images: no external dataset or model download required."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def image_factory(tmp_path):
    def create(name="sample.png", size=(120, 80), mode="RGB"):
        image = Image.new("RGB", size, (20, 30, 40))
        draw = ImageDraw.Draw(image)
        width, height = size
        draw.ellipse((width // 5, height // 5, 4 * width // 5, 4 * height // 5), fill=(230, 180, 70))
        draw.rectangle((width // 2, height // 4, 3 * width // 4, height // 2), fill=(80, 160, 210))
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.convert(mode).save(path)
        return path
    return create


@pytest.fixture
def texture_image(tmp_path):
    # Includes edges at several scales and enough grayscale levels for Otsu.
    y, x = np.indices((256, 256))
    pixels = ((x // 8 % 2) * 100 + (y // 16 % 2) * 80 + x // 4).astype(np.uint8)
    path = tmp_path / "texture.png"
    Image.fromarray(pixels).save(path)
    return path
