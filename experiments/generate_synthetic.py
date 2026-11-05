"""Generate deliberately simple smoke data, never a real-performance benchmark."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_dataset(output, per_class=10, seed=42):
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 3:
        raise ValueError('per_class must be an integer >= 3.')
    root = Path(output)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError('Choose a new empty directory for synthetic images.')
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for label, color in enumerate(((200, 50, 40), (40, 180, 50), (50, 60, 190))):
        folder = root / f'synthetic_class_{label}'
        folder.mkdir()
        for i in range(per_class):
            width, height = 80 + i * 3, 64 + i * 2
            noise = rng.integers(-20, 21, (height, width, 3))
            pixels = np.clip(np.asarray(color) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            shift = int(rng.integers(0, 8))
            box = (12 + shift, 12, width - 15, height - 12)
            if label == 0:
                draw.ellipse(box, fill=(240, 220, 160))
            elif label == 1:
                draw.rectangle(box, fill=(170, 230, 220))
            else:
                for x in range(8 + shift, width - 8, 8):
                    draw.line((x, 10, x, height - 10), fill=(200, 170, 230), width=3)
            image.save(folder / f'{i:03d}.png')
    (root / 'SYNTHETIC_ONLY.txt').write_text(
        f'Synthetic smoke data only. Not a real image benchmark. seed={seed}; per_class={per_class}\n',
        encoding='utf-8',
    )
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-class', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    try:
        generate_dataset(args.output, args.per_class, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print('Synthetic smoke images generated; do not report as real-dataset performance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
