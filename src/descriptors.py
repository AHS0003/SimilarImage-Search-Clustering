"""Classical color, shape and texture descriptors and shared feature fusion.

SFTA follows Costa et al., SIBGRAPI 2012, doi:10.1109/SIBGRAPI.2012.15,
using boundary statistics as in https://github.com/alceufc/sfta. At nt=8 it
returns 45 features, matching the author's implementation (6*nt-3).
"""
import math
from time import perf_counter

import cv2
import mahotas as mt
import numpy as np

from .preprocessing import InvalidImageError, _as_rgb, letterbox, resize_preserving_aspect

DESCRIPTORS = ('color', 'shape', 'texture')
DIMENSIONS = {'color': 512, 'shape': 25, 'texture': 45}


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} doit être un entier strictement positif.')
    return int(value)


def zernike_dimension(degree=8):
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError('degree doit être un entier positif ou nul.')
    return sum(n // 2 + 1 for n in range(degree + 1))


class Color:
    def histogram(self, source):
        """512 RGB counts, padding excluded; L2 normalization is centralized."""
        rgb, valid = letterbox(_as_rgb(source), 256)
        return cv2.calcHist([rgb], [0, 1, 2], valid.astype(np.uint8),
                            [8, 8, 8], [0, 256] * 3).ravel().astype(np.float64)


class Shape:
    def zernike_moments(self, source, radius=None, degree=8):
        """Otsu foreground, border polarity, aspect preserved, absolute moments.

        Uniform images have no identifiable foreground and return zeros. The
        default circle encloses every foreground pixel around its centre of mass.
        """
        dimension = zernike_dimension(degree)
        gray = cv2.cvtColor(resize_preserving_aspect(_as_rgb(source), 128), cv2.COLOR_RGB2GRAY)
        if np.ptp(gray) == 0:
            return np.zeros(dimension, dtype=np.float64)
        _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
        if np.mean(border) > .5:
            mask = 1 - mask
        ys, xs = np.nonzero(mask)
        if not len(ys):
            return np.zeros(dimension, dtype=np.float64)
        cropped = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = cropped.shape
        scale = 120.0 / math.hypot(h, w)
        target = (max(1, round(w * scale)), max(1, round(h * scale)))
        # Nearest-neighbor downsampling can erase sparse foreground entirely.
        # Integrate its coverage before binarizing, retaining occupied pixels.
        fitted = cv2.resize(cropped.astype(np.float32), target,
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
        fitted = (fitted > 0).astype(np.uint8)
        canvas = np.zeros((128, 128), dtype=np.uint8)
        h, w = fitted.shape
        top, left = (128 - h) // 2, (128 - w) // 2
        canvas[top:top + h, left:left + w] = fitted
        ys, xs = np.nonzero(canvas)
        centre = (float(np.mean(ys)), float(np.mean(xs)))
        enclosing = max(1.0, float(np.hypot(ys - centre[0], xs - centre[1]).max()) + 1e-6)
        if radius is None:
            radius = enclosing
        elif not np.isfinite(radius) or radius < enclosing:
            raise ValueError('Le rayon Zernike doit contenir tout le premier plan.')
        values = mt.features.zernike_moments(canvas, radius, degree=degree, cm=centre)
        if values.shape != (dimension,) or not np.isfinite(values).all():
            raise ValueError('Moments de Zernike invalides.')
        return values


class Texture:
    @staticmethod
    def _recursive_otsu(gray, count):
        """Fixed recursive Otsu slots; degenerate branches repeat thresholds."""
        count = _positive_integer(count, 'num_thresh')
        histogram = np.bincount(gray.ravel(), minlength=256).astype(float)

        def split(low, high, slots):
            if not slots:
                return []
            hist = histogram[low:high + 1]
            occupied = np.flatnonzero(hist)
            if len(occupied) < 2:
                level = low + int(occupied[0]) if len(occupied) else low
                return [level] * slots
            probability = hist / hist.sum()
            omega = np.cumsum(probability)
            mu = np.cumsum(probability * np.arange(low, high + 1))
            denominator = omega * (1 - omega)
            variance = np.full(hist.size, -np.inf)
            valid = denominator > np.finfo(float).eps
            variance[valid] = (mu[-1] * omega[valid] - mu[valid]) ** 2 / denominator[valid]
            maxima = np.flatnonzero(np.isclose(variance, variance.max(), rtol=1e-12, atol=1e-12))
            threshold = low + int(np.floor(maxima.mean()))
            left = slots // 2
            return split(low, threshold, left) + [threshold] + split(threshold + 1, high, slots - 1 - left)

        return np.asarray(split(0, 255, count), dtype=float)

    @staticmethod
    def _boundary(mask):
        """8-neighbor inner boundary; exterior foreground avoids artificial frame."""
        binary = np.asarray(mask, dtype=np.uint8)
        interior = cv2.erode(binary, np.ones((3, 3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=1)
        return binary.astype(bool) & ~interior.astype(bool)

    @staticmethod
    def _box_counting(mask):
        """Slope of log N(e) versus log(1/e), at all dyadic box sizes."""
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() <= 1:
            return 0.0
        size = 1 << (max(mask.shape) - 1).bit_length()
        padded = np.zeros((size, size), dtype=bool)
        padded[:mask.shape[0], :mask.shape[1]] = mask
        scales = 2 ** np.arange(int(math.log2(size)) + 1)
        counts = [np.count_nonzero(padded.reshape(size // s, s, size // s, s).any(axis=(1, 3))) for s in scales]
        return float(np.clip(np.polyfit(np.log(1.0 / scales), np.log(counts), 1)[0], 0, 2))

    def sfta(self, source, num_thresh=8):
        """nt upper tails + nt-1 intervals; boundary dimension/mean/count.

        Returns 6*nt-3 values, matching the author's code (45 by default).
        Long edge=256; no synthetic padding pixels enter the statistics.
        """
        num_thresh = _positive_integer(num_thresh, 'num_thresh')
        gray = cv2.cvtColor(resize_preserving_aspect(_as_rgb(source), 256), cv2.COLOR_RGB2GRAY)
        thresholds = self._recursive_otsu(gray, num_thresh)
        masks = [gray > t for t in thresholds]
        masks += [(gray > lower) & (gray < upper) for lower, upper in zip(thresholds[:-1], thresholds[1:])]
        features = []
        for mask in masks:
            boundary = self._boundary(mask)
            pixels = gray[boundary]
            features.extend((self._box_counting(boundary), float(pixels.mean()) if pixels.size else 0.0,
                             float(pixels.size)))
        return np.asarray(features, dtype=np.float64)


def _configuration(weights=None, selected=None):
    if selected is None:
        selected = tuple(weights) if weights is not None else DESCRIPTORS
    requested = tuple(selected)
    if not requested or len(requested) != len(set(requested)) or set(requested) - set(DESCRIPTORS):
        raise ValueError('Sélection de descripteurs vide, dupliquée ou inconnue.')
    selected = tuple(name for name in DESCRIPTORS if name in requested)
    if weights is None:
        weights = {name: 1.0 for name in selected}
    if set(weights) != set(selected):
        raise ValueError('Un poids est requis pour chaque descripteur sélectionné uniquement.')
    try:
        weights = {name: float(weights[name]) for name in selected}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Les poids doivent être numériques.') from exc
    if any(not np.isfinite(value) or value < 0 for value in weights.values()) or not any(weights.values()):
        raise ValueError('Les poids doivent être finis, positifs ou nuls, et non tous nuls.')
    return selected, weights


def _normalize_block(block, name):
    """Validate dimensions, then L2-normalize each row; preserve zero vectors."""
    block = np.asarray(block, dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != DIMENSIONS[name] or not np.isfinite(block).all():
        raise ValueError(f'Descripteur {name} invalide : dimension ou valeurs non finies.')
    if not len(block):
        return block.copy()
    # Scale first so norm calculation cannot overflow for finite raw features.
    scales = np.max(np.abs(block), axis=1, keepdims=True)
    scaled = np.divide(block, scales, out=np.zeros_like(block), where=scales != 0)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norms, out=np.zeros_like(scaled), where=norms != 0)


def _extract_blocks(paths, selected, extractors, batch_size, skip_invalid):
    batch_size = _positive_integer(batch_size, 'batch_size')
    paths = list(paths)
    rows = {name: [] for name in selected}
    valid_paths, skipped = [], []
    times = {name: 0.0 for name in selected}
    times['preprocessing'] = 0.0
    for start in range(0, len(paths), batch_size):
        for path in paths[start:start + batch_size]:
            before = perf_counter()
            try:
                rgb = _as_rgb(path)
            except InvalidImageError as exc:
                times['preprocessing'] += perf_counter() - before
                if not skip_invalid:
                    raise
                skipped.append((path, str(exc)))
                continue
            times['preprocessing'] += perf_counter() - before
            for name in selected:
                before = perf_counter()
                raw = extractors[name](rgb)
                normalized = _normalize_block(np.asarray(raw)[None, :], name)[0]
                times[name] += perf_counter() - before
                rows[name].append(normalized)
            valid_paths.append(path)
    blocks = {name: np.vstack(rows[name]) if rows[name]
              else np.empty((0, DIMENSIONS[name]), dtype=np.float64) for name in selected}
    return blocks, valid_paths, skipped, times


def _default_extractors():
    return {'color': Color().histogram, 'shape': Shape().zernike_moments, 'texture': Texture().sfta}


def extract_descriptor_blocks(paths, batch_size=32, selected=None):
    """Return aligned, individually L2-normalized blocks, decoding each image once.

    This strict function raises InvalidImageError for a corrupt image. Extract
    all three blocks once and reuse them across descriptor combinations.
    """
    selected, _ = _configuration(selected=selected)
    blocks, _, _, _ = _extract_blocks(paths, selected, _default_extractors(), batch_size, False)
    return blocks


def extract_dataset_blocks(paths, batch_size=32, selected=None):
    """Return (blocks, valid_paths, skipped, descriptor_times).

    Corrupt files are excluded with aligned paths. Times are measured wall-clock
    seconds: each family includes its own resizing/extraction/L2 normalization;
    ``preprocessing`` is shared image decoding/orientation time. No labels enter
    these transformations. Empty datasets retain the declared block dimensions.
    """
    selected, _ = _configuration(selected=selected)
    return _extract_blocks(paths, selected, _default_extractors(), batch_size, True)


def combine_descriptor_blocks(blocks, weights=None, selected=None):
    """L2-normalize rows, apply weights, concatenate in color/shape/texture order.

    Accepts raw or already normalized cached blocks; repeated L2 normalization
    is harmless. There is no global normalization after concatenation.
    """
    selected, weights = _configuration(weights, selected)
    normalized = []
    count = None
    for name in selected:
        if name not in blocks:
            raise ValueError(f'Bloc manquant : {name}.')
        block = _normalize_block(blocks[name], name)
        if count is not None and len(block) != count:
            raise ValueError('Les blocs doivent contenir les mêmes images dans le même ordre.')
        count = len(block)
        normalized.append(block * weights[name])
    return np.concatenate(normalized, axis=1)


class FeatureExtractor:
    """One reusable transformation for dataset images and queries."""

    DESCRIPTORS = DESCRIPTORS
    DIMENSIONS = DIMENSIONS

    def __init__(self, weights=None, selected=None):
        self.selected, self.weights = _configuration(weights, selected)
        self.color, self.shape, self.texture = Color(), Shape(), Texture()

    @property
    def feature_dimension(self):
        return sum(DIMENSIONS[name] for name in self.selected)

    def _extract(self, paths, batch_size, skip_invalid):
        extractors = {'color': self.color.histogram, 'shape': self.shape.zernike_moments,
                      'texture': self.texture.sfta}
        active = tuple(name for name in self.selected if self.weights[name] > 0)
        blocks, valid, skipped, _ = _extract_blocks(paths, active, extractors, batch_size, skip_invalid)
        for name in self.selected:
            if name not in blocks:
                blocks[name] = np.zeros((len(valid), DIMENSIONS[name]), dtype=np.float64)
        return combine_descriptor_blocks(blocks, self.weights, self.selected), valid, skipped

    def extract_features(self, path):
        return self.extract_batch([path], batch_size=1)[0]

    def extract_batch(self, paths, batch_size=32):
        """Strict extraction; the first unreadable image raises InvalidImageError."""
        return self._extract(paths, batch_size, skip_invalid=False)[0]

    def extract_dataset(self, paths, batch_size=32):
        """Return features, aligned readable paths, and (path,error) exclusions."""
        return self._extract(paths, batch_size, skip_invalid=True)


