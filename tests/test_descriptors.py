import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.descriptors import (
    Color,
    FeatureExtractor,
    Shape,
    Texture,
    combine_descriptor_blocks,
    extract_dataset_blocks,
    extract_descriptor_blocks,
)
from src.preprocessing import InvalidImageError, read_rgb


def test_empty_and_all_corrupt_datasets_preserve_dimensions(tmp_path):
    path = tmp_path / 'broken.png'
    path.write_bytes(b'bad image')
    extractor = FeatureExtractor(weights={'color': 1})
    X, valid, skipped = extractor.extract_dataset([path])
    assert X.shape == (0, 512)
    assert valid == []
    assert len(skipped) == 1 and skipped[0][0] == path
    assert extractor.extract_batch([]).shape == (0, 512)


def test_cached_blocks_match_all_seven_weighted_combinations(image_factory):
    from itertools import combinations
    paths = [image_factory(f'image{i}.png', size=(60 + i * 10, 100)) for i in range(3)]
    blocks = extract_descriptor_blocks(paths, batch_size=2)
    assert set(blocks) == {'color', 'shape', 'texture'}
    assert [blocks[name].shape for name in ('color', 'shape', 'texture')] == [(3, 512), (3, 25), (3, 45)]
    for length in (1, 2, 3):
        for selected in combinations(('color', 'shape', 'texture'), length):
            weights = {name: (index + 1) / 2 for index, name in enumerate(selected)}
            cached = combine_descriptor_blocks(blocks, weights=weights, selected=selected)
            direct = FeatureExtractor(weights=weights, selected=selected).extract_batch(paths)
            np.testing.assert_allclose(cached, direct, atol=1e-12)


def test_cache_decodes_once_and_reports_corrupt_paths(monkeypatch, image_factory, tmp_path):
    import src.descriptors as descriptors
    paths = [image_factory('first.png'), tmp_path / 'broken.png', image_factory('last.png')]
    paths[1].write_bytes(b'not an image')
    original = descriptors._as_rgb
    decoded = []

    def tracked(source):
        if not isinstance(source, np.ndarray):
            decoded.append(source)
        return original(source)

    monkeypatch.setattr(descriptors, '_as_rgb', tracked)
    blocks, valid, skipped, times = extract_dataset_blocks(paths, batch_size=2)
    assert decoded == paths
    assert valid == [paths[0], paths[2]]
    assert len(skipped) == 1 and skipped[0][0] == paths[1]
    assert all(len(block) == 2 for block in blocks.values())
    assert set(times) == {'preprocessing', 'color', 'shape', 'texture'}
    assert all(np.isfinite(value) and value >= 0 for value in times.values())


def test_strict_blocks_reject_corrupt_images(tmp_path):
    path = tmp_path / 'corrupt.png'
    path.write_bytes(b'bad png')
    with pytest.raises(InvalidImageError):
        extract_descriptor_blocks([path])


def test_zero_weight_avoids_unneeded_descriptor(monkeypatch, image_factory):
    extractor = FeatureExtractor(weights={'color': 1, 'texture': 0})
    monkeypatch.setattr(extractor.texture, 'sfta', lambda image: pytest.fail('Zero-weight texture must not run'))
    vector = extractor.extract_features(image_factory())
    assert vector.shape == (557,)
    np.testing.assert_array_equal(vector[512:], 0)


@pytest.mark.parametrize('blocks', [
    {'color': np.ones((2, 511))},
    {'color': np.full((2, 512), np.nan)},
    {'color': np.ones((2, 512)), 'shape': np.ones((3, 25))},
])
def test_invalid_cached_blocks_are_rejected(blocks):
    with pytest.raises(ValueError):
        combine_descriptor_blocks(blocks, selected=tuple(blocks))


@pytest.mark.parametrize("size", [(20, 80), (80, 20), (256, 256), (513, 277)])
@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_classical_dimensions_and_finite_values(image_factory, size, mode):
    path = image_factory(size=size, mode=mode)
    for vector, dimension in [(Color().histogram(path), 512),
                              (Shape().zernike_moments(path), 25),
                              (Texture().sfta(path), 45)]:
        assert vector.shape == (dimension,)
        assert np.issubdtype(vector.dtype, np.number)
        assert np.isfinite(vector).all()


@pytest.mark.parametrize("degree,dimension", [(0, 1), (1, 2), (2, 4), (8, 25), (9, 30), (10, 36)])
def test_zernike_dimension_matches_mahotas(image_factory, degree, dimension):
    result = Shape().zernike_moments(image_factory(), degree=degree)
    assert result.shape == (dimension,)


@pytest.mark.parametrize("level", [0, 73, 255])
def test_uniform_images_have_finite_fixed_dimension(tmp_path, level):
    path = tmp_path / "uniform.png"
    Image.new("L", (77, 123), level).save(path)
    shape = Shape().zernike_moments(path)
    texture = Texture().sfta(path)
    assert shape.shape == (25,)
    assert texture.shape == (45,)
    assert np.isfinite(shape).all() and np.isfinite(texture).all()
    assert np.allclose(shape, 0)


def test_zernike_rotation_invariance_and_foreground_polarity(tmp_path):
    image = Image.new("L", (192, 192), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 40, 72, 140), fill=255)
    draw.rectangle((48, 115, 125, 140), fill=255)
    paths = [tmp_path / name for name in ("shape.png", "rotated.png", "inverted.png")]
    image.save(paths[0])
    image.transpose(Image.Transpose.ROTATE_90).save(paths[1])
    Image.fromarray(255 - np.asarray(image)).save(paths[2])
    vectors = [Shape().zernike_moments(path) for path in paths]
    assert np.linalg.norm(vectors[0]) > 0
    np.testing.assert_allclose(vectors[0], vectors[1], atol=0.012, rtol=0.05)
    np.testing.assert_allclose(vectors[0], vectors[2], atol=0.012, rtol=0.05)


def test_color_padding_does_not_invent_colors(tmp_path):
    paths = []
    for i, size in enumerate([(40, 160), (160, 40), (64, 64)]):
        path = tmp_path / f"solid{i}.png"
        Image.new("RGB", size, (240, 30, 20)).save(path)
        paths.append(path)
    vectors = [Color().histogram(path) for path in paths]
    assert all(np.count_nonzero(vector) == 1 for vector in vectors)
    for vector in vectors[1:]:
        np.testing.assert_allclose(vector / np.linalg.norm(vector),
                                   vectors[0] / np.linalg.norm(vectors[0]))


def test_zernike_sparse_foreground_survives_circle_downsampling(tmp_path):
    # A 128x128 bounding box shrinks to 85x85 to fit the circle. With nearest
    # sampling, source coordinate 127 is never sampled and both dots disappear.
    gray = np.zeros((128, 128), dtype=np.uint8)
    gray[0, 127] = 255
    gray[127, 0] = 255
    path = tmp_path / "sparse.png"
    Image.fromarray(gray).save(path)
    vector = Shape().zernike_moments(path)
    assert vector.shape == (25,)
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) > 0


def test_texture_resolution_standardized(texture_image, tmp_path):
    larger = tmp_path / "larger.png"
    with Image.open(texture_image) as image:
        image.resize((512, 512), Image.Resampling.NEAREST).save(larger)
    np.testing.assert_allclose(Texture().sfta(texture_image), Texture().sfta(larger), atol=1e-8)


@pytest.mark.parametrize("num_thresh", [1, 2, 4, 8])
def test_sfta_configurable_constant_dimension(texture_image, num_thresh):
    vector = Texture().sfta(texture_image, num_thresh=num_thresh)
    assert vector.shape == (6 * num_thresh - 3,)
    assert np.isfinite(vector).all()
    assert np.count_nonzero(vector) > 0


@pytest.mark.parametrize("method", [Color().histogram, Shape().zernike_moments, Texture().sfta])
def test_corrupt_images_are_reported_not_converted_to_fake_vectors(tmp_path, method):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError):
        method(path)


def test_decode_exif_orientation_and_transparency_consistently(tmp_path):
    oriented = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (10, 20), (240, 10, 20))
    exif = image.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    image.save(oriented, exif=exif)
    assert read_rgb(oriented).shape == (10, 20, 3)
    transparent = tmp_path / "transparent.png"
    Image.new("RGBA", (10, 20), (255, 0, 0, 0)).save(transparent)
    np.testing.assert_array_equal(read_rgb(transparent), 255)


@pytest.mark.parametrize("bad", [np.full(512, np.nan), np.full(512, np.inf), np.zeros(511)])
def test_invalid_descriptor_output_is_rejected(monkeypatch, image_factory, bad):
    extractor = FeatureExtractor(weights={"color": 1})
    monkeypatch.setattr(extractor.color, "histogram", lambda image: bad)
    with pytest.raises(ValueError):
        extractor.extract_features(image_factory())


def test_each_block_is_l2_normalized_then_weighted(image_factory):
    path = image_factory()
    extractor = FeatureExtractor(weights={"color": 2.0, "shape": 0.5, "texture": 3.0})
    vector = extractor.extract_features(path)
    raw = [Color().histogram(path), Shape().zernike_moments(path), Texture().sfta(path)]
    expected = np.concatenate([weight * block / np.linalg.norm(block)
                               for weight, block in zip([2.0, 0.5, 3.0], raw)])
    np.testing.assert_allclose(vector, expected, atol=1e-12)
    assert vector.shape == (582,)
    np.testing.assert_allclose([np.linalg.norm(vector[:512]),
                               np.linalg.norm(vector[512:537]),
                               np.linalg.norm(vector[537:])], [2, 0.5, 3])


def test_selected_descriptor_order_is_canonical(image_factory):
    path = image_factory()
    first = FeatureExtractor(weights={"texture": 1, "color": 2})
    second = FeatureExtractor(weights={"color": 2, "texture": 1})
    np.testing.assert_array_equal(first.extract_features(path), second.extract_features(path))


@pytest.mark.parametrize("weights", [{}, {"color": 0}, {"color": -1}, {"color": np.nan},
                                     {"color": np.inf}, {"unknown": 1}, {"color": "invalid"}])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises((ValueError, TypeError)):
        FeatureExtractor(weights=weights)


@pytest.mark.parametrize("selected", [[], ["unknown"], ["color", "unknown"]])
def test_invalid_selection_rejected(selected):
    with pytest.raises(ValueError):
        FeatureExtractor(selected=selected)


def test_zero_weight_block_does_not_affect_vector(image_factory):
    path = image_factory()
    extractor = FeatureExtractor(weights={"color": 1, "texture": 0})
    vector = extractor.extract_features(path)
    assert np.isfinite(vector).all()
    # An explicitly selected zero-weight family retains a constant-size zero block.
    assert vector.shape == (557,)
    np.testing.assert_array_equal(vector[512:], 0)


def test_batch_dataset_and_query_transform_are_identical(image_factory):
    paths = [image_factory(f"image{i}.png", size=(70 + 10 * i, 100)) for i in range(3)]
    extractor = FeatureExtractor(weights={"color": 0.2, "shape": 0.3, "texture": 0.7})
    batch = extractor.extract_batch(paths, batch_size=2)
    dataset, valid_paths, skipped = extractor.extract_dataset(paths)
    assert list(valid_paths) == paths
    assert not skipped
    np.testing.assert_allclose(dataset, batch, atol=1e-12)
    for row, path in zip(dataset, paths):
        np.testing.assert_allclose(row, extractor.extract_features(path), atol=1e-12)


def test_dataset_skips_corrupt_files_without_label_misalignment(image_factory, tmp_path):
    first = image_factory("classA/one.png")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"bad png")
    last = image_factory("classB/two.png", size=(90, 75))
    extractor = FeatureExtractor(weights={"color": 1})
    X, valid, skipped = extractor.extract_dataset([first, broken, last])
    assert list(valid) == [first, last]
    assert X.shape == (2, 512)
    assert len(skipped) == 1 and skipped[0][0] == broken
    for row, path in zip(X, valid):
        np.testing.assert_allclose(row, extractor.extract_features(path))


def test_sfta_boundary_statistics_match_known_rectangle(tmp_path):
    gray = np.zeros((256, 256), dtype=np.uint8)
    gray[60:160, 70:150] = 200  # width80 height100, inner 8-neighbor boundary356 pixels
    path = tmp_path / "rectangle.png"
    Image.fromarray(gray).save(path)
    result = Texture().sfta(path, num_thresh=1)
    assert 0.8 < result[0] < 1.2  # finite-scale estimate of a one-dimensional border
    assert result[1] == pytest.approx(200)
    assert result[2] == 2 * 80 + 2 * 100 - 4


def test_sfta_uses_both_upper_tails_and_adjacent_intervals(monkeypatch, tmp_path):
    gray = np.zeros((256, 256), dtype=np.uint8)
    gray[30:90, 30:80] = 100
    gray[150:210, 150:200] = 200
    path = tmp_path / "two_levels.png"
    Image.fromarray(gray).save(path)
    monkeypatch.setattr(Texture, "_recursive_otsu", staticmethod(lambda image, count: np.array([50, 150])))
    result = Texture().sfta(path, num_thresh=2).reshape(-1, 3)
    boundary_pixels = 2 * 50 + 2 * 60 - 4
    # >50 contains both objects; >150 contains gray200; (50,150) contains gray100.
    np.testing.assert_allclose(result[:, 1], [150, 200, 100])
    np.testing.assert_allclose(result[:, 2], [2 * boundary_pixels, boundary_pixels, boundary_pixels])


def test_sfta_boundary_excludes_artificial_outer_frame():
    assert not Texture._boundary(np.ones((32, 32), dtype=bool)).any()
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    assert Texture._boundary(mask).sum() == 60


def test_box_counting_known_geometric_dimensions():
    empty = np.zeros((256, 256), dtype=bool)
    point = empty.copy()
    point[120, 120] = True
    line = empty.copy()
    line[128, :] = True
    plane = np.ones((256, 256), dtype=bool)
    assert Texture._box_counting(empty) == 0
    assert Texture._box_counting(point) == pytest.approx(0, abs=1e-8)
    assert Texture._box_counting(line) == pytest.approx(1, abs=1e-8)
    assert Texture._box_counting(plane) == pytest.approx(2, abs=1e-8)


def test_recursive_otsu_adapts_to_histogram_with_fixed_threshold_count():
    gray = np.zeros((256, 256), dtype=np.uint8)
    gray[:, 64:128] = 40
    gray[:, 128:192] = 170
    gray[:, 192:] = 240
    thresholds = np.asarray(Texture._recursive_otsu(gray, 8))
    assert thresholds.shape == (8,)
    assert np.isfinite(thresholds).all()
    assert np.all(np.diff(thresholds) >= 0)
    np.testing.assert_array_equal(thresholds, Texture._recursive_otsu(gray[::-1, ::-1], 8))
    assert len(np.unique(thresholds)) >= 3
