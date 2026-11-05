import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import src.evaluation as evaluation
from src.evaluation import discover_images, infer_labels, make_split, remove_exact_duplicates, run_experiments


def synthetic_dataset(root, classes=2, per_class=5):
    root.mkdir()
    for label in range(classes):
        folder = root / f'class{label}'
        folder.mkdir()
        for i in range(per_class):
            # Distinct bytes and values, no duplicate copies across the split.
            pixels = np.random.default_rng(label * 100 + i).integers(0, 256, (32, 48, 3), dtype=np.uint8)
            pixels[:, :, label] //= 2
            Image.fromarray(pixels).save(folder / f'{i}.png')
    return root


def test_stratified_split_reproducible_disjoint_and_complete(tmp_path):
    paths = [str(tmp_path / label / f'{i}.png') for label in ('a', 'b') for i in range(10)]
    labels = [Path(path).parent.name for path in paths]
    first, second = make_split(paths, labels, 42), make_split(paths, labels, 42)
    assert {key: value.tolist() for key, value in first.items()} == {key: value.tolist() for key, value in second.items()}
    assert [len(first[name]) for name in first] == [12, 4, 4]
    union = set()
    for indices in first.values():
        assert not union.intersection(indices)
        union.update(indices)
        assert {labels[i] for i in indices} == {'a', 'b'}
    assert union == set(range(20))
    reversed_paths = list(reversed(paths))
    reversed_split = make_split(reversed_paths, list(reversed(labels)), 42)
    assert {name: {paths[i] for i in indices} for name, indices in first.items()} == {
        name: {reversed_paths[i] for i in indices} for name, indices in reversed_split.items()}


@pytest.mark.parametrize('size', [0, 1, 2])
def test_too_small_dataset_or_class_rejected(tmp_path, size):
    paths = [str(tmp_path / 'small' / str(i)) for i in range(size)]
    with pytest.raises(ValueError):
        make_split(paths, ['small'] * size)


def test_duplicate_files_removed_before_split(tmp_path):
    a, b, c = [tmp_path / n for n in ('a', 'b', 'c')]
    a.write_bytes(b'one')
    b.write_bytes(b'one')
    c.write_bytes(b'two')
    kept, removed, hashes = remove_exact_duplicates([str(a), str(b), str(c)])
    assert kept == [str(a), str(c)]
    assert removed[0]['retained_path'] == str(a)
    assert len(hashes) == 2


def test_root_images_make_labels_unavailable(tmp_path):
    assert infer_labels([str(tmp_path / 'root.png'), str(tmp_path / 'a' / '1.png')], tmp_path) is None
    assert infer_labels([str(tmp_path / 'a' / '1.png')], tmp_path) == ['a']


def test_discovery_excludes_output_tree(tmp_path):
    data = synthetic_dataset(tmp_path / 'data', classes=1, per_class=3)
    output = data / 'reports'
    output.mkdir()
    Image.new('RGB', (5, 5)).save(output / 'plot.png')
    assert len(discover_images(data, output)) == 3


def test_all_seven_configs_both_strategies_artifacts_and_no_leakage(tmp_path, monkeypatch):
    dataset = synthetic_dataset(tmp_path / 'data')
    calls = []
    original = evaluation.extract_dataset_blocks

    def counted(paths):
        calls.append(list(paths))
        return original(paths)

    monkeypatch.setattr(evaluation, 'extract_dataset_blocks', counted)
    output = tmp_path / 'report'
    metadata = run_experiments(dataset, output, k_values=[2], dataset_kind='synthetic')
    assert len(calls) == 1
    assert len(metadata['configurations']) == 7
    assert metadata['selection_data'] == 'gallery only'
    assert metadata['results_scope'] == 'synthetic smoke test only'
    assert set(metadata['descriptor_seconds_once']) == {'preprocessing', 'color', 'shape', 'texture'}
    manifest = pd.read_csv(output / 'split_manifest.csv')
    assert len(manifest) == 10 and not manifest.image_id.duplicated().any()
    ranks = pd.read_csv(output / 'rankings.csv')
    gallery_ids = set(manifest.loc[manifest.split == 'gallery', 'path'])
    query_ids = set(manifest.loc[manifest.split != 'gallery', 'path'])
    assert set(ranks.image_id) <= gallery_ids
    assert set(ranks.query_id) <= query_ids
    assert not any(ranks.query_id == ranks.image_id)
    metrics = pd.read_csv(output / 'retrieval_metrics.csv')
    assert len(metrics) == 7 * 2 * 2
    assert metrics['queries_evaluated'].min() > 0
    assert metrics['coverage'].min() == 1
    assert set(metrics.strategy) == {'feature', 'rank'}
    clustering = pd.read_csv(output / 'clustering_metrics.csv')
    assert clustering.configuration.nunique() == 7
    assert {'silhouette', 'davies_bouldin', 'calinski_harabasz', 'ari', 'nmi'} <= set(clustering)
    assert (output / 'clustering.png').stat().st_size > 100
    assert (output / 'retrieval.png').stat().st_size > 100
    assert json.loads((output / 'metadata.json').read_text(encoding='utf-8'))['seed'] == 42


def test_explicit_k_above_gallery_size_rejected(tmp_path):
    dataset = synthetic_dataset(tmp_path / 'data', classes=1, per_class=3)
    with pytest.raises(ValueError, match='gallery size'):
        run_experiments(dataset, tmp_path / 'report', configurations=[('color',)], k_values=[2], tasks=['clustering'])


def test_existing_output_reports_are_protected(tmp_path):
    output = tmp_path / 'report'
    output.mkdir()
    report = output / 'metadata.json'
    report.write_text('existing report', encoding='utf-8')
    with pytest.raises(ValueError, match='not empty'):
        run_experiments(tmp_path / 'missing-dataset', output)
    assert report.read_text(encoding='utf-8') == 'existing report'


def test_output_readme_is_allowed_without_erasing_it(tmp_path):
    dataset = synthetic_dataset(tmp_path / 'data', classes=1, per_class=3)
    output = tmp_path / 'report'
    output.mkdir()
    readme = output / 'README.md'
    readme.write_text('retained', encoding='utf-8')
    run_experiments(dataset, output, configurations=[('color',)], tasks=['retrieval'])
    assert readme.read_text(encoding='utf-8') == 'retained'


def test_retrieval_without_labels_keeps_rankings_and_marks_missing_metrics(tmp_path):
    nested = synthetic_dataset(tmp_path / 'nested', classes=1, per_class=5)
    for path in (nested / 'class0').glob('*.png'):
        path.rename(nested / path.name)
    output = tmp_path / 'report'
    metadata = run_experiments(nested, output, configurations=[('color',)], tasks=['retrieval'])
    assert metadata['stratified'] is False
    records = pd.read_csv(output / 'retrieval_metrics.csv')
    assert records['precision@5'].isna().all()
    assert (records.coverage == 0).all()
