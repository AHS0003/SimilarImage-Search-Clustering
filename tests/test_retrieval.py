from pathlib import Path

import numpy as np
import pytest

from src.retrieval import ImageRetriever, canonical_identity


def blocks(rows, descriptor='color'):
    dimensions = {'color': 512, 'shape': 25, 'texture': 45}
    output = np.zeros((len(rows), dimensions[descriptor]))
    output[:, :len(rows[0])] = rows
    return {descriptor: output}


@pytest.mark.parametrize('strategy', ['feature', 'rank'])
def test_no_self_and_deterministic_ties(tmp_path, strategy):
    paths = [str(tmp_path / name) for name in ('b.png', 'a.png', 'query.png')]
    data = blocks([[1, 0], [1, 0], [1, 0]])
    retriever = ImageRetriever(data, paths, selected=['color'], strategy=strategy)
    result = retriever.query(blocks([[1, 0]]), [str(tmp_path / '.' / 'query.png')], top_k=10)
    neighbors = next(iter(result.values()))
    assert [Path(n.image_id).name for n in neighbors] == ['a.png', 'b.png']
    assert all(canonical_identity(n.image_id) != canonical_identity(paths[-1]) for n in neighbors)


def test_feature_distances_are_true_euclidean_after_block_normalization(tmp_path):
    retriever = ImageRetriever(blocks([[10, 0], [0, 20], [-30, 0]]),
                               [str(tmp_path / str(i)) for i in range(3)],
                               selected=['color'], weights={'color': 2.0})
    neighbors = retriever.query(blocks([[5, 0]]), [str(tmp_path / 'q')])
    assert [n.distance for n in next(iter(neighbors.values()))] == pytest.approx([0, np.sqrt(8), 4])


def test_weighted_rrf_is_rank_score_with_constant_60(tmp_path):
    gallery = {**blocks([[1, 0], [0, 1]], 'color'),
               **blocks([[0, 1], [1, 0]], 'shape')}
    query = {**blocks([[1, 0]], 'color'), **blocks([[1, 0]], 'shape')}
    paths = [str(tmp_path / name) for name in ('a', 'b')]
    retriever = ImageRetriever(gallery, paths, selected=['color', 'shape'],
                               weights={'color': 1, 'shape': 3}, strategy='rank')
    neighbors = retriever.query(query, [str(tmp_path / 'q')])
    first, second = next(iter(neighbors.values()))
    assert first.image_id == paths[1]
    assert first.score == pytest.approx(1 / 62 + 3 / 61)
    assert second.score == pytest.approx(1 / 61 + 3 / 62)
    assert first.distance is None


@pytest.mark.parametrize('strategy', ['feature', 'rank'])
def test_raw_scale_does_not_change_rankings(tmp_path, strategy):
    paths = [str(tmp_path / str(i)) for i in range(2)]
    first = ImageRetriever(blocks([[1, 0], [0, 1]]), paths, selected=['color'], strategy=strategy)
    scaled = ImageRetriever(blocks([[9, 0], [0, 500]]), paths, selected=['color'], strategy=strategy)
    q = [str(tmp_path / 'q')]
    assert first.query(blocks([[1, 1]]), q) == scaled.query(blocks([[50, 50]]), q)


@pytest.mark.parametrize('kwargs', [{'strategy': 'unknown'}, {'rrf_constant': -1},
                                    {'rrf_constant': float('nan')}, {'weights': {'color': 0}}])
def test_invalid_configuration_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ImageRetriever(blocks([[1, 0]]), [str(tmp_path / 'a')], selected=['color'], **kwargs)


def test_unknown_descriptor_rejected_before_block_lookup(tmp_path):
    with pytest.raises(ValueError):
        ImageRetriever(blocks([[1, 0]]), [str(tmp_path / 'a')], selected=['unknown'])


@pytest.mark.parametrize('k', [0, -1, 1.5, True])
def test_invalid_neighbor_count(tmp_path, k):
    retriever = ImageRetriever(blocks([[1, 0]]), [str(tmp_path / 'a')], selected=['color'])
    with pytest.raises(ValueError):
        retriever.query(blocks([[1, 0]]), [str(tmp_path / 'q')], top_k=k)


def test_duplicate_canonical_gallery_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match='unique'):
        ImageRetriever(blocks([[1, 0], [0, 1]]), [str(tmp_path / 'a'), str(tmp_path / '.' / 'a')], selected=['color'])


def test_gallery_rows_and_query_rows_must_align(tmp_path):
    with pytest.raises(ValueError, match='row count'):
        ImageRetriever(blocks([[1, 0]]), [str(tmp_path / 'a'), str(tmp_path / 'b')], selected=['color'])
    retriever = ImageRetriever(blocks([[1, 0]]), [str(tmp_path / 'a')], selected=['color'])
    with pytest.raises(ValueError, match='row count'):
        retriever.query(blocks([[1, 0], [0, 1]]), [str(tmp_path / 'q')])
