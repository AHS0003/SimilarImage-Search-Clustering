import numpy as np
import pytest
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.clustering import clustering_metrics, evaluate_clustering


@pytest.fixture
def separated_gallery():
    return np.array([[0, 0], [0, .1], [.1, 0], [10, 10], [10, 10.1], [10.1, 10]])


def test_all_five_metrics_match_sklearn_on_known_gallery(separated_gallery):
    labels = np.array([0, 0, 0, 1, 1, 1])
    truth = np.array(["A", "A", "A", "B", "B", "B"])
    report = clustering_metrics(separated_gallery, labels, truth)
    assert report["silhouette"] == pytest.approx(silhouette_score(separated_gallery, labels, metric="euclidean"))
    assert report["davies_bouldin"] == pytest.approx(davies_bouldin_score(separated_gallery, labels))
    assert report["calinski_harabasz"] == pytest.approx(calinski_harabasz_score(separated_gallery, labels))
    assert report["ari"] == pytest.approx(adjusted_rand_score(truth, labels))
    assert report["nmi"] == pytest.approx(normalized_mutual_info_score(truth, labels))


@pytest.mark.parametrize("labels", [[0], [0, 0, 0], [0, 1, 2]])
def test_undefined_internal_metrics_are_null(labels):
    X = np.arange(2 * len(labels), dtype=float).reshape(-1, 2)
    result = clustering_metrics(X, labels)
    assert result["silhouette"] is None
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None
    assert result["ari"] is None and result["nmi"] is None


def test_semantic_scores_do_not_require_internal_metric_validity():
    result = clustering_metrics([[0], [1], [2]], [0, 0, 0], ["A", "A", "B"])
    assert result["silhouette"] is None
    assert result["ari"] == pytest.approx(0)
    assert np.isfinite(result["nmi"])


def test_auto_k_is_reproducible_and_uses_finite_gallery_silhouette(separated_gallery):
    first = evaluate_clustering(separated_gallery, [1, 2, 3, 4])
    second = evaluate_clustering(separated_gallery, [4, 3, 2, 1])
    assert first.selected_k == second.selected_k == 2
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.model.random_state == 42
    assert first.model.n_init == 10
    assert first.model.algorithm == "lloyd"
    np.testing.assert_allclose(first.model.cluster_centers_, second.model.cluster_centers_)
    assert [row["k"] for row in first.records] == [1, 2, 3, 4]


def test_truth_labels_are_reporting_only_and_cannot_select_k(separated_gallery):
    first = evaluate_clustering(separated_gallery, [2, 3], true_labels=[0, 0, 0, 1, 1, 1])
    second = evaluate_clustering(separated_gallery, [2, 3], true_labels=[0, 1, 0, 1, 0, 1])
    assert first.selected_k == second.selected_k
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.records[0]["ari"] != second.records[0]["ari"]


def test_equal_silhouette_selects_lower_k(monkeypatch, separated_gallery):
    import src.clustering as module
    original = module.clustering_metrics

    def tied_metrics(X, labels, truth=None):
        result = original(X, labels, truth)
        result["silhouette"] = 0.5
        return result

    monkeypatch.setattr(module, "clustering_metrics", tied_metrics)
    assert module.evaluate_clustering(separated_gallery, [3, 2]).selected_k == 2


def test_manual_k_bypasses_silhouette_selection(separated_gallery):
    result = evaluate_clustering(separated_gallery, [2], manual_k=3)
    assert result.selected_k == 3
    assert result.model.n_clusters == 3
    assert "Manual" in result.selection_reason
    assert [row["k"] for row in result.records] == [2, 3]


def test_one_image_and_no_candidates_fall_back_explicitly():
    result = evaluate_clustering([[1, 2]], [])
    assert result.selected_k == 1
    assert result.records[0]["inertia"] == 0
    assert "fallback" in result.selection_reason
    assert result.records[0]["silhouette"] is None


def test_constant_gallery_does_not_invent_silhouette_scores():
    result = evaluate_clustering(np.ones((4, 5)), [1, 2, 3, 4])
    assert result.selected_k == 1
    assert all(row["silhouette"] is None for row in result.records)
    assert all(row["n_clusters"] == 1 for row in result.records)
    assert result.records[1]["warning"]


def test_collapsed_candidate_is_reported_but_not_selected():
    X = np.array([[0], [0], [0], [10], [10], [10]], dtype=float)
    result = evaluate_clustering(X, [3])
    assert result.selected_k == 1
    collapsed = next(row for row in result.records if row["k"] == 3)
    assert collapsed["n_clusters"] == 2
    assert collapsed["silhouette"] is not None
    assert collapsed["warning"]


@pytest.mark.parametrize("k", [0, -1, 7, 2.5, True, "2"])
def test_invalid_k_rejected_instead_of_silently_clamped(separated_gallery, k):
    with pytest.raises(ValueError, match="gallery size"):
        evaluate_clustering(separated_gallery, [k])
    with pytest.raises(ValueError, match="gallery size"):
        evaluate_clustering(separated_gallery, [], manual_k=k)


@pytest.mark.parametrize("X", [[], [1, 2], np.empty((0, 2)), np.empty((2, 0)), [[np.nan]], [[np.inf]]])
def test_invalid_feature_matrices_rejected(X):
    with pytest.raises(ValueError, match="feature matrix"):
        evaluate_clustering(X, [])


def test_label_length_mismatch_is_not_silently_ignored(separated_gallery):
    with pytest.raises(ValueError, match="one label"):
        clustering_metrics(separated_gallery, [0, 1])
    with pytest.raises(ValueError, match="one label"):
        evaluate_clustering(separated_gallery, [2], true_labels=["A"])


def test_experiment_fits_actual_gallery_rows_and_never_query_rows(tmp_path, monkeypatch):
    """Instrument fits, rather than trusting a metadata 'gallery only' assertion."""
    import pandas as pd
    from PIL import Image

    import src.evaluation as experiment

    dataset = tmp_path / "dataset"
    for class_index in range(2):
        folder = dataset / f"class{class_index}"
        folder.mkdir(parents=True)
        for index in range(5):
            # Unique file bytes prevent the deliberate exact-duplicate filter.
            Image.new("RGB", (20 + index, 20), (20 + index, class_index * 100, 0)).save(folder / f"{index}.png")
    source_paths = []

    def sentinel_features(paths):
        source_paths.extend(paths)
        blocks = {name: np.eye(len(paths), dimension) for name, dimension in
                  (("color", 512), ("shape", 25), ("texture", 45))}
        return blocks, paths, [], {"preprocessing": 0., "color": 0., "shape": 0., "texture": 0.}

    clustering_inputs, fitted_retrievers, query_batches = [], [], []
    original_clustering = experiment.evaluate_clustering
    original_retriever = experiment.ImageRetriever

    def spy_clustering(X, *args, **kwargs):
        clustering_inputs.append(X.copy())
        return original_clustering(X, *args, **kwargs)

    class SpyRetriever(original_retriever):
        def __init__(self, gallery_blocks, gallery_ids, **kwargs):
            super().__init__(gallery_blocks, gallery_ids, **kwargs)
            fitted_retrievers.append(self)

        def query(self, query_blocks, query_ids, **kwargs):
            fitted_sizes = {name: model.n_samples_fit_ for name, model in self.models.items()}
            query_batches.append(set(query_ids))
            result = super().query(query_blocks, query_ids, **kwargs)
            assert fitted_sizes == {name: model.n_samples_fit_ for name, model in self.models.items()}
            return result

    monkeypatch.setattr(experiment, "extract_dataset_blocks", sentinel_features)
    monkeypatch.setattr(experiment, "evaluate_clustering", spy_clustering)
    monkeypatch.setattr(experiment, "ImageRetriever", SpyRetriever)
    output = tmp_path / "report"
    experiment.run_experiments(dataset, output, configurations=[("color",)], k_values=[2],
                               dataset_kind="synthetic")
    manifest = pd.read_csv(output / "split_manifest.csv")
    gallery_paths = set(manifest.loc[manifest.split == "gallery", "path"])
    query_paths = set(manifest.loc[manifest.split != "gallery", "path"])
    expected_indices = [index for index, path in enumerate(source_paths) if path in gallery_paths]
    assert len(clustering_inputs) == 1
    np.testing.assert_array_equal(clustering_inputs[0], np.eye(len(source_paths), 512)[expected_indices])
    assert len(expected_indices) == 6 < len(source_paths)
    assert len(fitted_retrievers) == 2  # feature and rank strategy, fixed gallery each
    assert all(set(retriever.gallery_ids) == gallery_paths for retriever in fitted_retrievers)
    assert set.union(*query_batches) == query_paths
    assert not gallery_paths.intersection(query_paths)
