"""Leakage-free experiment orchestration and reproducible artifact export."""
import hashlib
import json
import platform
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .clustering import evaluate_clustering
from .descriptors import combine_descriptor_blocks, extract_dataset_blocks
from .metrics import evaluate_rankings
from .retrieval import ImageRetriever, canonical_identity

DESCRIPTORS = ('color', 'shape', 'texture')
CONFIGURATIONS = tuple(config for size in range(1, 4) for config in combinations(DESCRIPTORS, size))
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}


def discover_images(dataset, output=None):
    root = Path(dataset).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    output = Path(output).resolve() if output else None
    paths, seen = [], set()
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if output and (path.resolve() == output or output in path.resolve().parents):
            continue
        identity = canonical_identity(path)
        if identity not in seen:
            paths.append(str(path.resolve()))
            seen.add(identity)
    if not paths:
        raise ValueError('Dataset contains no supported image files.')
    return paths


def remove_exact_duplicates(paths):
    """Remove byte-identical files before splitting, retaining first sorted path.

    This detects exact file duplicates, not perceptual duplicates or re-encodings.
    Unreadable files continue to the decoder, which reports them as corrupt.
    """
    kept, duplicates, hashes, first = [], [], {}, {}
    for path in paths:
        try:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            kept.append(path)
            continue
        if digest in first:
            duplicates.append({'path': path, 'retained_path': first[digest], 'sha256': digest})
        else:
            first[digest] = path
            hashes[path] = digest
            kept.append(path)
    return kept, duplicates, hashes


def infer_labels(paths, dataset):
    """Immediate parent labels, only when every image is in a subdirectory."""
    root = Path(dataset).resolve()
    if any(Path(path).parent == root for path in paths):
        return None
    return [Path(path).parent.name for path in paths]


def _split_counts(count):
    if count < 3:
        raise ValueError('Each split group needs at least 3 readable unique images (gallery/validation/test).')
    desired = np.array([.6, .2, .2]) * count
    counts = np.floor(desired).astype(int)
    for i in sorted(range(3), key=lambda i: (-(desired[i] - counts[i]), i))[:count - counts.sum()]:
        counts[i] += 1
    for i in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[i] += 1
    return counts


def make_split(paths, labels=None, seed=42):
    """Deterministic 60/20/20 split, stratified when labels are available.

    Integer rounding keeps at least one image in every split for every class.
    No query image is used to fit K-Means, choose K, or fit nearest neighbors.
    """
    paths = [str(path) for path in paths]
    if len(paths) < 3:
        raise ValueError('At least 3 readable unique images are required for three disjoint splits.')
    identities = [canonical_identity(path) for path in paths]
    if len(identities) != len(set(identities)):
        raise ValueError('Split input must contain unique canonical image identities.')
    if labels is not None and len(labels) != len(paths):
        raise ValueError('Label count must match image count.')
    groups = defaultdict(list)
    for i in sorted(range(len(paths)), key=lambda i: identities[i]):
        groups[str(labels[i]) if labels is not None else '__unlabelled__'].append(i)
    rng = np.random.default_rng(seed)
    result = {'gallery': [], 'validation': [], 'test': []}
    for label, group in sorted(groups.items()):
        try:
            counts = _split_counts(len(group))
        except ValueError as exc:
            raise ValueError(f'Class/group {label!r}: {exc}') from exc
        shuffled = rng.permutation(group)
        boundaries = np.cumsum(counts)
        for name, indices in zip(result, np.split(shuffled, boundaries[:-1])):
            result[name].extend(int(i) for i in indices)
    return {name: np.array(sorted(indices, key=lambda i: identities[i]), dtype=int)
            for name, indices in result.items()}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(rows, path, columns=None):
    frame = pd.DataFrame(rows)
    if not len(frame) and columns:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)


def _plot_clustering(records, output, dataset_kind):
    fig = Figure(figsize=(10, 4), layout='constrained')
    if dataset_kind == 'synthetic':
        fig.suptitle('Synthetic smoke test — not real-dataset performance')
    axes = fig.subplots(1, 2)
    for config in dict.fromkeys(row['configuration'] for row in records):
        rows = [row for row in records if row['configuration'] == config]
        axes[0].plot([r['k'] for r in rows], [r['inertia'] for r in rows], marker='o', label=config)
        axes[1].plot([r['k'] for r in rows], [r['silhouette'] for r in rows], marker='o', label=config)
    axes[0].set(xlabel='k', ylabel='Inertia', title='Gallery elbow curves')
    axes[1].set(xlabel='k', ylabel='Silhouette', title='Gallery silhouette (undefined values omitted)')
    axes[1].legend(fontsize=7)
    fig.savefig(output / 'clustering.png', dpi=140)


def _plot_retrieval(records, output, dataset_kind):
    rows = [r for r in records if r['split'] == 'test']
    if not rows:
        return
    fig = Figure(figsize=(max(8, len(rows) * .7), 5), layout='constrained')
    if dataset_kind == 'synthetic':
        fig.suptitle('Synthetic smoke test — not real-dataset performance')
    ax = fig.subplots()
    x = np.arange(len(rows))
    for offset, key in ((-.18, 'precision@5'), (.18, 'map@10')):
        values = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        ax.bar(x + offset, values, width=.35, label=key)
    ax.set(ylim=(0, 1), ylabel='Macro average over evaluable queries', title='Held-out test retrieval')
    ax.set_xticks(x, [f"{r['configuration']}\n{r['strategy']}" for r in rows], rotation=45, ha='right')
    ax.legend()
    fig.savefig(output / 'retrieval.png', dpi=140)


def run_experiments(dataset, output, configurations=None, strategies=('feature', 'rank'),
                    seed=42, weights=None, clusters='auto', k_values=None,
                    tasks=('clustering', 'retrieval'), dataset_kind='real'):
    """Extract once; fit only gallery; evaluate fixed settings on val and test.

    Weights are fixed a priori. Validation metrics are reported separately but do
    not select settings. Extraction timings are measured once per descriptor;
    configuration sums are labelled sums, not independent benchmark timings.
    """
    root, output = Path(dataset).expanduser().resolve(), Path(output).expanduser().resolve()
    if root == output:
        raise ValueError('Output directory must differ from the dataset root.')
    if output.exists() and (not output.is_dir() or any(
            path.name not in {'README.md', '.gitkeep'} for path in output.iterdir())):
        raise ValueError('Output directory is not empty; choose a new or empty directory to avoid mixing experiment reports.')
    if dataset_kind not in ('real', 'synthetic'):
        raise ValueError('dataset_kind must be real or synthetic.')
    configurations = tuple(CONFIGURATIONS if configurations is None else configurations)
    if not configurations:
        raise ValueError('At least one descriptor configuration is required.')
    weights = dict({name: 1.0 for name in DESCRIPTORS} if weights is None else weights)
    # Reject invalid configurations and weights before expensive image decoding.
    for config in configurations:
        dummy = {name: np.ones((1, dim)) for name, dim in zip(DESCRIPTORS, (512, 25, 45))}
        combine_descriptor_blocks(dummy, {name: weights[name] for name in config}, config)
    if set(strategies) - {'feature', 'rank'} or not strategies:
        raise ValueError('Retrieval strategies must be feature and/or rank.')
    if set(tasks) - {'clustering', 'retrieval'} or not tasks:
        raise ValueError('Tasks must be clustering and/or retrieval.')
    started = perf_counter()
    paths = discover_images(root, output)
    discovered_count = len(paths)
    hash_started = perf_counter()
    paths, duplicates, hashes = remove_exact_duplicates(paths)
    hashing_seconds = perf_counter() - hash_started
    blocks, paths, skipped, descriptor_times = extract_dataset_blocks(paths)
    if len(paths) < 3:
        raise ValueError('At least 3 readable unique images are required after corrupt/duplicate filtering.')
    labels = infer_labels(paths, root)
    split = make_split(paths, labels, seed)
    gallery_indices = split['gallery']
    gallery_paths = [paths[i] for i in gallery_indices]
    if k_values is None:
        candidates = list(range(2, min(10, len(gallery_paths)) + 1)) or [1]
    else:
        candidates = list(k_values)
    manual_k = None if clusters == 'auto' else int(clusters)
    if manual_k is not None and manual_k not in candidates:
        candidates.append(manual_k)
    output.mkdir(parents=True, exist_ok=True)
    split_for_index = {int(i): name for name, indices in split.items() for i in indices}
    manifest = [{'dataset_kind': dataset_kind, 'image_id': canonical_identity(path), 'path': path,
                 'relative_path': str(Path(path).relative_to(root)),
                 'sha256': hashes.get(path), 'label': labels[i] if labels is not None else None,
                 'split': split_for_index[i]} for i, path in enumerate(paths)]
    _write_csv(manifest, output / 'split_manifest.csv')
    _write_csv([{'dataset_kind': dataset_kind, 'descriptor': name, 'seconds': value, 'readable_images': len(paths)}
                for name, value in descriptor_times.items()], output / 'descriptor_times.csv')
    clustering_rows, assignment_rows, retrieval_rows, per_query_rows, ranking_rows, selections = [], [], [], [], [], {}
    for config in configurations:
        config = tuple(config)
        name = '+'.join(config)
        config_weights = {descriptor: weights[descriptor] for descriptor in config}
        gallery_blocks = {d: blocks[d][gallery_indices] for d in config}
        extraction_sum = sum(descriptor_times[d] for d in config)
        if 'clustering' in tasks:
            X = combine_descriptor_blocks(gallery_blocks, config_weights, config)
            fitting_started = perf_counter()
            evaluated = evaluate_clustering(X, candidates,
                                            true_labels=[labels[i] for i in gallery_indices] if labels is not None else None,
                                            random_state=seed, manual_k=manual_k)
            fitting_seconds = perf_counter() - fitting_started
            selections[name] = {'selected_k': evaluated.selected_k, 'reason': evaluated.selection_reason}
            assignment_rows.extend({'dataset_kind': dataset_kind, 'configuration': name,
                                    'image_id': path, 'cluster': int(cluster),
                                    'class_label': labels[index] if labels is not None else None}
                                   for path, index, cluster in zip(gallery_paths, gallery_indices, evaluated.labels))
            for record in evaluated.records:
                clustering_rows.append({'dataset_kind': dataset_kind, 'configuration': name, **record,
                                        'selected': record['k'] == evaluated.selected_k,
                                        'descriptor_seconds_sum': extraction_sum,
                                        'gallery_k_search_seconds': fitting_seconds})
        if 'retrieval' not in tasks:
            continue
        for strategy in strategies:
            fitting_started = perf_counter()
            retriever = ImageRetriever(gallery_blocks, gallery_paths, selected=config,
                                       weights=config_weights, strategy=strategy, rrf_constant=60)
            fit_seconds = perf_counter() - fitting_started
            for split_name in ('validation', 'test'):
                query_indices = split[split_name]
                query_paths = [paths[i] for i in query_indices]
                query_blocks = {d: blocks[d][query_indices] for d in config}
                querying_started = perf_counter()
                neighbors = retriever.query(query_blocks, query_paths, top_k=10)
                query_seconds = perf_counter() - querying_started
                rankings = {q: [n.image_id for n in ns] for q, ns in neighbors.items()}
                relevance = {paths[i]: {paths[j] for j in gallery_indices
                                       if labels is not None and labels[i] == labels[j]
                                       and canonical_identity(paths[i]) != canonical_identity(paths[j])}
                             for i in query_indices}
                metrics = evaluate_rankings(rankings, relevance)
                retrieval_rows.append({'dataset_kind': dataset_kind, 'configuration': name, 'strategy': strategy,
                                       'split': split_name, **metrics.summary,
                                       'labels_available': labels is not None,
                                       'descriptor_seconds_sum': extraction_sum,
                                       'gallery_fit_seconds': fit_seconds, 'query_seconds': query_seconds})
                per_query_rows.extend({'dataset_kind': dataset_kind, 'configuration': name, 'strategy': strategy, 'split': split_name, **row}
                                      for row in metrics.per_query)
                for query, ns in neighbors.items():
                    ranking_rows.extend({'dataset_kind': dataset_kind, 'configuration': name, 'strategy': strategy, 'split': split_name,
                                         'query_id': query, 'rank': rank, 'image_id': neighbor.image_id,
                                         'euclidean_distance': neighbor.distance, 'rrf_score': neighbor.score,
                                         'relevant': neighbor.image_id in relevance[query] if labels is not None else None}
                                        for rank, neighbor in enumerate(ns, 1))
    if clustering_rows:
        _write_csv(clustering_rows, output / 'clustering_metrics.csv')
        _write_csv(assignment_rows, output / 'gallery_clusters.csv')
        _plot_clustering(clustering_rows, output, dataset_kind)
    if retrieval_rows:
        _write_csv(retrieval_rows, output / 'retrieval_metrics.csv')
        _write_csv(per_query_rows, output / 'retrieval_per_query.csv')
        _write_csv(ranking_rows, output / 'rankings.csv')
        _plot_retrieval(retrieval_rows, output, dataset_kind)
    metadata = {'dataset_root': str(root), 'dataset_kind': dataset_kind,
                'results_scope': 'synthetic smoke test only' if dataset_kind == 'synthetic' else 'provided dataset only',
                'seed': seed, 'split_proportions': [.6, .2, .2], 'stratified': labels is not None,
                'tasks': tasks,
                'split_counts': {name: len(indices) for name, indices in split.items()},
                'discovered_images': discovered_count, 'readable_unique_images': len(paths),
                'skipped_corrupt': skipped, 'removed_byte_identical_duplicates': duplicates,
                'duplicate_detection_limit': 'Byte-identical files only; re-encoded or perceptually similar copies are not detected.',
                'configurations': configurations, 'weights_fixed_before_evaluation': weights,
                'strategies': strategies, 'rrf_constant': 60, 'k_candidates': candidates,
                'cluster_selections': selections, 'selection_data': 'gallery only',
                'validation_use': 'reported separately; no parameter tuning',
                'descriptor_seconds_once': descriptor_times, 'hashing_seconds': hashing_seconds,
                'elapsed_seconds': perf_counter() - started,
                'timing_note': 'Measured wall times vary across runs; per-configuration descriptor time is a sum of shared single-pass measurements.',
                'precision_denominator': 'fixed cutoff k, including when fewer than k gallery images exist',
                'map10_denominator': 'min(number of relevant gallery images, 10)',
                'queries_without_relevance': 'excluded from relevance averages and reported with coverage',
                'python': sys.version, 'platform': platform.platform()}
    (output / 'metadata.json').write_text(json.dumps(_plain(metadata), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return metadata
