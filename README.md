# Classical Image Retrieval & Clustering

A reproducible Python experiment suite for **RGB histograms, Zernike moments and SFTA texture features**, K-Means clustering and Euclidean nearest-neighbor image retrieval. This refactors the original CBIR demo into a classical computer-vision project; Flask, the frontend, VGG19 and TensorFlow have been removed.

## Install

Use 64-bit Python 3.12. From a terminal:

```bash
git clone https://github.com/AHS0003/SimilarImage-Search-Clustering.git
cd SimilarImage-Search-Clustering
# While the migration PR is open:
git switch feature/classical-image-retrieval-evaluation
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python -m ruff check .
```

`requirements.txt` contains only runtime libraries; `requirements-dev.txt` additionally installs pytest and Ruff. Direct dependencies are pinned. No pretrained models or weight downloads are needed. CI runs the tests, lint and dependency checks on Python 3.12/Linux.

## Architecture

```text
src/
  preprocessing.py     # RGB decoding, EXIF, transparency, proportional resizing
  descriptors.py       # three families, normalization, cached feature fusion
  clustering.py        # candidate K-Means models, diagnostics and K selection
  retrieval.py         # Euclidean neighbors and weighted rank fusion
  metrics.py           # retrieval measures and evaluable-query coverage
  evaluation.py        # fixed splits, seven configurations, CSVs and plots
experiments/
  compare_descriptors.py
  evaluate_clustering.py
  evaluate_retrieval.py
tests/
results/                # generated reports, ignored by Git
.github/workflows/tests.yml
```

## Dataset

The real dataset of approximately 1,200 images in 12 categories is **not included**. Supply a directory with images inside class folders:

```text
dataset/
  Bike/image001.jpg
  Bike/image002.jpg
  Cat/image001.png
  Cat/image002.png
```

Use at least **three readable, unique images per class**, so that gallery, validation and test each contain every class. Larger classes are strongly preferable for meaningful evaluation. The immediate parent folder is the semantic label. If any image is directly in the dataset root, labels are treated as unavailable: clustering still works, ARI/NMI and relevance-based retrieval metrics remain unavailable, and coverage is reported explicitly. Do not mix labelled and unlabelled layouts.

Unreadable images are skipped and recorded. Byte-identical files are deduplicated with SHA256 **before splitting**, keeping the first sorted path; this does not detect re-encodings or perceptual near-duplicates. Inspect duplicates and label quality before claiming real-dataset performance. Supported formats are JPEG, PNG, BMP, TIFF and GIF (first frame).

## Descriptors and fusion

| Family | Preprocessing | Dimension |
|---|---|---:|
| RGB | Proportional fit to 256×256; padding excluded from statistics; 8 bins per channel | 512 |
| Zernike | Otsu foreground segmentation with border-based polarity; proportional fit into 128×128 support; radius enclosing the foreground; absolute Mahotas moments, degree 8 | 25 |
| SFTA | Grayscale, longest edge 256 with aspect ratio preserved; recursive Otsu thresholds, TTBD binary decomposition and boundary statistics | 45 |

RGB decoding applies EXIF orientation and composites transparency onto white. Uniform images yield finite, constant-size vectors; when no foreground or texture boundary exists, those blocks are zero. Segmentation is a documented heuristic for natural scenes, not a semantic object detector. Zernike magnitudes retain rotation invariance up to segmentation and pixel sampling.

SFTA follows Costa, Humpire-Mamani and Traina, *An Efficient Algorithm for Fractal Analysis of Textures*, [SIBGRAPI 2012](https://doi.org/10.1109/SIBGRAPI.2012.15), and the [author's reference code](https://github.com/alceufc/sfta/blob/master/matlab/sfta.m). Eight recursive Otsu thresholds produce eight upper-tail masks and seven adjacent-interval masks. For each mask the three attributes are the box-counting fractal dimension of its eight-neighbor inner boundary, mean grayscale at boundary pixels, and boundary pixel count. Thus the current reference convention is `6*nt-3`, or 45 values for `nt=8`; this replaces the old contour-area/perimeter approximation. Degenerate threshold branches retain fixed slots and empty boundaries yield zero triplets.

`FeatureExtractor` and the cached-block helpers apply **L2 independently to each family → multiply by its weight → concatenate** in color/shape/texture order. Resizing and feature normalization are separate operations. Zero blocks stay zero; weights must be finite, nonnegative and not all zero for a selected configuration. There is no additional global normalization. With Euclidean distance, a feature weight contributes its **square** to squared distance. Dataset and queries use exactly the same stateless transformation.

All descriptor blocks are decoded/extracted once and reused across experiments. Extraction does not estimate any dataset-wide scaler or model, so computing these stateless blocks before partitioning does not learn from test queries.

## Reproducible evaluation protocol

1. Sort canonical image paths, deduplicate exact files, decode and filter corrupt inputs.
2. Make one seeded **60% gallery / 20% validation queries / 20% test queries** partition, stratified by class when labels exist. Integer rounding preserves at least one item per split/class. Reuse that partition for all seven configurations and both retrieval strategies.
3. Fit K-Means and nearest-neighbor indices **only on gallery vectors**. Labels are used for splitting, relevance and external diagnostics, never as input to K-Means.
4. Choose K by maximum finite **gallery silhouette**, considering only candidate models that actually produce the requested number of groups. Scores within 1e-12 tie in favor of lower K. If no eligible silhouette exists, explicitly fall back to K=1. A manually supplied K bypasses selection. ARI/NMI and held-out queries do not select K.
5. Weights and RRF constant are fixed before the run. Validation and test metrics are reported separately; the driver does not tune weights or select a descriptor on test scores. For future tuning, use validation results and a new locked test evaluation; do not repeatedly choose settings against test results.
6. Queries and gallery are disjoint. Retrieval additionally excludes any candidate with the query's canonical file identity. Relevant items are gallery images with the same semantic label; semantic classes and cluster IDs remain distinct.

K-Means uses `random_state=seed`, `n_init=10`, Lloyd's algorithm and Euclidean geometry. Defaults evaluate K=2 through min(10, gallery size); invalid explicitly supplied values are rejected. The elbow curve is diagnostic, not an undocumented automatic decision rule. Undefined diagnostics are empty/null values with an explicit selection reason, never invented scores.

### Clustering measures

| Measure | Interpretation |
|---|---|
| Silhouette | Higher is better; average cohesion/separation under Euclidean distance |
| Davies–Bouldin | Lower is better; within-cluster scatter relative to separation |
| Calinski–Harabasz | Higher is better; between/within dispersion ratio |
| Adjusted Rand Index (ARI) | Label agreement adjusted for chance; requires semantic ground truth |
| Normalized Mutual Information (NMI) | Normalized agreement with semantic labels; requires ground truth |

Internal measures require 2 to N−1 distinct groups. Collapsed or degenerate fits are reported; undefined values stay unavailable. External scores describe gallery clustering, not held-out classification accuracy. See [scikit-learn clustering evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation).

### Retrieval measures and two strategies

- **Precision@5 / Precision@10:** relevant hits divided by the fixed cutoff (5 or 10), including when the gallery is smaller.
- **Recall@10:** relevant top-10 hits divided by all relevant gallery items.
- **AP@10:** sum of precision at relevant ranks up to 10, divided by `min(number of relevant gallery items, 10)`. **mAP@10** is its macro average across evaluable queries.
- Queries with zero relevant gallery items are excluded from relevance averages and counted in `queries_without_relevant`; report `queries_evaluated`, `queries_total` and `coverage` alongside scores. An empty ranking for an evaluable query scores zero.

**Feature fusion** concatenates the weighted normalized descriptor blocks and retrieves with `NearestNeighbors(metric='euclidean')`. **Rank fusion** retrieves independently per family and adds weighted reciprocal ranks: `score(image) = sum(weight_d / (60 + rank_d(image)))`, ranks starting at 1. Higher RRF scores are better; these are similarity scores, not Euclidean distances. Equal distances/scores use canonical path order as a deterministic tie breaker. All gallery distances are considered before truncation to avoid cutting ties inconsistently. Rank fusion applies to retrieval; K-Means diagnostics use the feature vectors.

## Run experiments

Run from the repository root. Always choose a **new, empty output directory**; a directory containing only `README.md` or `.gitkeep` is accepted. Existing reports are never silently mixed or overwritten. An output subdirectory inside the dataset is excluded from discovery, but a separate directory is preferable.

```bash
# All seven descriptor configurations, both retrieval strategies and clustering
python -m experiments.compare_descriptors --dataset /path/to/dataset --output results/comparison --seed 42 --k-values 2 3 4 5 6 8 10 --clusters auto --strategy both

# Only clustering for a selected combination, with manual K
python -m experiments.evaluate_clustering --dataset /path/to/dataset --output results/clustering --descriptors color texture --clusters 3 --k-values 2 3 4

# Only retrieval; individual descriptors or a chosen combination
python -m experiments.evaluate_retrieval --dataset /path/to/dataset --output results/retrieval --descriptors color shape texture --strategy both
```

Replace the example K values with valid values for your gallery. `--weights COLOR SHAPE TEXTURE` defaults to `1 1 1`. Comparing all seven configurations requires all three weights to be positive, because each singleton must have a nonzero weight. `--dataset-kind synthetic` explicitly labels a synthetic smoke run in CSVs, metadata and charts. The default is `real` for your supplied dataset. `--help` lists all options.

For a fully reproducible software smoke run without the real dataset:

```bash
python -m experiments.generate_synthetic --output datasets/synthetic --seed 42 --per-class 10
python -m experiments.compare_descriptors --dataset datasets/synthetic --output results/synthetic-smoke --seed 42 --k-values 2 3 4 --dataset-kind synthetic
```

These deliberately easy generated classes validate the software, not its real-world quality.

| ID | Configuration |
|---|---|
| E1 | RGB (`color`) |
| E2 | Zernike (`shape`) |
| E3 | SFTA (`texture`) |
| E4 | RGB + Zernike |
| E5 | RGB + SFTA |
| E6 | Zernike + SFTA |
| E7 | RGB + Zernike + SFTA |

Generated artifacts include:

- `split_manifest.csv`: paths, labels, hashes and split assignments.
- `descriptor_times.csv`: measured extraction time per family plus shared decoding time.
- `clustering_metrics.csv`, `gallery_clusters.csv`, `clustering.png`: every candidate K's diagnostics and selected assignments.
- `retrieval_metrics.csv`, `retrieval_per_query.csv`, `rankings.csv`, `retrieval.png`: aggregate/per-query metrics, returned neighbors, distances or RRF scores, and comparisons.
- `metadata.json`: seed, split counts, settings, selection reasons, skipped/duplicate files, timing conventions and runtime information.

Measured search time includes distance computation, deterministic ranking and exclusion. Feature extraction is shared across configurations: reported per-configuration extraction times are **sums of measured family times**, not seven independently timed extractions. Hashing/decoding are reported separately. Wall times and absolute paths differ across machines; identical settings, image content and library versions reproduce the partition and numerical results up to platform floating-point effects, not the timing values.

## Tests and limitations

Pytest covers descriptor dimensions/numerics, varied image modes/resolutions, EXIF/alpha, uniform/corrupt images, scientific SFTA cases, Zernike rotation, normalization/weights, K-Means and every metric, deterministic Euclidean/RRF retrieval, self exclusion, disjoint stratified splits, duplicate removal, gallery-only fitting, repeatability, all seven configurations and CLI artifact generation. Tests create controlled synthetic images and need no real dataset or network access.

No real-dataset performance is claimed here. Synthetic runs validate software and protocol correctness only. Before publishing experimental conclusions, run on the real labelled dataset, inspect near-duplicates and label quality, freeze hyperparameters, evaluate held-out queries and consider multiple split seeds with uncertainty intervals. One fixed-seed run is reproducible but does not measure that uncertainty. The gallery-based K choice is exploratory internal validation, not evidence that it recovers the semantic class count.
