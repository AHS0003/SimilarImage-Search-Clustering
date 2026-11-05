# Classical Image Retrieval & Clustering

This branch migrates the original CBIR Flask demo to a reproducible classical
computer-vision experiment suite. The web interface, VGG19, TensorFlow, generated
web outputs and committed Python bytecode have been removed.

## Implementation plan (draft PR)

- Keep RGB histograms (8 bins per channel), rotation-invariant Zernike moments
  and scientific SFTA boundary descriptors.
- Centralize image preprocessing and per-family L2 normalization before weighting.
- Compare all seven descriptor subsets using K-Means, Euclidean nearest neighbors,
  weighted feature concatenation and weighted reciprocal-rank fusion.
- Select K from gallery-only internal clustering metrics; keep validation and test
  queries disjoint from the gallery and never tune against test results.
- Report silhouette, Davies-Bouldin, Calinski-Harabasz, ARI/NMI when labels exist,
  Precision@5/10, Recall@10 and mAP@10, with coverage and measured timings.
- Add reproducible splits, CSV reports, plots, pytest coverage and GitHub Actions.

The complete real dataset is not in the repository. Synthetic tests will verify
correctness; they will not be presented as real retrieval performance.

Implementation and validation are in progress on the draft pull request. Full
CLI instructions and verified test outcomes will replace this migration outline.
