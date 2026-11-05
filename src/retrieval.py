"""Deterministic Euclidean retrieval and weighted reciprocal-rank fusion."""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .descriptors import combine_descriptor_blocks


def canonical_identity(path):
    """Canonical file identity used to exclude the query itself from results."""
    return os.path.normcase(str(Path(path).expanduser().resolve()))


@dataclass(frozen=True)
class Neighbor:
    image_id: str
    distance: float | None = None
    score: float | None = None


class ImageRetriever:
    """Fit once on gallery blocks; rank queries without fitting on queries.

    Feature fusion concatenates weighted, per-descriptor L2-normalized blocks.
    Rank fusion sums weight/(60 + rank), with rank starting at one. Rank fusion
    scores are similarity scores; they are not Euclidean distances.
    """

    def __init__(self, gallery_blocks, gallery_ids, selected=None, weights=None,
                 strategy='feature', rrf_constant=60):
        self.gallery_ids = [str(p) for p in gallery_ids]
        self.identities = [canonical_identity(p) for p in self.gallery_ids]
        if not self.gallery_ids or len(set(self.identities)) != len(self.identities):
            raise ValueError('Gallery must be nonempty and contain unique file identities.')
        if strategy not in ('feature', 'rank'):
            raise ValueError('strategy must be feature or rank.')
        if not np.isfinite(rrf_constant) or rrf_constant < 0:
            raise ValueError('rrf_constant must be finite and nonnegative.')
        self.strategy, self.rrf_constant = strategy, float(rrf_constant)
        self.selected = tuple((tuple(weights) if weights is not None else ('color', 'shape', 'texture'))
                              if selected is None else selected)
        self.weights = dict({name: 1.0 for name in self.selected} if weights is None else weights)
        # Central validation also checks descriptor names, weights and values.
        fused = combine_descriptor_blocks(gallery_blocks, self.weights, self.selected)
        if len(fused) != len(self.gallery_ids):
            raise ValueError('Gallery row count differs from gallery identities.')
        self._gallery_blocks = {name: combine_descriptor_blocks(
            gallery_blocks, {name: 1.0}, (name,)) for name in self.selected}
        if strategy == 'feature':
            self.models = {'fused': NearestNeighbors(metric='euclidean').fit(fused)}
        else:
            self.models = {name: NearestNeighbors(metric='euclidean').fit(self._gallery_blocks[name])
                           for name in self.selected if self.weights[name] > 0}

    def query(self, query_blocks, query_ids, top_k=10):
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError('top_k must be a positive integer.')
        query_ids = [str(p) for p in query_ids]
        query_identities = [canonical_identity(p) for p in query_ids]
        if len(set(query_identities)) != len(query_ids):
            raise ValueError('Query file identities must be unique.')
        fused = combine_descriptor_blocks(query_blocks, self.weights, self.selected)
        if len(fused) != len(query_ids):
            raise ValueError('Query row count differs from query identities.')
        if not query_ids:
            return {}
        # Ask for every gallery distance: a top-k call can truncate a tie before
        # we apply its deterministic canonical-path tie breaker.
        model_inputs = {'fused': fused} if self.strategy == 'feature' else {
            name: combine_descriptor_blocks(query_blocks, {name: 1.0}, (name,)) for name in self.models}
        distances = {}
        for name, model in self.models.items():
            values, indices = model.kneighbors(model_inputs[name], n_neighbors=len(self.gallery_ids))
            ordered = np.empty_like(values)
            np.put_along_axis(ordered, indices, values, axis=1)
            distances[name] = ordered
        result = {}
        for row, (query_id, identity) in enumerate(zip(query_ids, query_identities)):
            eligible = [i for i, candidate in enumerate(self.identities) if candidate != identity]
            if self.strategy == 'feature':
                order = sorted(eligible, key=lambda i: (distances['fused'][row, i], self.identities[i]))
                result[query_id] = [Neighbor(self.gallery_ids[i], distance=float(distances['fused'][row, i]))
                                    for i in order[:top_k]]
            else:
                scores = np.zeros(len(self.gallery_ids), dtype=float)
                for name in self.models:
                    order = sorted(eligible, key=lambda i: (distances[name][row, i], self.identities[i]))
                    for rank, i in enumerate(order, 1):
                        scores[i] += self.weights[name] / (self.rrf_constant + rank)
                order = sorted(eligible, key=lambda i: (-scores[i], self.identities[i]))
                result[query_id] = [Neighbor(self.gallery_ids[i], score=float(scores[i])) for i in order[:top_k]]
        return result
