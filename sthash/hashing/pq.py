"""Product-quantization stable hash codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PQIndex:
    M: int
    K: int
    centroids: np.ndarray  # (M, K, D/M)
    code_to_ids: Dict[str, List[int]]

    def encode(self, emb: np.ndarray) -> np.ndarray:
        """emb: (N, D) → codes (N, M) int."""
        n, d = emb.shape
        assert d % self.M == 0
        sub = d // self.M
        codes = np.zeros((n, self.M), dtype=np.int32)
        for m in range(self.M):
            chunk = emb[:, m * sub : (m + 1) * sub]
            # (N, K)
            dists = ((chunk[:, None, :] - self.centroids[m][None, :, :]) ** 2).sum(axis=2)
            codes[:, m] = dists.argmin(axis=1)
        return codes

    @staticmethod
    def code_key(code: np.ndarray) -> str:
        return "-".join(map(str, code.tolist()))

    def lookup(self, code: np.ndarray) -> List[int]:
        return list(self.code_to_ids.get(self.code_key(code), []))


def fit_pq(embs: np.ndarray, M: int = 8, K: int = 16, seed: int = 42, max_iter: int = 20) -> np.ndarray:
    """Lloyd k-means per subspace. Returns centroids (M, K, D/M)."""
    rng = np.random.RandomState(seed)
    n, d = embs.shape
    assert d % M == 0
    sub = d // M
    centroids = np.zeros((M, K, sub), dtype=np.float64)
    for m in range(M):
        chunk = embs[:, m * sub : (m + 1) * sub].astype(np.float64)
        # init
        idx = rng.choice(n, size=min(K, n), replace=False)
        cents = chunk[idx].copy()
        if cents.shape[0] < K:
            pad = rng.randn(K - cents.shape[0], sub) * 0.01
            cents = np.vstack([cents, pad])
        for _ in range(max_iter):
            dists = ((chunk[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
            assign = dists.argmin(axis=1)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    cents[k] = chunk[mask].mean(axis=0)
                else:
                    cents[k] = chunk[rng.randint(0, n)]
        centroids[m] = cents
    return centroids


def build_index(
    gallery_embs: np.ndarray,
    gallery_ids: np.ndarray,
    M: int = 8,
    K: int = 16,
    seed: int = 42,
) -> Tuple[PQIndex, np.ndarray]:
    cents = fit_pq(gallery_embs, M=M, K=K, seed=seed)
    tmp = PQIndex(M=M, K=K, centroids=cents, code_to_ids={})
    codes = tmp.encode(gallery_embs)
    lut: Dict[str, List[int]] = {}
    for i, code in enumerate(codes):
        key = PQIndex.code_key(code)
        lut.setdefault(key, []).append(int(gallery_ids[i]))
    return PQIndex(M=M, K=K, centroids=cents, code_to_ids=lut), codes
