"""Randomness extraction: the leftover hash lemma as a random linear map.

This module bridges the matrix-ml project's random projections
(Johnson-Lindenstrauss, sketching) with cryptography.  JL says a random
linear map preserves geometry; the leftover hash lemma (LHL) says a
random linear map purifies distributions.  Both are concentration-of-
measure statements about random linear maps, and both live on the same
mathematical street.

Leftover Hash Lemma (Impagliazzo-Levin-Luby; Hastad-Impagliazzo-Levin-
Luby):  let H be a universal family of hash functions from {0,1}^n to
{0,1}^L.  If a source X has min-entropy at least k, then for a random
h in H, the pair (h, h(X)) is epsilon-close to (h, uniform) in
statistical distance with

    epsilon  <=  (1/2) * sqrt( 2^{L - k} ).

So to extract L nearly-uniform bits one needs a min-entropy surplus of
2 log(1/epsilon).  A random binary matrix (a linear universal hash) is
the canonical extractor; this is a Johnson-Lindenstrauss transform over
the field F_2.

We also implement the inner-product two-source extractor, whose
analysis is a slice of additive combinatorics (bias of characters over
F_2), tying back to the character-sum module and thence to the
recurrence project.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "min_entropy",
    "linear_extractor",
    "statistical_distance",
    "lhl_bound",
    "inner_product_extractor",
]


def min_entropy(prob: np.ndarray) -> float:
    """H_inf(X) = -log2 max_x P(x)."""
    p = np.asarray(prob, dtype=float)
    return float(-np.log2(p.max()))


def lhl_bound(entropy: float, out_bits: int) -> float:
    """The leftover hash lemma's statistical-distance bound."""
    return 0.5 * np.sqrt(2.0 ** (out_bits - entropy))


def linear_extractor(n_in: int, out_bits: int, rng=None) -> np.ndarray:
    """A random binary matrix in F_2^{out_bits x n_in}: a linear universal
    hash / strong extractor (the F_2 cousin of a JL sketch)."""
    rng = np.random.default_rng(rng)
    return rng.integers(0, 2, size=(out_bits, n_in))


def _apply_f2(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (M @ x) % 2


def statistical_distance(samples: np.ndarray, out_bits: int) -> float:
    """Statistical distance of an empirical bit-string distribution from
    uniform on {0,1}^out_bits.  samples: array of integers in [0, 2^L)."""
    counts = np.bincount(samples, minlength=2 ** out_bits).astype(float)
    emp = counts / counts.sum()
    unif = 1.0 / 2 ** out_bits
    return float(0.5 * np.abs(emp - unif).sum())


def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def extract_samples(M: np.ndarray, source_samples: np.ndarray) -> np.ndarray:
    """Apply the linear extractor to each source sample (rows = samples)."""
    L = M.shape[0]
    out = np.empty(source_samples.shape[0], dtype=int)
    for i, x in enumerate(source_samples):
        out[i] = _bits_to_int(_apply_f2(M, x))
    return out


def inner_product_extractor(x: np.ndarray, y: np.ndarray) -> int:
    """Two-source extractor: single output bit <x, y> mod 2.

    Its bias is controlled by the additive characters of F_2^n; if x and
    y are independent with enough min-entropy, the output is near
    unbiased -- an additive-combinatorics guarantee.
    """
    return int(np.dot(x, y) % 2)
