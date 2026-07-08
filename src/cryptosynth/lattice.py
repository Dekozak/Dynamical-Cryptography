"""Lattice reduction: the geometry of numbers meets the SVD.

A lattice is the set of integer combinations of basis vectors, and the
security of lattice cryptography rests on the difficulty of finding
short vectors in high dimension.  The tools are pure matrix theory,
continuous with the matrix-ml project:

* Gram-Schmidt orthogonalization is the QR decomposition; the product
  of the norms of the Gram-Schmidt vectors is the lattice determinant,
  and their ratios measure a basis's 'orthogonality defect'.
* The LLL algorithm (Lenstra-Lenstra-Lovasz 1982) produces, in
  polynomial time, a basis whose first vector is within a factor
  2^{(n-1)/2} of the shortest, by repeatedly size-reducing and swapping
  adjacent vectors that violate the Lovasz condition.
* Babai's nearest-plane algorithm solves approximate closest-vector
  problems by rounding in the Gram-Schmidt basis.

Reducing a cleverly built lattice recovers the LWE secret at small
parameters -- the dual of the story in lwe.py, now attacked through
geometry rather than algebra.  Everything here is at toy scale.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gram_schmidt",
    "orthogonality_defect",
    "lll",
    "babai_nearest_plane",
    "shortest_vector_norm",
]


def gram_schmidt(B: np.ndarray):
    """Return (Bstar, mu): Gram-Schmidt vectors and coefficients.

    B has lattice basis vectors as rows.  This is exactly the Q (up to
    normalization) and R of a QR decomposition, in number-theorist's
    notation."""
    B = B.astype(float)
    n = B.shape[0]
    Bstar = np.zeros_like(B)
    mu = np.zeros((n, n))
    for i in range(n):
        Bstar[i] = B[i]
        for j in range(i):
            mu[i, j] = np.dot(B[i], Bstar[j]) / np.dot(Bstar[j], Bstar[j])
            Bstar[i] -= mu[i, j] * Bstar[j]
    return Bstar, mu


def orthogonality_defect(B: np.ndarray) -> float:
    """prod ||b_i|| / |det|.  Equals 1 for an orthogonal basis, grows
    with skew; the quantity LLL is designed to shrink."""
    B = B.astype(float)
    det = abs(np.linalg.det(B))
    prod = np.prod(np.linalg.norm(B, axis=1))
    return float(prod / det) if det > 0 else float("inf")


def lll(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL basis reduction (rows are basis vectors).  Integer lattice.

    Returns a reduced basis whose first row is a short lattice vector.
    """
    B = B.astype(float).copy()
    n = B.shape[0]
    Bstar, mu = gram_schmidt(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                B[k] -= round(mu[k, j]) * B[j]
                Bstar, mu = gram_schmidt(B)
        if np.dot(Bstar[k], Bstar[k]) >= (delta - mu[k, k - 1] ** 2) * np.dot(
            Bstar[k - 1], Bstar[k - 1]
        ):
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            Bstar, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return np.rint(B).astype(int)


def babai_nearest_plane(B: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Approximate closest lattice vector to target t (Babai rounding)."""
    B = B.astype(float)
    Bstar, mu = gram_schmidt(B)
    b = t.astype(float).copy()
    n = B.shape[0]
    coeffs = np.zeros(n)
    for i in range(n - 1, -1, -1):
        c = np.dot(b, Bstar[i]) / np.dot(Bstar[i], Bstar[i])
        c = round(c)
        coeffs[i] = c
        b = b - c * B[i]
    return (coeffs @ B).astype(int)


def shortest_vector_norm(B: np.ndarray) -> float:
    """Norm of the shortest row after reduction: a heuristic SVP value."""
    R = lll(B)
    return float(np.min(np.linalg.norm(R, axis=1)))
