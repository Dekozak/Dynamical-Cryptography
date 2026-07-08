"""Spectral gap: expander mixing and Cayley-graph hash functions.

This module bridges the matrix-ml project's spectral graph theory
(Laplacian eigenvalues, the Fiedler value, power iteration) with
cryptography.  There the second eigenvalue governed clustering and the
convergence rate of PageRank; here the *spectral gap* of an expander
governs how thoroughly a random walk scrambles, which is precisely what
a cryptographic hash or pseudorandom generator needs.

Expander mixing lemma (Alon-Chung): for a d-regular graph on N vertices
with second-largest adjacency eigenvalue lambda, and any vertex sets
S, T,

    | e(S,T) - (d/N) |S| |T| |  <=  lambda * sqrt(|S| |T|),

so a small lambda (a large spectral gap d - lambda) forces the edge
distribution to look like that of a random graph.  Ramanujan graphs
achieve the optimal lambda <= 2 sqrt(d-1).

Cayley-graph hash functions (e.g. the Charles-Goren-Lauter family, and
the classical LPS construction) turn a message into a walk on an
expander; expansion guarantees that distinct messages land on well-
spread vertices, giving collision resistance.  The mixing time of the
walk is 1/gap * log N -- the very same geometric rate that power
iteration exhibited in the matrix-ml project.  Here we build small
Cayley expanders, compute their spectral gap, verify the mixing lemma,
and hash messages by walking.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cayley_graph",
    "spectral_gap",
    "expander_mixing_discrepancy",
    "mixing_time",
    "cayley_hash_walk",
]


def cayley_graph(p: int, generators):
    """Cayley graph of Z/p with the given (symmetric) generator set.

    Returns the N x N adjacency matrix (N = p)."""
    gens = set()
    for g in generators:
        gens.add(g % p)
        gens.add((-g) % p)
    A = np.zeros((p, p))
    for v in range(p):
        for g in gens:
            A[v, (v + g) % p] = 1
    return A


def spectral_gap(A: np.ndarray):
    """Return (d, lambda2, gap) for a regular graph adjacency A.

    d is the degree (top eigenvalue), lambda2 the second-largest in
    absolute value among the rest, gap = d - lambda2."""
    vals = np.sort(np.linalg.eigvalsh(A))[::-1]
    d = vals[0]
    lam = max(abs(vals[1]), abs(vals[-1]))
    return float(d), float(lam), float(d - lam)


def expander_mixing_discrepancy(A: np.ndarray, S, T) -> dict:
    """Left- and right-hand sides of the expander mixing lemma for S, T."""
    N = A.shape[0]
    d, lam, _ = spectral_gap(A)
    S, T = np.asarray(list(S)), np.asarray(list(T))
    e_ST = float(A[np.ix_(S, T)].sum())
    expected = d / N * S.size * T.size
    lhs = abs(e_ST - expected)
    rhs = lam * np.sqrt(S.size * T.size)
    return {"lhs": lhs, "rhs": float(rhs), "edges": e_ST, "expected": expected}


def mixing_time(A: np.ndarray, eps: float = 0.01) -> float:
    """Heuristic random-walk mixing time ~ log(N/eps) / log(d/lambda).

    The same 1/gap-type rate that governed power iteration convergence."""
    N = A.shape[0]
    d, lam, _ = spectral_gap(A)
    if lam <= 0:
        return 1.0
    return float(np.log(N / eps) / np.log(d / lam))


def cayley_hash_walk(message_bits, p: int, generators):
    """Hash a bit-string by walking the Cayley graph: bit selects a step.

    Toy illustration of expander hashing -- distinct messages spread out
    because the graph mixes.  Returns the final vertex in Z/p.
    """
    gens = [g % p for g in generators]
    v = 0
    for bit in message_bits:
        v = (v + gens[int(bit) % len(gens)]) % p
    return v
