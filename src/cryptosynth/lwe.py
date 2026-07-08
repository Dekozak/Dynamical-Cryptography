"""Learning With Errors: cryptography as noisy linear algebra.

This module is the direct bridge from matrix theory (the matrix-ml
project) to post-quantum cryptography.  The Learning With Errors (LWE)
problem of Regev (2005) is:

    given a random matrix A in Z_q^{m x n} and
    b = A s + e  (mod q),   with s the secret and e a *small* error,
    recover s.

Without the error term this is a linear system over Z_q: Gaussian
elimination recovers s in polynomial time.  The single ingredient that
turns a trivial linear-algebra exercise into the bedrock of lattice
cryptography is the small noise e.  Regev proved that solving LWE on
average is as hard as approximating worst-case lattice problems
(GapSVP, SIVP) -- a reduction that makes LWE a favoured foundation for
schemes believed secure against quantum computers (e.g. Kyber).

We implement (a) the search/decision problem, (b) a textbook symmetric
and public-key encryption built on LWE, and (c) an explicit
demonstration that the *noiseless* variant is broken instantly by the
linear algebra of the matrix-ml project, while the noisy variant
resists the same attack.  All parameters here are deliberately tiny and
pedagogical; they provide no purchase on deployed cryptosystems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LWEParams", "gen_lwe", "solve_noiseless", "LWEPublicKey"]


@dataclass
class LWEParams:
    n: int          # secret dimension
    m: int          # number of samples
    q: int          # modulus
    sigma: float    # error standard deviation


def _mod(x, q):
    return np.mod(x, q)


def gen_lwe(params: LWEParams, rng=None):
    """Generate an LWE instance (A, b, s, e) with b = A s + e mod q."""
    rng = np.random.default_rng(rng)
    A = rng.integers(0, params.q, size=(params.m, params.n))
    s = rng.integers(0, params.q, size=params.n)
    e = np.round(rng.normal(0, params.sigma, size=params.m)).astype(int)
    b = _mod(A @ s + e, params.q)
    return A, b, s, e


def solve_noiseless(A: np.ndarray, b: np.ndarray, q: int):
    """Recover s from b = A s (mod q) by Gaussian elimination over Z_q.

    Works instantly when the error is zero; returns None if the recovered
    vector fails to reproduce b (i.e. noise was present and broke it).
    This is the matrix-ml project's linear algebra used as an attack.
    """
    A = _mod(A.astype(np.int64), q)
    b = _mod(b.astype(np.int64), q)
    m, n = A.shape
    M = np.concatenate([A, b[:, None]], axis=1) % q
    row = 0
    pivots = []
    for col in range(n):
        piv = None
        for r in range(row, m):
            if np.gcd(int(M[r, col]), q) == 1:
                piv = r
                break
        if piv is None:
            continue
        M[[row, piv]] = M[[piv, row]]
        inv = pow(int(M[row, col]), -1, q)
        M[row] = (M[row] * inv) % q
        for r in range(m):
            if r != row and M[r, col] != 0:
                M[r] = (M[r] - M[r, col] * M[row]) % q
        pivots.append(col)
        row += 1
        if row == m:
            break
    s = np.zeros(n, dtype=np.int64)
    for i, col in enumerate(pivots):
        s[col] = M[i, n]
    ok = np.array_equal(_mod(A @ s, q), b)
    return (s % q) if ok else None


@dataclass
class LWEPublicKey:
    """Regev-style public-key encryption of single bits.

    Public key (A, b) with b = A s + e.  A ciphertext for bit mu is
    (a subset sum of rows of A, the matching sum of b entries + mu*floor(q/2)).
    Decryption subtracts <a, s> and tests closeness to 0 or q/2.
    """
    A: np.ndarray
    b: np.ndarray
    s: np.ndarray
    q: int

    @classmethod
    def keygen(cls, params: LWEParams, rng=None):
        A, b, s, _ = gen_lwe(params, rng)
        return cls(A, b, s, params.q)

    def encrypt(self, bit: int, rng=None):
        rng = np.random.default_rng(rng)
        m = self.A.shape[0]
        subset = rng.integers(0, 2, size=m)
        a = _mod(subset @ self.A, self.q)
        c = _mod(int(subset @ self.b) + bit * (self.q // 2), self.q)
        return a, c

    def decrypt(self, ct):
        a, c = ct
        v = _mod(int(c) - int(a @ self.s), self.q)
        # closer to q/2 -> bit 1, closer to 0 -> bit 0
        return int(min(v, self.q - v) > self.q // 4)
