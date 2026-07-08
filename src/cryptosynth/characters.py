"""Character sums: Weyl equidistribution as a certificate of pseudorandomness.

This module is the bridge from the additive-number-theory project
(recurrence-arithmetic), where Weyl sums measured equidistribution of
n*alpha and n^2*alpha, to cryptography, where exponential and character
sums certify that number-theoretic sequences 'look random'.

Two classical facts drive many constructions:

* Weil's bound: for a nontrivial multiplicative character chi mod p and
  a polynomial f of degree d that is not a perfect power,
      | sum_{x mod p} chi(f(x)) |  <=  (d - 1) * sqrt(p).
  The square-root cancellation says the character values behave like a
  random +-1 sequence -- exactly the pseudorandomness a cipher wants.

* Consequently the Legendre symbol sequence ( (n^2 + a) / p ) and, more
  simply, the quadratic-residue indicator, are balanced and have small
  autocorrelation: the basis of Legendre/Damgard pseudorandom sequences.

We also treat the discrete-logarithm / Diffie-Hellman side: powers of a
primitive root g^k mod p equidistribute, quantified by the same Weyl
criterion used in the recurrence project, now over the multiplicative
group.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "legendre_symbol",
    "quadratic_residues",
    "character_sum_bound",
    "legendre_sequence",
    "autocorrelation",
    "power_map_discrepancy",
]


def legendre_symbol(a: int, p: int) -> int:
    """(a/p) in {-1, 0, 1} via Euler's criterion."""
    a %= p
    if a == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def quadratic_residues(p: int) -> np.ndarray:
    """Indicator over 1..p-1 of quadratic residues mod p."""
    return np.array([1 if legendre_symbol(a, p) == 1 else 0 for a in range(1, p)])


def character_sum_bound(p: int, a: int = 0) -> dict:
    """Compare the actual sum S = sum_n ((n^2 + a)/p) to the Weil bound.

    Weil's bound |S| <= (deg f - 1) sqrt(p) requires f(n) = n^2 + a to
    NOT be a perfect square as a polynomial, i.e. a != 0.  For a = 0 the
    polynomial degenerates (n^2 is a perfect square), ((n^2)/p) = 1 for
    every n != 0, and the sum is the trivial p - 1 -- the exact case the
    hypothesis excludes.  The returned dict flags this via 'degenerate'.
    """
    s = sum(legendre_symbol(n * n + a, p) for n in range(p))
    return {
        "sum": int(s),
        "weil_bound": float(np.sqrt(p)),
        "p": p,
        "degenerate": a % p == 0,
    }


def legendre_sequence(p: int) -> np.ndarray:
    """The +-1 Legendre sequence ((n)/p) for n = 1..p-1 (a PRNG seed source)."""
    return np.array([legendre_symbol(n, p) for n in range(1, p)])


def autocorrelation(seq: np.ndarray, shift: int) -> float:
    """Normalized periodic autocorrelation at a given shift.

    A pseudorandom +-1 sequence has autocorrelation near 0 for all
    nonzero shifts -- the low-correlation property Weil's bound
    guarantees for Legendre sequences.
    """
    s = np.asarray(seq, dtype=float)
    N = s.size
    return float(np.dot(s, np.roll(s, shift)) / N)


def power_map_discrepancy(g: int, p: int, grid: int = 100) -> float:
    """Star discrepancy of {g^k / p : k = 0..p-2}, the discrete-log orbit.

    Reuses the equidistribution viewpoint of the recurrence project: the
    multiplicative orbit fills [0,1) uniformly, quantifying that discrete
    exponentiation 'scrambles'.
    """
    xs = []
    v = 1
    for _ in range(p - 1):
        xs.append(v / p)
        v = (v * g) % p
    x = np.sort(np.array(xs))
    t = np.linspace(0, 1, grid, endpoint=False)
    emp = np.searchsorted(x, t, side="right") / x.size
    return float(np.max(np.abs(emp - t)))
