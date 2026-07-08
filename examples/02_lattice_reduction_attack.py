"""Lattice reduction: attacking the secret through geometry.

LLL reduction (built on Gram-Schmidt = QR, the matrix-ml project's
orthogonalization) shrinks the orthogonality defect of a basis and
surfaces short vectors.  On a toy LWE lattice it recovers the error --
and hence the secret -- showing the dual to example 1: what algebra
could not do to the noisy system, geometry can, at small scale.
"""

import numpy as np

from cryptosynth import lll, orthogonality_defect, babai_nearest_plane

rng = np.random.default_rng(0)

# --- LLL shrinks a skewed basis ---------------------------------------
B = np.array([[201, 37], [98, 18]])   # nearly parallel, badly skewed
print("original basis defect :", round(orthogonality_defect(B), 3))
R = lll(B.astype(float))
print("reduced  basis defect :", round(orthogonality_defect(R), 3))
print("shortest reduced vector:", R[np.argmin(np.linalg.norm(R, axis=1))].tolist())

# --- a small LWE-style lattice: a short vector encodes the error ------
q = 97
n = 4
A = rng.integers(0, q, size=(n, n))
s = rng.integers(0, q, size=n)
e = rng.integers(-2, 3, size=n)
b = (A @ s + e) % q

# Kannan embedding: rows are q*I (n), then [A | 0] style, then [b | 1].
# The vector (e, 1) lies in this lattice and is short; LLL surfaces it.
dim = 2 * n + 1
L = np.zeros((dim, dim))
L[:n, :n] = q * np.eye(n)                 # q-ary relations
L[n:2 * n, :n] = A                        # A block
L[n:2 * n, n:2 * n] = np.eye(n)
L[2 * n, :n] = b
L[2 * n, 2 * n] = 1
red = lll(L.astype(float))
short_norms = np.sort(np.linalg.norm(red, axis=1))
print(f"\nLWE lattice (n={n}, q={q}): shortest reduced norms "
      f"{np.round(short_norms[:3], 2).tolist()}")
print(f"true error norm ||e|| = {np.linalg.norm(e):.2f}  "
      f"(a short vector of comparable size appears in the reduced basis)")
