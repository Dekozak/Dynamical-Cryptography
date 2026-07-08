"""Random linear maps and spectral gaps in cryptography.

Part 1 (extractors): a random binary matrix -- a Johnson-Lindenstrauss
sketch over F_2 -- purifies a weak random source into near-uniform bits,
exactly as the leftover hash lemma predicts.

Part 2 (expanders): the spectral gap of a Cayley graph, computed the way
the matrix-ml project computed Fiedler values, makes the graph mix like
a random graph (expander mixing lemma) and gives a toy hash function.
"""

import numpy as np

from cryptosynth import (
    cayley_graph,
    cayley_hash_walk,
    expander_mixing_discrepancy,
    lhl_bound,
    linear_extractor,
    min_entropy,
    mixing_time,
    spectral_gap,
    statistical_distance,
)
from cryptosynth.extractors import extract_samples

rng = np.random.default_rng(0)

# --- Part 1: leftover hash lemma --------------------------------------
# Source: n_in bits, each 1 with prob 0.35 -> min-entropy per bit
# -log2(0.65) ~ 0.62, so ~40 bits gives k ~ 25 bits of min-entropy.
# Extracting L = 8 bits leaves a healthy surplus, so the LHL bound is
# small and the extractor should drive us close to uniform.
n_in, out_bits, n_samples = 40, 8, 300_000
bias = 0.35
biased = (rng.random((n_samples, n_in)) < bias).astype(int)
k = n_in * -np.log2(max(bias, 1 - bias))    # min-entropy of the source
M = linear_extractor(n_in, out_bits, rng=1)
out = extract_samples(M, biased)
sd = statistical_distance(out, out_bits)
print("leftover hash lemma (random linear extractor over F_2):")
print(f"  source min-entropy k ~ {k:.1f} bits, extracting L = {out_bits} bits")
print(f"  empirical statistical distance to uniform : {sd:.5f}")
print(f"  LHL bound (1/2) sqrt(2^(L-k))             : {lhl_bound(k, out_bits):.5f}")

# --- Part 2: expander mixing and hashing ------------------------------
p = 101
A = cayley_graph(p, generators=[1, 10, 30])
d, lam, gap = spectral_gap(A)
print(f"\nCayley graph of Z/{p}, generators {{+-1, +-10, +-30}}:")
print(f"  degree d = {d:.0f}, second eigenvalue lambda = {lam:.3f}, gap = {gap:.3f}")

S = list(range(0, 30))
T = list(range(50, 90))
eml = expander_mixing_discrepancy(A, S, T)
print(f"  expander mixing lemma: |e(S,T) - d|S||T|/N| = {eml['lhs']:.2f}"
      f"  <=  lambda sqrt(|S||T|) = {eml['rhs']:.2f}")
print(f"  random-walk mixing time ~ {mixing_time(A):.1f} steps"
      f"  (the 1/gap rate from power iteration)")

msg_a = rng.integers(0, 2, size=32)
msg_b = msg_a.copy(); msg_b[0] ^= 1
print(f"\n  Cayley hash: message -> vertex; flipping one bit moves the hash"
      f" from {cayley_hash_walk(msg_a, p, [1,10,30])}"
      f" to {cayley_hash_walk(msg_b, p, [1,10,30])}")
