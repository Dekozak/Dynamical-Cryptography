"""Learning With Errors: the error term is the whole game.

We show that the *noiseless* system b = A s (mod q) is solved instantly
by Gaussian elimination -- the matrix-ml project's linear algebra used
as an attack -- while adding a small error defeats exactly that attack.
Then we run Regev-style public-key encryption of a bit string and
confirm correct decryption.
"""

import numpy as np

from cryptosynth import LWEParams, LWEPublicKey, gen_lwe, solve_noiseless

params = LWEParams(n=15, m=40, q=257, sigma=2.0)

# --- noiseless: linear algebra breaks it instantly --------------------
A, b, s, e = gen_lwe(params, rng=0)
b_clean = (A @ s) % params.q
s_rec = solve_noiseless(A, b_clean, params.q)
print("noiseless LWE (b = A s mod q):")
print(f"  secret recovered by Gaussian elimination: {np.array_equal(s_rec, s)}")

# --- noisy: the same attack fails -------------------------------------
s_rec_noisy = solve_noiseless(A, b, params.q)
print("noisy LWE (b = A s + e mod q):")
print(f"  same linear-algebra attack recovers s: {s_rec_noisy is not None and np.array_equal(s_rec_noisy, s)}")
print(f"  (error vector had entries up to |e| = {np.abs(e).max()})")

# --- encryption on top of the noisy problem ---------------------------
pk = LWEPublicKey.keygen(params, rng=1)
message = [1, 0, 1, 1, 0, 0, 1, 0]
ct = [pk.encrypt(bit, rng=10 + i) for i, bit in enumerate(message)]
dec = [pk.decrypt(c) for c in ct]
print("\nRegev public-key encryption:")
print(f"  message  : {message}")
print(f"  decrypted: {dec}")
print(f"  correct  : {dec == message}")
