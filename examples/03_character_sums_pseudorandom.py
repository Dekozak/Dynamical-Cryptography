"""Character sums: Weyl equidistribution certifies cryptographic randomness.

The Weyl-sum machinery from the recurrence-arithmetic project reappears
here.  There it proved n*alpha and n^2*alpha equidistribute; here the
Weil bound proves Legendre-symbol sequences are balanced and
low-correlation -- the pseudorandomness a stream cipher relies on.
"""

import numpy as np

from cryptosynth import (
    autocorrelation,
    character_sum_bound,
    legendre_sequence,
    power_map_discrepancy,
    quadratic_residues,
)

p = 1009  # a prime

# --- square-root cancellation in a character sum ----------------------
for a in (1, 7, 42):  # f(n)=n^2+a not a perfect square: Weil applies
    r = character_sum_bound(p, a)
    print(f"sum_n ((n^2+{a})/{p}) = {r['sum']:>4}   "
          f"Weil bound ~ sqrt(p) = {r['weil_bound']:.1f}")
deg = character_sum_bound(p, 0)
print(f"sum_n ((n^2+0)/{p})  = {deg['sum']:>4}   "
      f"(degenerate: n^2 is a perfect square, Weil excludes this)")

# --- balance and low autocorrelation of the Legendre sequence ---------
qr = quadratic_residues(p)
print(f"\nquadratic residues mod {p}: {qr.sum()} of {p-1} "
      f"(expected ~{(p-1)//2})")

seq = legendre_sequence(p)
acs = [abs(autocorrelation(seq, k)) for k in range(1, 20)]
print(f"Legendre sequence autocorrelation (shifts 1..19):")
print(f"  max |R(k)| = {max(acs):.4f}  (near 0 => pseudorandom)")

# --- discrete-log orbit equidistributes -------------------------------
disc = power_map_discrepancy(11, p)
print(f"\ndiscrete-exp orbit g^k mod p (g=11): star discrepancy {disc:.4f}")
print("(small discrepancy: exponentiation scrambles uniformly -- the same"
      " equidistribution idea as n*alpha in the recurrence project.)")
