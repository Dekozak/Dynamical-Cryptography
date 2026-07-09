# Structure vs. Randomness: Number Theory, Machine Learning, and Cryptography

**The grand synthesis. Three fields, one dichotomy — made computational.**

This is the third and capstone repository in a trilogy:

1. [`recurrence-arithmetic`](https://github.com/you/recurrence-arithmetic) — dynamical recurrence proving theorems in additive number theory (Poincaré → Szemerédi → Sárközy).
2. [`matrix-ml`](https://github.com/you/matrix-ml) — matrix theory powering big-data machine learning (Eckart–Young → randomized SVD → spectral clustering → PageRank).
3. **`crypto-synthesis`** (this repo) — the claim that both of the above, and modern cryptography, are the same subject viewed from three angles.

## The thesis

Every theorem in the first two projects, and the security of modern cryptography, is an instance of the **structure-versus-randomness dichotomy**: an object is either *structured* (and therefore exploitable — you can find its arithmetic progressions, compress it, or break its cipher) or *pseudorandom* (and therefore indistinguishable from noise). Three mathematical instruments recur across all three fields, and this repository implements the bridges:

| Instrument | Additive number theory (repo 1) | Machine learning (repo 2) | Cryptography (this repo) |
|---|---|---|---|
| **Character / exponential sums** | Weyl sums prove nα, n²α equidistribute | — | Weil bound makes Legendre sequences pseudorandom |
| **Random linear maps** | — | Johnson–Lindenstrauss, sketching | leftover hash lemma, randomness extractors |
| **Spectral gap** | — | Fiedler value, PageRank's \|λ₂\| bound | expander mixing, Cayley-graph hashing |
| **Noisy linear algebra** | — | least squares, SVD | **Learning With Errors** (the whole of lattice crypto) |

The last row is the sharpest: strip the noise from LWE and it is Gaussian elimination, the most basic algorithm of repo 2. The single small error term is what stands between a trivial exercise and a conjectured post-quantum-secure cryptosystem.

## What's in the package

`cryptosynth.lwe` implements Learning With Errors and Regev-style public-key encryption, plus a Gaussian-elimination attack that breaks the *noiseless* system instantly and fails on the noisy one — the bridge from matrix theory to lattice cryptography. `cryptosynth.lattice` provides Gram–Schmidt (= QR), the LLL reduction algorithm, orthogonality defect, and Babai's nearest-plane, and shows a short lattice vector encoding the LWE error at toy scale — the geometric dual of the algebraic attack. `cryptosynth.characters` computes Legendre symbols, quadratic residues, character sums against the Weil bound, Legendre-sequence autocorrelation, and discrete-log-orbit discrepancy — the Weyl-equidistribution machinery of repo 1 recast as a pseudorandomness certificate. `cryptosynth.extractors` implements the leftover hash lemma via random binary matrices (a Johnson–Lindenstrauss sketch over F₂), with statistical-distance verification and the inner-product two-source extractor. `cryptosynth.expander` builds Cayley graphs, computes their spectral gap the way repo 2 computed Fiedler values, verifies the expander mixing lemma, relates mixing time to the power-iteration rate, and hashes messages by walking the graph.

Seventeen unit tests bind the code to the mathematics: noiseless LWE falls to linear algebra while noisy LWE resists it; LLL never increases the orthogonality defect and preserves the lattice determinant; the Weil bound holds for non-degenerate character sums (and the excluded perfect-square case is flagged); the linear extractor drives a high-min-entropy source to within the leftover-hash-lemma bound of uniform; and the expander mixing lemma holds for random vertex subsets.

## Quick start

```bash
git clone https://github.com/Dekozak/Dynamical-Cryptography
cd crypto-synthesis
pip install -e ".[dev]"
pytest                       # ~2 s: LWE, LLL, Weil, LHL, expander mixing

python examples/01_lwe_noise_is_hardness.py
python examples/02_lattice_reduction_attack.py
python examples/03_character_sums_pseudorandom.py
python examples/04_extractors_and_expanders.py
```

A taste, from example 1:

```
noiseless LWE (b = A s mod q):
  secret recovered by Gaussian elimination: True
noisy LWE (b = A s + e mod q):
  same linear-algebra attack recovers s: False
Regev public-key encryption:
  message  : [1, 0, 1, 1, 0, 0, 1, 0]
  decrypted: [1, 0, 1, 1, 0, 0, 1, 0]  correct: True
```

and from example 3, the Weil bound in action: for a prime p = 1009, the character sums Σ ((n²+a)/p) stay near ±1 against a √p ≈ 31.8 ceiling — square-root cancellation — while the Legendre sequence has maximum autocorrelation 0.004 and the discrete-log orbit fills [0,1) with star discrepancy 0.0009.

## A note on scale and honesty

All parameters here are deliberately tiny and pedagogical: LWE dimension n ≈ 15, primes around 10³, Cayley graphs on ~100 vertices. They exist to make the mathematics visible, not to attack anything — they provide no purchase whatsoever on deployed cryptosystems, whose parameters are orders of magnitude larger and whose structure is chosen precisely to resist these textbook methods. Two caveats worth stating plainly: the small circulant Cayley graphs used here have a modest spectral gap (a 6-regular graph on 101 vertices is not a good expander — true expanders like the Ramanujan graphs of Lubotzky–Phillips–Sarnak achieve the optimal λ ≤ 2√(d−1)), and the LLL/lattice demonstrations recover short vectors at scales where any method works. The point is the *bridge*, not the break.

## Reading list

Regev, *On lattices, learning with errors, random linear codes, and cryptography*, JACM 56 (2009). Lenstra, Lenstra & Lovász, *Factoring polynomials with rational coefficients*, Math. Ann. 261 (1982). Impagliazzo, Levin & Luby, *Pseudo-random generation from one-way functions*, STOC 1989 (the leftover hash lemma). Hoory, Linial & Wigderson, *Expander graphs and their applications*, Bull. AMS 43 (2006). Tao, *Structure and Randomness*, AMS (2008). Vadhan, *Pseudorandomness*, Found. Trends TCS (2012).

## License

MIT
