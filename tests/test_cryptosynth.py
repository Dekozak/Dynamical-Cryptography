import numpy as np
import pytest

from cryptosynth import (
    LWEParams,
    LWEPublicKey,
    autocorrelation,
    babai_nearest_plane,
    cayley_graph,
    character_sum_bound,
    expander_mixing_discrepancy,
    gen_lwe,
    gram_schmidt,
    legendre_sequence,
    legendre_symbol,
    lhl_bound,
    linear_extractor,
    lll,
    min_entropy,
    orthogonality_defect,
    quadratic_residues,
    solve_noiseless,
    spectral_gap,
    statistical_distance,
)
from cryptosynth.extractors import extract_samples, inner_product_extractor


# ---------- LWE --------------------------------------------------------------

def test_noiseless_lwe_broken_by_linear_algebra():
    params = LWEParams(n=12, m=30, q=257, sigma=2.0)
    A, b, s, e = gen_lwe(params, rng=0)
    b_clean = (A @ s) % params.q
    s_rec = solve_noiseless(A, b_clean, params.q)
    assert s_rec is not None and np.array_equal(s_rec, s)


def test_noisy_lwe_resists_the_same_attack():
    params = LWEParams(n=12, m=30, q=257, sigma=2.0)
    A, b, s, e = gen_lwe(params, rng=1)
    if np.all(e == 0):
        pytest.skip("degenerate zero-error draw")
    s_rec = solve_noiseless(A, b, params.q)
    assert s_rec is None or not np.array_equal(s_rec, s)


def test_regev_encryption_roundtrip():
    params = LWEParams(n=20, m=60, q=521, sigma=1.5)
    pk = LWEPublicKey.keygen(params, rng=2)
    bits = [0, 1, 1, 0, 1, 0, 0, 1]
    dec = [pk.decrypt(pk.encrypt(b, rng=100 + i)) for i, b in enumerate(bits)]
    # allow rare decryption error but require high fidelity
    assert sum(x == y for x, y in zip(bits, dec)) >= len(bits) - 1


# ---------- lattice ----------------------------------------------------------

def test_gram_schmidt_orthogonal():
    B = np.array([[2.0, 1.0], [1.0, 3.0]])
    Bstar, mu = gram_schmidt(B)
    assert abs(np.dot(Bstar[0], Bstar[1])) < 1e-10


def test_lll_reduces_orthogonality_defect():
    B = np.array([[201.0, 37.0], [98.0, 18.0]])
    R = lll(B)
    assert orthogonality_defect(R) <= orthogonality_defect(B) + 1e-9
    assert orthogonality_defect(R) < 2.0


def test_lll_preserves_lattice_determinant():
    B = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 5.0]])
    R = lll(B)
    assert abs(abs(np.linalg.det(R)) - abs(np.linalg.det(B))) < 1e-6


def test_babai_returns_lattice_point():
    B = np.array([[3.0, 0.0], [0.0, 5.0]])
    v = babai_nearest_plane(B, np.array([3.4, 9.1]))
    assert v[0] % 3 == 0 and v[1] % 5 == 0


# ---------- character sums ---------------------------------------------------

def test_legendre_symbol_values():
    # mod 7: QRs are 1,2,4
    assert [legendre_symbol(a, 7) for a in range(7)] == [0, 1, 1, -1, 1, -1, -1]


def test_quadratic_residues_count():
    p = 101
    assert quadratic_residues(p).sum() == (p - 1) // 2


def test_weil_bound_holds():
    p = 1009
    for a in (1, 5, 13):  # non-degenerate: n^2 + a not a perfect square
        r = character_sum_bound(p, a)
        assert not r["degenerate"]
        assert abs(r["sum"]) <= r["weil_bound"] + 2  # sqrt-cancellation
    # a = 0 is the excluded perfect-square case: trivial sum p-1
    assert character_sum_bound(p, 0)["degenerate"]


def test_legendre_sequence_low_autocorrelation():
    p = 1009
    seq = legendre_sequence(p)
    for k in range(1, 10):
        assert abs(autocorrelation(seq, k)) < 0.1


# ---------- extractors -------------------------------------------------------

def test_lhl_bound_monotone():
    assert lhl_bound(20, 8) < lhl_bound(20, 12)   # more output bits -> looser
    assert lhl_bound(30, 8) < lhl_bound(20, 8)    # more entropy -> tighter


def test_linear_extractor_purifies_source():
    rng = np.random.default_rng(3)
    n_in, L, N = 40, 6, 200_000
    bias = 0.35
    biased = (rng.random((N, n_in)) < bias).astype(int)
    M = linear_extractor(n_in, L, rng=4)
    out = extract_samples(M, biased)
    k = n_in * -np.log2(max(bias, 1 - bias))
    # with k >> L the bound is small and non-vacuous
    assert lhl_bound(k, L) < 0.1
    assert statistical_distance(out, L) <= lhl_bound(k, L) + 0.02


def test_inner_product_extractor_bit():
    x = np.array([1, 0, 1, 1])
    y = np.array([1, 1, 1, 0])
    assert inner_product_extractor(x, y) == (1 + 0 + 1 + 0) % 2


# ---------- expander ---------------------------------------------------------

def test_cayley_graph_regular():
    A = cayley_graph(11, [1, 3])
    assert np.allclose(A.sum(axis=1), A.sum(axis=1)[0])  # regular
    assert np.allclose(A, A.T)                            # undirected


def test_spectral_gap_positive_for_expander():
    A = cayley_graph(101, [1, 10, 30])
    d, lam, gap = spectral_gap(A)
    assert gap > 0 and lam < d


def test_expander_mixing_lemma_holds():
    A = cayley_graph(101, [1, 10, 30])
    rng = np.random.default_rng(5)
    for _ in range(20):
        S = rng.choice(101, size=25, replace=False)
        T = rng.choice(101, size=40, replace=False)
        r = expander_mixing_discrepancy(A, S, T)
        assert r["lhs"] <= r["rhs"] + 1e-6
