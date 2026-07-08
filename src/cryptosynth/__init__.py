"""cryptosynth: the structure-versus-randomness dichotomy as the common
root of additive number theory, big-data machine learning, and
cryptography.

Three instruments recur across all three fields:

    character / exponential sums   (Weyl equidistribution)
        recurrence project  ->  pseudorandomness of Legendre sequences
    random linear maps              (Johnson-Lindenstrauss / sketching)
        matrix-ml project   ->  leftover hash lemma, randomness extraction
    spectral gap                    (Fiedler value / power iteration)
        matrix-ml project   ->  expander mixing, Cayley-graph hashing

and 'noisy linear algebra' (the matrix-ml SVD/least-squares world with a
small error term) is exactly Learning With Errors, the foundation of
post-quantum cryptography.  This package makes each bridge computational.
"""

from .lwe import LWEParams, LWEPublicKey, gen_lwe, solve_noiseless
from .lattice import (
    babai_nearest_plane,
    gram_schmidt,
    lll,
    orthogonality_defect,
    shortest_vector_norm,
)
from .characters import (
    autocorrelation,
    character_sum_bound,
    legendre_sequence,
    legendre_symbol,
    power_map_discrepancy,
    quadratic_residues,
)
from .extractors import (
    inner_product_extractor,
    lhl_bound,
    linear_extractor,
    min_entropy,
    statistical_distance,
)
from .expander import (
    cayley_graph,
    cayley_hash_walk,
    expander_mixing_discrepancy,
    mixing_time,
    spectral_gap,
)

__version__ = "0.1.0"
