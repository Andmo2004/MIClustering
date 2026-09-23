"""
tests/distances/test_bag_centroid_vectorized.py

Tests unitarios para la computación vectorizada bolsa-centroide (N x K)
utilizada por MIKMeans.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from miclustering.data.attribute import Attribute
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.distances.hausdorff import (
    hausdorff_distance,
    hausdorff_distance_min,
    hausdorff_distance_avg,
)
from miclustering.distances.probability_distribution import (
    cauchy_schwarz_distance,
    earth_movers_distance,
    mahalanobis_distance,
)
from miclustering.distances.bag_centroid_vectorized import (
    compute_bag_centroid_matrix,
    precompute_bag_cache,
    BagCentroidCache,
)


def _make_bag(bag_id: str, matrix: list) -> Bag:
    schema = [Attribute(f"f{i}", "real") for i in range(len(matrix[0]))]
    insts = [Instance(list(row), schema) for row in matrix]
    return Bag(bag_id=bag_id, label="0", instances=insts)


def _make_singleton_centroid(cid: int, vec: list) -> Bag:
    schema = [Attribute(f"f{i}", "real") for i in range(len(vec))]
    c_mat = np.ascontiguousarray(np.array(vec, dtype=np.float64).reshape(1, -1))
    inst = Instance(vec, schema)
    b = Bag(bag_id=f"__centroid_{cid}__", label="-1", instances=[inst])
    b._matrix_cache = c_mat
    return b


class TestBagCentroidVectorizedEquivalence:

    def setup_method(self):
        rng = np.random.RandomState(42)
        d = 4
        self.bags = []
        for i in range(12):
            n_inst = rng.randint(2, 8)
            mat = (rng.randn(n_inst, d) * 2.0).tolist()
            self.bags.append(_make_bag(f"b_{i}", mat))

        self.centroids = []
        for k in range(3):
            vec = (rng.randn(d) * 1.5).tolist()
            self.centroids.append(_make_singleton_centroid(k, vec))

    def test_hausdorff_max_equivalence(self):
        expected = np.array([[hausdorff_distance(b, c) for c in self.centroids] for b in self.bags])
        cache = precompute_bag_cache(self.bags, "hausdorff")
        actual_cached = compute_bag_centroid_matrix(self.bags, self.centroids, "hausdorff", hausdorff_distance, bag_cache=cache)
        actual_uncached = compute_bag_centroid_matrix(self.bags, self.centroids, "hausdorff", hausdorff_distance)

        np.testing.assert_allclose(actual_cached, expected, atol=1e-12)
        np.testing.assert_allclose(actual_uncached, expected, atol=1e-12)

    def test_hausdorff_min_equivalence(self):
        expected = np.array([[hausdorff_distance_min(b, c) for c in self.centroids] for b in self.bags])
        cache = precompute_bag_cache(self.bags, "hausdorff_min")
        actual = compute_bag_centroid_matrix(self.bags, self.centroids, "hausdorff_min", hausdorff_distance_min, bag_cache=cache)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_hausdorff_avg_equivalence(self):
        expected = np.array([[hausdorff_distance_avg(b, c) for c in self.centroids] for b in self.bags])
        cache = precompute_bag_cache(self.bags, "hausdorff_avg")
        actual = compute_bag_centroid_matrix(self.bags, self.centroids, "hausdorff_avg", hausdorff_distance_avg, bag_cache=cache)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_cauchy_schwarz_equivalence(self):
        expected = np.array([[cauchy_schwarz_distance(b, c) for c in self.centroids] for b in self.bags])
        cache = precompute_bag_cache(self.bags, "cauchy_schwarz")
        actual = compute_bag_centroid_matrix(self.bags, self.centroids, "cauchy_schwarz", cauchy_schwarz_distance, bag_cache=cache)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_earth_movers_equivalence(self):
        expected = np.array([[earth_movers_distance(b, c) for c in self.centroids] for b in self.bags])
        cache = precompute_bag_cache(self.bags, "earth_movers")
        actual = compute_bag_centroid_matrix(self.bags, self.centroids, "earth_movers", earth_movers_distance, bag_cache=cache)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_mahalanobis_equivalence(self):
        expected = np.array([[mahalanobis_distance(b, c) for c in self.centroids] for b in self.bags])
        cache = precompute_bag_cache(self.bags, "mahalanobis")
        actual = compute_bag_centroid_matrix(self.bags, self.centroids, "mahalanobis", mahalanobis_distance, bag_cache=cache)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_non_singleton_fallback(self):
        # Centroide con 2 instancias debe caer al fallback genérico
        c_multi = _make_bag("c_multi", [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        actual = compute_bag_centroid_matrix(self.bags, [c_multi], "hausdorff", hausdorff_distance)
        expected = np.array([[hausdorff_distance(b, c_multi)] for b in self.bags])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
