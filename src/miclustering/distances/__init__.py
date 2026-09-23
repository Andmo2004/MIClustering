from typing import Dict, Callable
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
from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.distances.torch_backend import (
    is_torch_available,
    is_cuda_usable,
    get_torch_device,
    compute_distance_matrix_torch,
)
from miclustering.distances.bag_centroid_vectorized import (
    compute_bag_centroid_matrix,
    precompute_bag_cache,
    BagCentroidCache,
)

DISTANCE_REGISTRY: Dict[str, Callable] = {
    'hausdorff': hausdorff_distance,
    'hausdorff_max': hausdorff_distance,
    'hausdorff_min': hausdorff_distance_min,
    'hausdorff_avg': hausdorff_distance_avg,
    'cauchy_schwarz': cauchy_schwarz_distance,
    'earth_movers': earth_movers_distance,
    'mahalanobis': mahalanobis_distance
}

__all__ = [
    "DISTANCE_REGISTRY",
    "compute_distance_matrix",
    "compute_bag_centroid_matrix",
    "precompute_bag_cache",
    "BagCentroidCache",
    "hausdorff_distance",
    "hausdorff_distance_min",
    "hausdorff_distance_avg",
    "cauchy_schwarz_distance",
    "earth_movers_distance",
    "mahalanobis_distance",
    "is_torch_available",
    "is_cuda_usable",
    "get_torch_device",
    "compute_distance_matrix_torch",
]