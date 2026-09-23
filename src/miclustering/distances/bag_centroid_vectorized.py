"""
Matriz vectorizada (N bolsas x K centroides) para MIKMeans.

Explota una invariante estructural de MIKMeans: CADA CENTROIDE ES SIEMPRE
UNA BOLSA DE 1 SOLA INSTANCIA (el vector medio). Con esa invariante, las
métricas de `miclustering.distances` colapsan a formas cerradas calculables con
operaciones NumPy vectorizadas (BLAS), sin bucle Python sobre bolsas ni centroides:

  - Hausdorff (max/min/avg): con B={c} singleton,
        h_max(A,{c}) = max_a d(a,c)
        h_min(A,{c}) = min_a d(a,c)
        h_avg(A,{c}) = (sum_a d(a,c) + min_a d(a,c)) / (|A|+1)
    -> se calcula con un solo cdist sobre todas las instancias apiladas contra
       la matriz (K,d) de centroides, seguido de reducciones segmentadas
       (np.reduceat) por los límites de cada bolsa.

  - Cauchy-Schwarz: agrega cada bolsa a su media antes de comparar, y el
    centroide ES esa media -> similitud coseno entre (N,d) y (K,d), un
    único producto matricial BLAS (@).

  - Earth Mover's Distance: con un único destino, la única solución factible
    del transporte óptimo es que cada instancia envíe toda su masa al
    centroide -> EMD(A,{c}) = mean_a d(a,c). Mismo patrón que Hausdorff.

  - Mahalanobis: la covarianza del centroide (Sigma_b) es SIEMPRE la
    identidad, porque `len(mat2) < 2` en `mahalanobis_distance`. Por tanto
    `cov_comb = 0.5*Sigma_a + 0.5*I + eps*I` NO depende del centroide, solo
    de la bolsa -> se invierte UNA VEZ por bolsa al principio de fit()
    y se reutiliza en todas las iteraciones y contra todos los centroides.

Todas las funciones de este módulo asumen centroides de 1 instancia. El
dispatcher `compute_bag_centroid_matrix` comprueba esa invariante y, si no
se cumple, cae al bucle genérico.
"""

from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple, Any
import numpy as np
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

from miclustering.data.bag import Bag


_HAUSDORFF_MODES = {
    "hausdorff": "max",
    "hausdorff_max": "max",
    "hausdorff_min": "min",
    "hausdorff_avg": "avg",
}


@dataclass
class BagCentroidCache:
    """Caché de estructuras estáticas de bolsas de entrenamiento."""
    # Para Hausdorff y EMD
    stacked_instances: Optional[np.ndarray] = None
    sizes: Optional[np.ndarray] = None
    non_empty: Optional[np.ndarray] = None
    split_points: Optional[np.ndarray] = None

    # Para Cauchy-Schwarz
    bag_means: Optional[np.ndarray] = None
    norm_bags: Optional[np.ndarray] = None
    valid: Optional[np.ndarray] = None

    # Para Mahalanobis: lista de (mu_a, cov_inv_a)
    mahalanobis_stats: Optional[List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None


def _is_singleton_centroids(centroids: list) -> bool:
    """Comprueba que todos los centroides sean bolsas de exactamente 1 instancia."""
    return all(len(c) == 1 for c in centroids)


# ---------------------------------------------------------------------------
# Precomputación de estructuras de bolsas (se ejecuta 1 sola vez en fit())
# ---------------------------------------------------------------------------

def precompute_mahalanobis_bag_stats(
    bags: List[Bag],
) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Precomputa (mu_a, cov_inv_a) por bolsa de entrenamiento — UNA VEZ por fit().

    Como Sigma_b es siempre la identidad para centroides singleton,
    `cov_comb = 0.5*Sigma_a + 0.5*I + eps*I` no depende del centroide.
    Se invierte aquí una sola vez y se reutiliza en todas las iteraciones
    de K-Means y contra todos los centroides.
    """
    stats = []
    for b in bags:
        mat = b.as_matrix()
        if len(mat) == 0:
            stats.append((None, None))
            continue
        mu_a = np.mean(mat, axis=0)
        d = mu_a.shape[0]
        if len(mat) < 2:
            cov_a = np.eye(d)
        else:
            cov_a = np.atleast_2d(np.cov(mat, rowvar=False))
            if cov_a.shape != (d, d):
                cov_a = np.eye(d)

        cov_comb = 0.5 * cov_a + 0.5 * np.eye(d) + np.eye(d) * 1e-5
        try:
            cov_inv = np.linalg.inv(cov_comb)
        except np.linalg.LinAlgError:
            try:
                cov_inv = np.linalg.pinv(cov_comb)
            except np.linalg.LinAlgError:
                cov_inv = np.eye(d)

        if not np.all(np.isfinite(cov_inv)):
            try:
                cov_inv = np.linalg.pinv(cov_comb)
            except np.linalg.LinAlgError:
                cov_inv = np.eye(d)

        if not np.all(np.isfinite(cov_inv)):
            cov_inv = np.eye(d)

        stats.append((mu_a, cov_inv))
    return stats


def precompute_bag_cache(bags: List[Bag], metric_name: str) -> BagCentroidCache:
    """Precomputa representaciones estáticas de las bolsas según la métrica elegida.

    Permite que el bucle iterativo de K-Means no reasigne memoria ni reextraiga
    matrices de instancias en cada paso.
    """
    name = metric_name.lower()
    cache = BagCentroidCache()
    n = len(bags)
    if n == 0:
        return cache

    if name in _HAUSDORFF_MODES or name == "earth_movers":
        bag_mats = [b.as_matrix() for b in bags]
        sizes = np.array([len(m) for m in bag_mats], dtype=np.int64)
        non_empty = np.where(sizes > 0)[0]
        if len(non_empty) > 0:
            stacked = np.vstack([bag_mats[i] for i in non_empty])
            split_points = np.r_[0, np.cumsum(sizes[non_empty])[:-1]]
        else:
            stacked = np.empty((0, 0), dtype=np.float64)
            split_points = np.empty(0, dtype=np.int64)

        cache.sizes = sizes
        cache.non_empty = non_empty
        cache.stacked_instances = stacked
        cache.split_points = split_points

    elif name == "cauchy_schwarz":
        d = 0
        for b in bags:
            m = b.as_matrix()
            if len(m) > 0:
                d = m.shape[1]
                break
        bag_means = np.zeros((n, d), dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        for i, b in enumerate(bags):
            mat = b.as_matrix()
            if len(mat) > 0:
                bag_means[i] = np.mean(mat, axis=0)
                valid[i] = True
        norm_bags = np.linalg.norm(bag_means, axis=1)

        cache.bag_means = bag_means
        cache.norm_bags = norm_bags
        cache.valid = valid

    elif name == "mahalanobis":
        cache.mahalanobis_stats = precompute_mahalanobis_bag_stats(bags)

    return cache


# ---------------------------------------------------------------------------
# Hausdorff (max / min / avg)
# ---------------------------------------------------------------------------

def hausdorff_matrix(
    bags: List[Bag],
    centroids: List[Bag],
    mode: str = "max",
    cache: Optional[BagCentroidCache] = None,
) -> np.ndarray:
    """Matriz (N x K) Hausdorff bolsas vs centroides-singleton."""
    n, k = len(bags), len(centroids)
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float64)

    centroid_mat = np.vstack([c.as_matrix() for c in centroids])  # (K, d)

    sizes: np.ndarray
    non_empty: np.ndarray
    stacked: np.ndarray
    split_points: np.ndarray

    if (
        cache is not None
        and cache.stacked_instances is not None
        and cache.sizes is not None
        and cache.non_empty is not None
        and cache.split_points is not None
    ):
        sizes = cache.sizes
        non_empty = cache.non_empty
        stacked = cache.stacked_instances
        split_points = cache.split_points
    else:
        bag_mats = [b.as_matrix() for b in bags]
        sizes = np.array([len(m) for m in bag_mats], dtype=np.int64)
        non_empty = np.where(sizes > 0)[0]
        if len(non_empty) > 0:
            stacked = np.vstack([bag_mats[i] for i in non_empty])
            split_points = np.r_[0, np.cumsum(sizes[non_empty])[:-1]]
        else:
            stacked = np.empty((0, centroid_mat.shape[1]), dtype=np.float64)
            split_points = np.empty(0, dtype=np.int64)

    matrix = np.full((n, k), np.inf, dtype=np.float64)
    if len(non_empty) == 0:
        return matrix

    cd = cdist(stacked, centroid_mat, metric="euclidean")  # (total_inst, K)

    if mode == "max":
        rows = np.maximum.reduceat(cd, split_points, axis=0)
    elif mode == "min":
        rows = np.minimum.reduceat(cd, split_points, axis=0)
    elif mode == "avg":
        sums = np.add.reduceat(cd, split_points, axis=0)
        mins = np.minimum.reduceat(cd, split_points, axis=0)
        rows = (sums + mins) / (sizes[non_empty][:, None] + 1)
    else:
        raise ValueError(f"Modo Hausdorff no reconocido: '{mode}'")

    matrix[non_empty] = rows
    return matrix


# ---------------------------------------------------------------------------
# Cauchy-Schwarz
# ---------------------------------------------------------------------------

def cauchy_schwarz_matrix(
    bags: List[Bag],
    centroids: List[Bag],
    cache: Optional[BagCentroidCache] = None,
) -> np.ndarray:
    """Matriz (N x K) Cauchy-Schwarz bolsas vs centroides-singleton."""
    n, k = len(bags), len(centroids)
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float64)

    centroid_mat = np.vstack([c.as_matrix() for c in centroids])  # (K, d)
    d = centroid_mat.shape[1]

    bag_means: np.ndarray
    norm_bags: np.ndarray
    valid: np.ndarray

    if (
        cache is not None
        and cache.bag_means is not None
        and cache.norm_bags is not None
        and cache.valid is not None
    ):
        bag_means = cache.bag_means
        norm_bags = cache.norm_bags
        valid = cache.valid
    else:
        bag_means = np.zeros((n, d), dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        for i, b in enumerate(bags):
            mat = b.as_matrix()
            if len(mat) > 0:
                bag_means[i] = np.mean(mat, axis=0)
                valid[i] = True
        norm_bags = np.linalg.norm(bag_means, axis=1)

    dot = bag_means @ centroid_mat.T                     # (N, K)
    norm_cent = np.linalg.norm(centroid_mat, axis=1)       # (K,)
    denom = np.outer(norm_bags, norm_cent)                  # (N, K)

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_sim = np.where(denom > 0, dot / denom, np.nan)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    matrix = 1.0 - cos_sim
    matrix[~valid, :] = np.inf
    matrix[:, norm_cent == 0] = np.inf
    matrix[np.isnan(matrix)] = np.inf
    return matrix


# ---------------------------------------------------------------------------
# Earth Mover's Distance
# ---------------------------------------------------------------------------

def earth_movers_matrix(
    bags: List[Bag],
    centroids: List[Bag],
    cache: Optional[BagCentroidCache] = None,
) -> np.ndarray:
    """Matriz (N x K) EMD bolsas vs centroides-singleton."""
    n, k = len(bags), len(centroids)
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float64)

    centroid_mat = np.vstack([c.as_matrix() for c in centroids])

    sizes: np.ndarray
    non_empty: np.ndarray
    stacked: np.ndarray
    split_points: np.ndarray

    if (
        cache is not None
        and cache.stacked_instances is not None
        and cache.sizes is not None
        and cache.non_empty is not None
        and cache.split_points is not None
    ):
        sizes = cache.sizes
        non_empty = cache.non_empty
        stacked = cache.stacked_instances
        split_points = cache.split_points
    else:
        bag_mats = [b.as_matrix() for b in bags]
        sizes = np.array([len(m) for m in bag_mats], dtype=np.int64)
        non_empty = np.where(sizes > 0)[0]
        if len(non_empty) > 0:
            stacked = np.vstack([bag_mats[i] for i in non_empty])
            split_points = np.r_[0, np.cumsum(sizes[non_empty])[:-1]]
        else:
            stacked = np.empty((0, centroid_mat.shape[1]), dtype=np.float64)
            split_points = np.empty(0, dtype=np.int64)

    matrix = np.full((n, k), np.inf, dtype=np.float64)
    if len(non_empty) == 0:
        return matrix

    cd = cdist(stacked, centroid_mat, metric="euclidean")
    sums = np.add.reduceat(cd, split_points, axis=0)
    matrix[non_empty] = sums / sizes[non_empty][:, None]
    return matrix


# ---------------------------------------------------------------------------
# Mahalanobis
# ---------------------------------------------------------------------------

def mahalanobis_matrix(
    bag_stats: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    centroids: List[Bag],
) -> np.ndarray:
    """Matriz (N x K) Mahalanobis usando estadísticos de bolsa precomputados."""
    n, k = len(bag_stats), len(centroids)
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float64)

    centroid_mat = np.vstack([c.as_matrix() for c in centroids])  # (K, d)

    matrix = np.full((n, k), np.inf, dtype=np.float64)
    for i, (mu_a, cov_inv_a) in enumerate(bag_stats):
        if mu_a is None or cov_inv_a is None:
            continue
        diffs = mu_a[None, :] - centroid_mat                       # (K, d)
        quad = np.einsum("kd,de,ke->k", diffs, cov_inv_a, diffs)   # forma cuadrática por fila
        finite = np.isfinite(quad)
        row = np.empty(k, dtype=np.float64)
        row[finite] = np.sqrt(np.maximum(quad[finite], 0.0))
        if np.any(~finite):
            row[~finite] = np.linalg.norm(diffs[~finite], axis=1)
        matrix[i] = row

    return matrix


# ---------------------------------------------------------------------------
# Dispatcher unificado
# ---------------------------------------------------------------------------

def compute_bag_centroid_matrix(
    bags: List[Bag],
    centroids: List[Bag],
    metric_name: str,
    metric_func: Callable[[Bag, Bag], float],
    bag_cache: Optional[BagCentroidCache] = None,
) -> np.ndarray:
    """Punto de entrada unificado: matriz (N x K) bolsas vs centroides.

    Usa formas analíticas vectorizadas cuando los centroides son singletons.
    Cae al fallback genérico si la métrica o los centroides no cumplen las precondiciones.
    """
    name = metric_name.lower()

    if _is_singleton_centroids(centroids):
        if name in _HAUSDORFF_MODES:
            return hausdorff_matrix(bags, centroids, mode=_HAUSDORFF_MODES[name], cache=bag_cache)
        if name == "cauchy_schwarz":
            return cauchy_schwarz_matrix(bags, centroids, cache=bag_cache)
        if name == "earth_movers":
            return earth_movers_matrix(bags, centroids, cache=bag_cache)
        if name == "mahalanobis":
            stats = (
                bag_cache.mahalanobis_stats
                if (bag_cache is not None and bag_cache.mahalanobis_stats is not None)
                else precompute_mahalanobis_bag_stats(bags)
            )
            return mahalanobis_matrix(stats, centroids)

    # Fallback genérico: bucle estándar
    n, k = len(bags), len(centroids)
    matrix = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            matrix[i, j] = metric_func(bags[i], centroids[j])
    return matrix
