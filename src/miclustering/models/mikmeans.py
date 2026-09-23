import logging
from typing import List, Dict, Optional, Callable, Any
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClusterMixin

from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.distances import DISTANCE_REGISTRY
from miclustering.distances.bag_centroid_vectorized import (
    compute_bag_centroid_matrix,
    precompute_bag_cache,
    BagCentroidCache,
)

logger = logging.getLogger(__name__)

class MIKMeans(BaseEstimator, ClusterMixin):
    r"""
    Implementación del algoritmo K-Means adaptado para Multi-Instance Learning (MIL).

    Diseño Algorítmico MIL:
    -----------------------
    En el aprendizaje multi-instancia estándar, las muestras de entrada son bolsas (Bags)
    que contienen conjuntos de cardinalidad variable de vectores de instancia. Para aplicar
    K-Means:
      1. Cada centroide se modela y representa como un objeto `Bag` sintético que contiene
         exactamente 1 instancia representativa (el vector medio $\mu_k = \frac{1}{|C_k|} \sum_{b \in C_k} \bar{x}_b$).
      2. La distancia entre cualquier bolsa de entrada (con $M$ instancias) y el centroide
         sintético (con 1 instancia) se calcula utilizando directamente la métrica MIL
         configurada (por ejemplo, Hausdorff o Mahalanobis), manteniendo la coherencia
         del dominio y la invariancia de tipos.
      3. En cada iteración se reasignan las bolsas al centroide más cercano y se actualizan
         los centroides sintéticos hasta convergencia o alcanzar `max_iters`.
    """

    def __init__(
        self,
        k: int,
        metric: str = 'hausdorff',
        max_iters: int = 100,
        tol: float = 0.01,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        device: str = "cpu",
    ):
        """Constructor del modelo MIKMeans.
        
        Args:
            k: Número de clústeres.
            metric: Métrica de distancia a utilizar entre bolsas y centroides.
            max_iters: Número máximo de iteraciones.
            tol: Tolerancia relativa de convergencia (fracción máxima de bolsas que
                 pueden cambiar de clúster entre iteraciones consecutivas, por defecto
                 0.01 = 1%). Si es 0.0, exige convergencia exacta.
            random_state: Semilla para la inicialización aleatoria.
            n_jobs: Número de procesos paralelos para cómputo (-1 para todos los núcleos).
            device: Dispositivo de cómputo ('cpu', 'cuda', 'mps', 'auto').
        """
        if k < 1:
            raise ValueError(f"El parámetro 'k' debe ser >= 1. Recibido: {k}")
        if max_iters < 1:
            raise ValueError(f"El parámetro 'max_iters' debe ser >= 1. Recibido: {max_iters}")
        if tol < 0:
            raise ValueError(f"El parámetro 'tol' debe ser >= 0. Recibido: {tol}")
            
        self._k = k
        self._metric_name = metric.lower()
        self._max_iters = max_iters
        self._tol = tol
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._device = (device or "cpu").lower().strip()

        self._metric_func = self._get_metric_function(self._metric_name)

        # Estado del modelo
        self._labels: Dict[str, int] = {}
        self._fitted = False
        self._train_bags: List[Bag] = []
        self._centroids: List[Bag] = []

        logger.debug(f"MIKMeans inicializado: k={k}, metric={metric}, tol={self._tol}")

    @property
    def k(self) -> int:
        return self._k

    @property
    def tol(self) -> float:
        """Tolerancia de convergencia relativa configurada."""
        return self._tol

    @property
    def n_jobs(self) -> int:
        """Número de procesos paralelos para cómputo."""
        return self._n_jobs

    @property
    def device(self) -> str:
        """Dispositivo de cómputo configurado."""
        return self._device

    @property
    def labels(self) -> Dict[str, int]:
        return self._labels.copy()

    @property
    def labels_(self) -> np.ndarray:
        """Etiquetas de clúster asignadas a cada bolsa de entrenamiento (array numpy)."""
        if not self._fitted:
            raise AttributeError("El modelo no ha sido entrenado. Ejecuta fit() primero.")
        return np.array([self._labels.get(bag.bag_id, -1) for bag in self._train_bags])

    @property
    def is_fitted(self) -> bool:
        return self._fitted
        
    @property
    def centroids(self) -> List[Bag]:
        """Devuelve los centroides actuales (objetos Bag)."""
        return self._centroids

    @property
    def cluster_centers_(self) -> List[Bag]:
        """Alias estándar Scikit-Learn para los centroides calculados."""
        return self._centroids

    def _reset_state(self):
        self._labels = {}
        self._fitted = False
        self._train_bags = []
        self._centroids = []
        self._bag_cache = None

    def _get_metric_function(self, name: str) -> Callable[[Bag, Bag], float]:
        if name not in DISTANCE_REGISTRY:
            valid_keys = list(DISTANCE_REGISTRY.keys())
            raise ValueError(f"Métrica '{name}' no reconocida. Disponibles: {valid_keys}")
        
        return DISTANCE_REGISTRY[name]

    def _array_to_bag(self, centroid: np.ndarray, cluster_id: int) -> Bag:
        """Envuelve un vector centroide en un Bag sintético de una instancia."""
        schema_raw = next((b[0].schema for b in self._train_bags if len(b) > 0), None)
        schema = schema_raw if schema_raw is not None else []
        c_mat = np.ascontiguousarray(centroid.reshape(1, -1), dtype=np.float64)
        instance = Instance(c_mat[0].tolist(), schema)
        bag = Bag(bag_id=f"__centroid_{cluster_id}__", label=-1, instances=[instance])
        bag._matrix_cache = c_mat
        return bag

    def _calculate_centroid(self, cluster_bags: List[Bag], cluster_id: int) -> Bag:
        """
        Calcula el centroide de un conjunto de bolsas mediante la agregación de sus instancias.
        Calcula la media de todas las instancias agrupadas (ponderación a nivel de instancia)
        para crear un vector representativo, y lo empaqueta de vuelta en un objeto Bag
        con una única instancia para mantener la coherencia con las abstracciones de dominio.
        """
        valid_matrices = []
        for bag in cluster_bags:
            if len(bag) > 0:
                mat = bag.as_matrix()
                if mat.ndim == 2 and mat.shape[0] > 0:
                    valid_matrices.append(mat)

        if not valid_matrices:
            raise ValueError(f"No se puede calcular el centroide del clúster {cluster_id}: no contiene instancias válidas.")
            
        # Agregamos todas las instancias de todas las bolsas del clúster
        all_instances = np.vstack(valid_matrices)
        
        # El centroide es la media de todas las instancias
        return self._array_to_bag(np.mean(all_instances, axis=0), cluster_id)

    def fit(self, dataset: MIData) -> "MIKMeans":
        """
        Entrena el modelo KMeans adaptado a MIL.
        """
        if dataset.get_num_bags() == 0:
            raise ValueError("El dataset de entrenamiento está vacío.")
            
        if dataset.get_num_bags() < self._k:
            logger.warning(f"El número de bolsas ({dataset.get_num_bags()}) es menor que k ({self._k}). Se ajustará k.")
            self._k = dataset.get_num_bags()

        self._reset_state()
        self._train_bags = dataset.bags
        num_bags = len(self._train_bags)

        # Precomputación O(N) una sola vez: estructuras vectorizadas según la métrica
        self._bag_cache = precompute_bag_cache(self._train_bags, self._metric_name)

        # Inicialización de centroides (elegimos k bolsas aleatorias inicialmente)
        rng = np.random.RandomState(self._random_state)
        initial_indices = rng.choice(num_bags, self._k, replace=False).tolist()
        
        # Los centroides iniciales serán la media de las instancias de cada bolsa inicial
        self._centroids = []
        for c_idx, idx in enumerate(initial_indices):
            bag = self._train_bags[idx]
            if len(bag) == 0:
                for b in self._train_bags:
                    if len(b) > 0:
                        bag = b
                        break
            centroid_bag = self._calculate_centroid([bag], c_idx)
            self._centroids.append(centroid_bag)
        
        cluster_assignments = np.full(num_bags, -1, dtype=int)
        prev_assignments = None
        prev_prev_assignments = None
        
        logger.info(f"Iniciando MIKMeans (k={self._k}, max_iters={self._max_iters}, tol={self._tol})...")

        for iteration in range(self._max_iters):
            # 1. Asignar cada bolsa al centroide más cercano (matriz N x K vectorizada)
            distance_matrix = compute_bag_centroid_matrix(
                self._train_bags,
                self._centroids,
                self._metric_name,
                self._metric_func,
                bag_cache=self._bag_cache,
            )
            new_assignments = np.argmin(distance_matrix, axis=1)
                
            # Comprobar convergencia
            n_changed = int(np.sum(cluster_assignments != new_assignments))
            rel_change = n_changed / num_bags

            if n_changed == 0:
                logger.debug(f"Convergencia exacta alcanzada en la iteración {iteration}.")
                cluster_assignments = new_assignments
                break
            elif self._tol > 0 and rel_change <= self._tol:
                logger.debug(
                    f"Convergencia por tolerancia ({rel_change:.2%} <= {self._tol:.2%}) "
                    f"alcanzada en la iteración {iteration} ({n_changed}/{num_bags} bolsas cambiaron)."
                )
                cluster_assignments = new_assignments
                break
            elif prev_prev_assignments is not None and np.array_equal(new_assignments, prev_prev_assignments):
                logger.debug(
                    f"Ciclo oscilatorio de frontera detectado en iteración {iteration}. "
                    f"Convergencia estable alcanzada."
                )
                cluster_assignments = new_assignments
                break
                
            prev_prev_assignments = prev_assignments
            prev_assignments = cluster_assignments
            cluster_assignments = new_assignments
            
            # 2. Actualizar centroides
            new_centroids = []
            used_indices = set()
            for c in range(self._k):
                cluster_points = np.where(cluster_assignments == c)[0]
                cluster_bags = [self._train_bags[idx] for idx in cluster_points] if len(cluster_points) > 0 else []
                valid_bags = [b for b in cluster_bags if len(b) > 0]
                
                if len(cluster_points) == 0 or len(valid_bags) == 0:
                    # Clúster vacío o sin instancias válidas: reinicializar con el punto más alejado del centroide global
                    logger.debug(f"Clúster {c} vacío o sin instancias en iteración {iteration}. Reinicializando centroide.")
                    global_centroid = self._calculate_centroid(self._train_bags, -1)
                    distances_to_global = compute_bag_centroid_matrix(
                        self._train_bags,
                        [global_centroid],
                        self._metric_name,
                        self._metric_func,
                        bag_cache=self._bag_cache,
                    )[:, 0]
                    sorted_indices = np.argsort(distances_to_global)[::-1]
                    farthest_idx = None
                    for s_idx in sorted_indices:
                        if int(s_idx) not in used_indices:
                            farthest_idx = int(s_idx)
                            break
                    if farthest_idx is None:
                        farthest_idx = int(sorted_indices[0])
                    used_indices.add(farthest_idx)
                    new_centroids.append(self._calculate_centroid([self._train_bags[farthest_idx]], c))
                    continue
                    
                new_centroids.append(self._calculate_centroid(valid_bags, c))
                
            self._centroids = new_centroids
            
        else:
            logger.warning(f"MIKMeans no convergió después de {self._max_iters} iteraciones (tol={self._tol}).")

        self._labels = {self._train_bags[i].bag_id: int(cluster_assignments[i]) for i in range(num_bags)}
        self._fitted = True
        
        return self

    def predict(self, test_dataset: MIData) -> Dict[str, int]:
        """
        Asigna cada bolsa del test_dataset al centroide más cercano.
        """
        if not self._fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de llamar a predict().")
            
        if test_dataset.get_num_bags() == 0:
            raise ValueError("El dataset de prueba no puede estar vacío.")

        logger.info(f"Prediciendo {test_dataset.get_num_bags()} bolsas de prueba...")

        distance_matrix = compute_bag_centroid_matrix(
            test_dataset.bags,
            self._centroids,
            self._metric_name,
            self._metric_func,
        )
        assignments = np.argmin(distance_matrix, axis=1)
        test_labels = {
            bag.bag_id: int(a) for bag, a in zip(test_dataset.bags, assignments)
        }
            
        return test_labels

    def fit_predict(self, X: MIData, y: Optional[MIData] = None) -> Dict[str, int]:  # type: ignore
        self.fit(X)
        if y is not None:
            return self.predict(y)
        return getattr(self, "labels", {})

    def get_cluster_sizes(self) -> Dict[int, int]:
        if not self._fitted:
            return {}
        return dict(Counter(self._labels.values()))

    def get_statistics(self) -> Dict[str, Any]:
        if not self._fitted:
            return {"status": "not_fitted"}
            
        return {
            "k": self._k,
            "metric": self._metric_name,
            "total_bags": len(self._labels),
            "cluster_sizes": self.get_cluster_sizes()
        }

    def __repr__(self,  N_CHAR_MAX: int = 700) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return f"<MIKMeans(k={self._k}, metric={self._metric_name}, tol={self._tol}, status={state})>"

    def __str__(self) -> str:
        if not self._fitted:
            return f"MIKMeans (Unfitted): k={self._k}, metric={self._metric_name}, tol={self._tol}"
            
        stats = self.get_statistics()
        return (f"MIKMeans Model:\n"
                f"  - Config: k={self._k}, metric={self._metric_name}, tol={self._tol}\n"
                f"  - Status: Fitted on {stats['total_bags']} bags\n"
                f"  - Cluster Sizes: {stats['cluster_sizes']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from miclustering.data.arff_reader import ArffToMIData
        full_data = ArffToMIData.from_arff("datasets/musk1.arff") 
        train_data, test_data = full_data.split_data(percentage_train=70, seed=42)
        
        model = MIKMeans(k=2, metric='hausdorff_avg')
        model.fit(train_data)
        
        print(model)
        preds = model.predict(test_data)
        print("Predicciones primeras 5:", list(preds.items())[:5])
    except Exception as e:
        logger.error(f"Error: {e}")
