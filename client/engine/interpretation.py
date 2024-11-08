
from typing import Dict, List, Tuple
import numpy as np


# def variable_scores(losses: Dict[Tuple[int, ...], float], n_features: int) -> np.ndarray:
#     scores = np.zeros(n_features)
#     for subset in losses:
#         for feature in subset:
#             scores[feature] += 1
#     return scores

# def average_variable_loss(losses: Dict[Tuple[int, ...], float], n_features: int) -> List[float]:
#     variable_loss = np.zeros(n_features)
#     variable_count = np.zeros(n_features)
    
#     for subset, loss in losses.items():
#         for feature in subset:
#             variable_loss[feature] += loss
#             variable_count[feature] += 1

#     return [(variable_loss[i] / variable_count[i]) if variable_count[i] > 0 else float('inf') for i in range(n_features)]