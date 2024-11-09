from typing import Dict, List, Tuple
import numpy as np

def average_loss_per_variable(results: Dict[Tuple[str, ...], float]) -> Dict[str, float]:
    variable_losses = {var: [] for subset in results.keys() for var in subset}

    for subset, loss in results.items():
        for var in subset:
            variable_losses[var].append(loss)

    avg_losses = {var: np.mean(losses) for var, losses in variable_losses.items()}
    return avg_losses

def normalized_contribution_scores(avg_losses: Dict[str, float], min_loss: float, max_loss: float) -> Dict[str, float]:
    scores = {var: 1 - (loss - min_loss) / (max_loss - min_loss) for var, loss in avg_losses.items()}
    return scores

def softmax_contribution_scores(avg_losses: Dict[str, float]) -> Dict[str, float]:
    exp_scores = {var: np.exp(-loss) for var, loss in avg_losses.items()}
    total = sum(exp_scores.values())
    softmax_scores = {var: score / total for var, score in exp_scores.items()}
    return softmax_scores

def compute_min_max_losses(results: Dict[Tuple[str, ...], float]):
    return min(results.values()), max(results.values())

def compute_variable_contributions(results: Dict[Tuple[str, ...], float]) -> Dict[str, Dict[str, float]]:
    avg_losses = average_loss_per_variable(results)
    min_loss, max_loss = compute_min_max_losses(results)
    normalized_scores = normalized_contribution_scores(avg_losses, min_loss, max_loss)
    softmax_scores = softmax_contribution_scores(avg_losses)
    return {
        "average_losses": avg_losses,
        "normalized_scores": normalized_scores,
        # "softmax_scores": softmax_scores
    }
