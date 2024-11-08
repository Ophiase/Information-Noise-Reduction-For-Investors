from typing import Callable, Dict, Generator, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.model import Model

ModelGenerator = Callable[[int], Model]

def evaluate_model(X: np.ndarray, y: np.ndarray, model_generator: ModelGenerator, 
                   epochs: int = 10, batch_size: int = 32
                   ) -> float:
    model = model_generator(X.shape[1])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    loss = model.evaluate(X, y, verbose=0)

    return loss

def evaluate_subsets(X: np.ndarray, y: np.ndarray, subset_gen: Generator[Tuple[int, ...], None, None], 
                     model_generator: ModelGenerator, 
                     epochs: int = 10, batch_size: int = 32,
                     verbose: bool = True
                    ) -> Dict[Tuple[int, ...], float]:

    subset_losses = {}
    
    for subset in subset_gen:
        X_subset = X[:, subset]
        mean_loss = evaluate_model(X_subset, y, model_generator, epochs=epochs, batch_size=batch_size)
        subset_losses[subset] = mean_loss

        if verbose:
            print(f"Evaluated subset {subset} with loss: {mean_loss:.4f}")
    
    return subset_losses
