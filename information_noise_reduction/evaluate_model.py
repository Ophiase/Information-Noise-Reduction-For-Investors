from typing import Callable, Dict, Generator, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
# from keras.models import Sequential
# import keras

ModelGenerator = Callable[[int], Model]

def evaluate_model(X: np.ndarray, y: np.ndarray, model_generator: Callable[[int], Model], 
                   epochs: int = 10, batch_size: int = 32) -> float:
    model = model_generator(X.shape[1])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    loss = model.evaluate(X, y, verbose=0)
    return loss

def evaluate_subsets(df: pd.DataFrame, target_col: str, 
                     subset_gen: Generator[Tuple[str, ...], None, None], 
                     model_generator: Callable[[int], Model],
                     max_subsets: int = None,
                     epochs: int = 10, batch_size: int = 32,
                     verbose: bool = True) -> Dict[Tuple[str, ...], float]:
    X, y = df.drop(columns=[target_col]), df[target_col].values
    subset_losses = {}
    
    n_evaluated = 0
    for subset in subset_gen:
        if n_evaluated == max_subsets: break

        X_subset = X[list(subset)].values
        mean_loss = evaluate_model(X_subset, y, model_generator, epochs=epochs, batch_size=batch_size)
        subset_losses[subset] = mean_loss

        n_evaluated += 1
        if verbose:
            print(f"Evaluated subset {subset} with loss: {mean_loss:.4f}")
    
    return subset_losses