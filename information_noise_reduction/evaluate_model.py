import random
from typing import Callable, Dict, Generator, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model

from subset_generator import random_proper_subset, subset_with_variable, weighted_random_choice, SubsetGenerator
# from keras.models import Sequential
# import keras

ModelGenerator = Callable[[int], Model]

###################################################################################

def evaluate_model(X: np.ndarray, y: np.ndarray, model_generator: ModelGenerator, 
                   epochs: int = 10, batch_size: int = 32) -> float:
    model = model_generator(X.shape[1])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    loss = model.evaluate(X, y, verbose=0)
    return loss

def evaluate_a_subset(
        X: np.ndarray, y: np.ndarray, subset : Tuple[str, ...], 
        subset_losses : Dict[Tuple[str, ...], float], subset_weights : Dict[Tuple[str, ...], int],
        model_generator: ModelGenerator, epochs: int = 10, batch_size: int = 32, verbose: bool = True
        ):
    
    subset = tuple(subset)
    X_subset = X[list(subset)].values
    mean_loss = evaluate_model(X_subset, y, model_generator, epochs=epochs, batch_size=batch_size)
    
    if subset in subset_weights:
        prev_weight = subset_weights[subset]
        subset_weights[subset] += 1
        subset_losses[subset] = (subset_losses[subset] * prev_weight + mean_loss) / subset_weights[subset]
    else:
        subset_weights[subset] = 1
        subset_losses[subset] = mean_loss

    if verbose:
        print(f"Evaluated subset {subset} with loss: {mean_loss:.4f}")

###################################################################################

def evaluate_subsets(df: pd.DataFrame, target_col: str,
                     model_generator: ModelGenerator,
                     max_subsets: int = 100,
                     target_max_variables=10,
                     epochs: int = 10, batch_size: int = 32,
                     verbose: bool = True) -> Dict[Tuple[str, ...], float]:
    
    X, y = df.drop(columns=[target_col]), df[target_col].values
    subset_losses: Dict[Tuple[str, ...], float] = {}
    subset_weights: Dict[Tuple[str, ...], int] = {}
    
    target_max_variables = min(target_max_variables, len(X.columns))

    #########################################
    # INIT
    #########################################

    if verbose: print("\nInit\n---\n")
    
    # evaluate the full model
    # evaluate_a_subset(X, y, X.columns, subset_losses, subset_weights, model_generator, epochs, batch_size, verbose)
    
    # get enough information to be able to discriminate variables
    # ie. evaluate the model with subsets of size min(N-1, M) (at least once for each variable)
        # where N = number of variables, M = target_max_variables
    
    max_subset_size = min(len(X.columns) - 1, target_max_variables)
    for variable in X.columns :
        subset = subset_with_variable(variable, X.columns, max_subset_size)
        evaluate_a_subset(X, y, subset, subset_losses, subset_weights, model_generator, epochs, batch_size, verbose)

    #########################################
    # PROCESS (max_subsets steps)
    #########################################
        # pick a random subset in subset_losses (low loss should have a better probability to be picked, but weight should penalize)
            # pick a random subset of this subset
                # evaluate_subset

    if verbose: print("\nRec\n---\n")
    
    for _ in range(max_subsets):
        subset = weighted_random_choice(subset_losses, subset_weights)
        sampled_subset = subset if len(subset) == 1 else random_proper_subset(subset)
        
        evaluate_a_subset(X, y, sampled_subset, subset_losses, subset_weights, model_generator, epochs, batch_size, verbose)

    return subset_losses, subset_weights

# @deprecated
def evaluate_subsets_obsolete(df: pd.DataFrame, target_col: str, 
                     subset_gen: SubsetGenerator, 
                     model_generator: ModelGenerator,
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

        if verbose:
            print(f"Evaluated subset {subset} with loss: {mean_loss:.4f}")

        n_evaluated += 1
    
    return subset_losses