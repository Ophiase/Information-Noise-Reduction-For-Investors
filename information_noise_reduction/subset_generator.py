import itertools
import random
from typing import Dict, Generator, List, Tuple

###################################################################################

SubsetGenerator = Generator[Tuple[str, ...], None, None]

def all_subsets_generator(columns: List[str]) -> SubsetGenerator:
    for subset_size in range(1, len(columns) + 1):
        yield from itertools.combinations(columns, subset_size)

def reverse_all_subsets_generator(columns: List[str]) -> SubsetGenerator:
    for subset_size in range(len(columns), 0, -1):
        yield from itertools.combinations(columns, subset_size)
        
def random_subset_generator(columns: List[str], num_subsets: int) -> SubsetGenerator:
    # not working preperly
    for _ in range(num_subsets):
        subset_size = random.randint(1, len(columns))
        subset = tuple(sorted(random.sample(columns, subset_size)))
        yield subset

###################################################################################

def subset_with_variable(target_variable: str, variable_names: List[str], target_size: int) -> List[str]:
    subset = [var for var in variable_names if var != target_variable]
    selected_subset = random.sample(subset, target_size - 1)
    selected_subset.append(target_variable)
    return selected_subset

def weighted_random_choice(subset_losses: Dict[Tuple[str, ...], float], subset_weights: Dict[Tuple[str, ...], int]) -> Tuple[str, ...]:
    weighted_subsets = [
        (subset, (1 / (loss + 1e-5)) / (weight + 1)) for subset, loss in subset_losses.items() for weight in [subset_weights[subset]]]
    subsets, weights = zip(*weighted_subsets)
    return random.choices(subsets, weights=weights, k=1)[0]

def random_proper_subset(subset: Tuple[str, ...]) -> Tuple[str, ...]:
    if len(subset) <= 1:
        raise ValueError("Subset size should be strictly greater than 1")
    to_remove = random.choice(subset)
    return tuple(elem for elem in subset if elem != to_remove)
