import itertools
import random
from typing import Generator, List, Tuple


def all_subsets_generator(columns: List[str]) -> Generator[Tuple[str, ...], None, None]:
    for subset_size in range(1, len(columns) + 1):
        yield from itertools.combinations(columns, subset_size)

def reverse_all_subsets_generator(columns: List[str]) -> Generator[Tuple[str, ...], None, None]:
    for subset_size in range(len(columns), 0, -1):
        yield from itertools.combinations(columns, subset_size)
        
def random_subset_generator(columns: List[str], num_subsets: int) -> Generator[Tuple[str, ...], None, None]:
    # not working preperly
    for _ in range(num_subsets):
        subset_size = random.randint(1, len(columns))
        subset = tuple(sorted(random.sample(columns, subset_size)))
        yield subset