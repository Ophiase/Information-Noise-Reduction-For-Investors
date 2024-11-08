from typing import Callable
from tensorflow.keras.models import Model

ModelGenerator = Callable[[int], Model]