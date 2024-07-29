
import numpy as np

from numpy.typing import NDArray

from typing import Any, Callable, Tuple
from numbers import Number



def cobweb_points(function: Callable[[Number], Number], initial: Number, steps: int) -> Tuple[NDArray[Any], NDArray[Any]]:
    x_points = np.zeros((2 * steps) + 1)
    y_points = np.zeros((2 * steps) + 1)

    x_points[0] = initial

    for i in range(1, 2 * steps, 2):
        x_points[i] = x_points[i - 1]
        y_points[i] = function(x_points[i])

        x_points[i + 1] = y_points[i]
        y_points[i + 1] = y_points[i]

    return x_points, y_points
