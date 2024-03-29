
import numpy             as np
import matplotlib.pyplot as plt

from pylab import rcParams

from typing import Tuple, Callable, TypeVar


class Plot:

    def __init__(
        self,
        title: str,
        labels: Tuple[str, str],
        function: Callable[[int], float],
        x_bounds: Tuple[int, int] = (-1, 1),
        y_bounds: Tuple[int, int] = (-1, 1),
        figsize: Tuple[int, int] = (12, 9),
        linewidth: int = 2
    ) -> None:

        self.title     = title
        self.function  = function
        self.x_bounds  = x_bounds
        self.y_bounds  = y_bounds
        self.x_range   = np.arange(x_bounds[0], x_bounds[1], 0.01)
        self.linewidth = linewidth

        rcParams['figure.figsize'] = figsize

        plt.clf()
        plt.figure()
        plt.title(self.title)

        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        plt.axis('equal')
        plt.axis([x_bounds[0] - 0.1, x_bounds[1] + 0.1, y_bounds[0] - 0.1, y_bounds[1] + 0.1])

        plt.axhline(y = 0, color = 'k')
        plt.axvline(x = 0, color = 'k')



    def plot_function(self) -> None:
        plt.plot(self.x_range, self.function(self.x_range), 'b', linewidth = self.linewidth)



    def plot_yequalsx(self) -> None:
        plt.plot(self.x_range, self.x_range, 'g', linewidth = self.linewidth)



    def plot_cobweb(self, x0: float, steps: int) -> None:
        x_points = np.zeros((2 * steps) + 1)
        y_points = np.zeros((2 * steps) + 1)

        x = x0

        x_points[0] = x
        y_points[0] = 0

        for i in range(1, 2 * steps, 2):
            y = self.function(x)

            x_points[i]     = x
            y_points[i]     = y

            x_points[i + 1] = y
            y_points[i + 1] = y

            x = y

        plt.plot(x_points, y_points, 'r', linewidth = self.linewidth)



    def plot_show(self) -> None:
        plt.show()



    def plot_save(self, file: str) -> None:
        plt.savefig(file, format = 'png')
        plt.close()
