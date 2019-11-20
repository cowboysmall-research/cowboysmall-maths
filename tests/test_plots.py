import sys
import warnings

import numpy as np

from maths.plots import Plot


def main(argv):
    title  = 'Cobweb Plot'
    labels = ('$x_n$', '$x_{n + 1}$')

    plot1  = Plot(title, labels, lambda x: x / (1 + x))
    plot1.plot_function()
    plot1.plot_yequalsx()
    plot1.plot_cobweb(0.9, 20)
    plot1.plot_save('./images/plots/plot_cobweb_01.png')

    plot2  = Plot(title, labels, lambda x: x * 2.75 * (1 - x), (-0.5, 1.5), (-0.5, 1))
    plot2.plot_function()
    plot2.plot_yequalsx()
    plot2.plot_cobweb(0.1, 20)
    plot2.plot_save('./images/plots/plot_cobweb_02.png')

    plot3  = Plot(title, labels, lambda x: x * np.exp(0.5 * x), (-1, 3), (-1, 3))
    plot3.plot_function()
    plot3.plot_yequalsx()
    plot3.plot_cobweb(0.2, 20)
    plot3.plot_save('./images/plots/plot_cobweb_03.png')

    plot4  = Plot(title, labels, lambda x: np.sqrt(x + 2), (-2, 3), (-3, 3))
    plot4.plot_function()
    plot4.plot_yequalsx()
    plot4.plot_cobweb(0.5, 20)
    plot4.plot_save('./images/plots/plot_cobweb_04.png')

    plot5  = Plot(title, labels, lambda x: -np.sqrt(x + 2), (-2, 3), (-3, 3))
    plot5.plot_function()
    plot5.plot_yequalsx()
    plot5.plot_cobweb(0.5, 20)
    plot5.plot_save('./images/plots/plot_cobweb_05.png')

    plot6  = Plot(title, labels, lambda x: x * np.log(x**2), (-1, 3), (-1, 3))
    plot6.plot_function()
    plot6.plot_yequalsx()
    plot6.plot_cobweb(-0.1, 20)
    plot6.plot_save('./images/plots/plot_cobweb_06.png')

    plot7  = Plot(title, labels, lambda x: x * np.log(x**2), (-1, 3), (-1, 3))
    plot7.plot_function()
    plot7.plot_yequalsx()
    plot7.plot_cobweb(0.1, 20)
    plot7.plot_save('./images/plots/plot_cobweb_07.png')

    plot8  = Plot(title, labels, lambda x: x * np.log(x**2), (-1, 3), (-1, 3))
    plot8.plot_function()
    plot8.plot_yequalsx()
    plot8.plot_cobweb(1.01, 20)
    plot8.plot_save('./images/plots/plot_cobweb_08.png')

    plot9  = Plot(title, labels, lambda x: x * np.log(x**2), (-1, 3), (-1, 3))
    plot9.plot_function()
    plot9.plot_yequalsx()
    plot9.plot_cobweb(1.9, 20)
    plot9.plot_save('./images/plots/plot_cobweb_09.png')


if __name__ == "__main__":
    main(sys.argv[1:])

