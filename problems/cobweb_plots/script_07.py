
# %% 0 - import libraries
import numpy as np
import matplotlib.pyplot as plt

from cowboysmall.maths.plots.cobweb import cobweb_points



# %% 1 -
plt.style.use("ggplot")



# %% 1 - 
def function(x):
    return x * np.log(x**2)



# %% 2 - 
ax = (-1, 3)
ay = (-1, 3)



# %% 2 - 
x = np.arange(*ax, 0.01)
y = function(x)



# %% 3 - 
c, d = cobweb_points(function, 0.1, 20)



# %% 4 - 
plt.figure(figsize = (12, 9))
plt.title("Cobweb Plot")
plt.xlabel("$x_n$")
plt.ylabel("$x_{n + 1}$")

plt.axis('equal')
plt.axis([ax[0] - 0.1, ax[1] + 0.1, ay[0] - 0.1, ay[1] + 0.1])

plt.plot(x, y, 'b', linewidth = 2)
plt.plot(x, x, 'g', linewidth = 2)
plt.plot(c, d, 'r', linewidth = 2)

plt.axhline(y = 0, color = 'k')
plt.axvline(x = 0, color = 'k')

plt.show()
