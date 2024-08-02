
# %% 0 - import libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import perm



# %% 1 -
plt.style.use("ggplot")



# %% 1 - 
def function(n):
    return 1 - np.exp(np.log(perm(365, n)) - n * np.log(365))



# %% 2 - 
x = np.arange(0, 51, 1)
y = function(x)



# %% 3 - 
n = np.argmax(y > 0.5)
p = round(y[n], 2)



# %% 4 - 
plt.figure(figsize = (12, 9))
plt.title("Probability of Shared Birthdays")
plt.xlabel("$n$")
plt.ylabel("$P(n)$")

plt.plot(x, y, 'b', linewidth = 2)
plt.axvline(x = n, color = 'g', linestyle = "--", label = f"n = {n}")
plt.axhline(y = p, color = 'r', linestyle = "--", label = f"P(n) = {p}")

plt.legend(loc = "lower right")
plt.show()
