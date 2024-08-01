
# %% 0 - import libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import perm



# %% 1 -
plt.style.use("ggplot")



# %% 1 - 
def function(n):
    return np.exp(np.log(perm(26, n)) - np.log(perm(52, n)))



# %% 2 - 
x = np.arange(1, 27, 1)
y = function(x)



# %% 4 - 
plt.figure(figsize = (12, 9))
plt.title("Drawing n Red Cards")
plt.xlabel("$n$")
plt.ylabel("$P(n)$")

plt.bar(x, y, color = "cadetblue")
plt.plot(x, y, linewidth = 2, color = "steelblue")

plt.show()
