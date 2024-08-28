
# %% 0 - import libraries
import numpy as np


# %% 1 -
def accumulation_rate(n, a):
    return np.exp(np.log(a) / n) - 1


# %% 2 -
print(accumulation_rate(1, 2))


# %% 3 -
print(accumulation_rate(2, 2))


# %% 4 -
print(accumulation_rate(3, 2))


# %% 5 -
print(accumulation_rate(4, 2))


# %% 6 -
print(accumulation_rate(5, 2))


# %% 7 -
print(accumulation_rate(6, 2))
