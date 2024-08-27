
# %% 0 - import libraries
import numpy as np



# %% 1 -
def accumulation_years(i, a):
    # return np.ceil(np.log(a) / np.log(1 + i))
    return np.log(a) / np.log(1 + i)


# %% 2 -
print(accumulation_years(0.1, 2))


# %% 3 -
print(accumulation_years(0.2, 2))


# %% 4 -
print(accumulation_years(0.3, 2))


# %% 5 -
print(accumulation_years(0.4, 2))


# %% 6 -
print(accumulation_years(0.5, 2))


# %% 7 -
print(accumulation_years(0.6, 2))
