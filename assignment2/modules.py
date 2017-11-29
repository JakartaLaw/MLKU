# All modules used for assignment

# %%
# libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%

# Exercise 1

def coin():
    return np.random.randint(0,2)

def experiment(nr_experiments):
    return np.array([coin() for i in range(nr_experiments)])

def simulation(nr_simulations):
    return np.array([np.mean(experiment(10)) for i in range(nr_simulations)])

def coin_simulation(nr_simulations=1000000, nr_experiments=20, bias=0.5):
    return np.random.binomial(1, 0.5, size=(nr_simulations, nr_experiments))

def plot_bins():
    return [0.05*i + 0.025 for i in range(21)]

def hoeffdingsbound(alpha, mu=0.5, n=20):
    epsilon = alpha - mu
    return np.exp((-2)*n*(epsilon**2))

def markovsbound(alpha, mu=0.5, n=20):
    if alpha == 0:
        return np.nan
    return mu/alpha

# %%

# Exercise 2
