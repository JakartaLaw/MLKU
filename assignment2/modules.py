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

def p_binomial(alpha, mu=0.5, n=20):

    k = int(alpha * n)
    binomial_koef = np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))

    return binomial_koef * (mu**(k)) * ((1 - mu)**(n-k))

def cum_binomial(alpha, mu=0.5, n=20):

    iterator = 1/n

    alpha_vals = [iterator*i for i in range(int(alpha*n + 1))]
    return np.sum([p_binomial(j) for j in alpha_vals])

# %%
Exercise2

def alt_p_binomial(k, mu, n):

    binomial_koef = np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))
    return binomial_koef * (mu**(k)) * ((1 - mu)**(n-k))

def alt_cum_binomial(k, mu, n):

    return np.sum([alt_p_binomial(i, mu=mu, n=n) for i in range(k)])


# Exercise 2


alt_cum_binomial(20, 0.5, 20)
