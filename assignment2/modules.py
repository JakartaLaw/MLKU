# All modules used for assignment

# %%
# libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
import scipy.linalg as lalg

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
# Exercise3

def alt_p_binomial(k, mu, n):

    binomial_koef = np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))
    return binomial_koef * (mu**(k)) * ((1 - mu)**(n-k))

def alt_cum_binomial(k, mu, n):

    return np.sum([alt_p_binomial(i, mu=mu, n=n) for i in range(k)])

def AllArive(p):
    """probability that people show up"""

    return p**100

def EmpiricalObs(p):

    return np.exp(-2*(10000*(0.95-p))**(2) / 10000)

# Exercise 4#

def LoadIrisData():

    """Loading Data

    Returns
    =======
    df_train, df_test

    """
    DATA_TRAIN = "IrisTrainML.dt"
    DATA_TEST = "IrisTestML.dt"

    df_train_raw = pd.read_table(DATA_TRAIN, sep='\s+',header=None,names=['length', 'width', 'class'])
    df_test_raw = pd.read_table(DATA_TEST, sep='\s+',header=None,names=['length', 'width', 'class'])

    df_train = df_train_raw.loc[df_train_raw['class'] != 2 ]
    df_test = df_test_raw.loc[df_test_raw['class'] != 2 ]

    df_train['class'][ df_train['class'] == 0] = -1
    df_test['class'][ df_test['class'] == 0] = -1

    df_train['intercept'] = 1
    df_test['intercept'] = 1

    return df_train, df_test

def grad_calc(x,y,W):
    yx = y*x
    #print('yx',yx)
    numerator = 1 + np.exp(y * (np.dot(W.T, x) ))
    #print('numerator',numerator)
    return (1 / numerator) * yx

def gradient(x, y, w, n) :

    numer = x * y[:,scp.newaxis]
    denom = scp.dot(x, w) * y
    denom = scp.exp(denom) + 1
    q = numer / denom[:,scp.newaxis]
    mean = scp.sum(q, axis = 0) / n
    return - mean

def Altgradient(X, Y, W):

    """ Calculate gradient

    Parameters
    ==========
    X : features
    y : labels

    Returns
    =======

    list : list of w_1, w_2
    """
    l = []
    avg = 1/len(Y)
    #print('avg',avg)
    for i in range(len(Y)):
        l.append((grad_calc(x=X[i],y=Y[i],W=W)))
    array = np.array(l)

    gradient =  - np.array([avg*np.sum(array.T[i]) for i in range(len(array[1]))])
    return gradient

def LogisticRegression(X, y, eta=0.5, t=1000):

    w0 = np.zeros(X.shape[1])
    w = w0

    w_list = [w]

    n = len(y)
    for i in range(t):

        g = gradient(X, y, w, n)
        v = -g
        w = w + eta*v

        w_list.append(w)

    return np.array(w), np.array(w_list)

def LogisticPrediction(X, W):

    assert X.shape==W.shape, 'shapes of X and W does not match'

    WX = np.dot(X.T,W)
    y_hat = np.exp(WX) / (1 + np.exp(WX))
    return y_hat

# %%

def LogReg(x, y, iterations = 10000):
    g = 1
    n = len(y)
    i = 1
    w = [0 ,0 ,0]

    while (lalg.norm(g) > np.finfo(float).eps) and (i< iterations):
        eta = 1
        g = gradient(x, y, w, n)
        v = -g
        w = w + eta*v
        i = i + 1

    return w

def LogClass(x):
    if x>0:
        return 1
    else:
        return -1
