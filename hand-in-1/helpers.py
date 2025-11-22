import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import brentq
import os

MAIN = os.path.dirname(os.path.realpath(__file__))
IMAGES = os.path.join(MAIN, 'images')
DATA = os.path.join(MAIN, 'data')

def likelihood(tau, samples):
    '''
        explicit form for the likelihood function of tau (parameter)
        input: single tau value and samples
        return logL(tau)
    '''
    tau = float(tau)
    if tau <= 0:
        return -np.inf 

    n = len(samples)

    logL = n * np.log(1/tau) - (1/tau) * samples.sum()
    L = np.exp(logL)
    
    return logL

def ML_estimator(samples):
    '''
        function to compute estimator for an exponential PDF
        the maximum likelihood when solving d/dx(likelihood) = 0 -> mean
    '''
    N = len(samples) # define number of samples
    tau_hat = 1/N * samples.sum()

    return tau_hat

def find_boundaries(f, samples, tau_hat, grid_factor=0.1, grid_points=10000, max_expand=10):
    '''
        find +delta_tau and -delta_tau for a 68.3% confidence interval
        it uses brentq root finding at lower boundary and grid search for upper boundary
        using only one of a kind result in errors at one side searching
    '''
    samples = np.asarray(samples)
    logL_max = f(tau_hat, samples)

    # find logF(x) value for confidence interval
    target = logL_max - 0.5

    # solve lower boundary as root finding problem
    tau_min = 1e-12
    # Define function for root: logL(tau) - target = 0
    func = lambda tau: f(tau, samples) - target
    ci_lower = brentq(func, tau_min, tau_hat)

    # solve upper boundary as grid search problem
    for _ in range(max_expand):
        tau_max = tau_hat * (1 + grid_factor)
        tau_grid = np.linspace(tau_hat, tau_max, grid_points)
        logL_grid = np.array([f(t, samples) for t in tau_grid])
        if logL_grid.min() <= target:
            ci_upper = np.interp(target, logL_grid[::-1], tau_grid[::-1])
            break
        grid_factor *= 2
    else:
        ci_upper = np.nan  # fallback


    return ci_lower, ci_upper
    
def expectation_value(samples):
    '''
        function to calculate the Expectation vaue E[theta]
    '''
    samples = np.array(samples)
    N = len(samples) # define number of samples
    E = 1/N * samples.sum()

    return E