import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import least_squares

IMAGES = os.path.join(os.path.dirname(__file__), 'images')

def polar_method(N, mu, sigma, rng):
    
    i=0
    while i < N:
        u = rng.uniform(-1, 1) # draw one sample
        v = rng.uniform(-1, 1) # draw one sample

        r = u**2 + v**2

        if r > 1 or r == 0:
            continue

        m = np.sqrt(-2*np.log(r)/r)
        x1 = mu + sigma * u * m
        x2 = mu + sigma * v * m

        i += 1
        yield u, v, x1, x2

def bin_data(data, bins=10):
    '''
        bin data
    '''
    # bin data data
    hist_counts, bin_edges = np.histogram(data, bins=bins, density=False)

    # assume poisson distributed of counts are sigma2 = hist_value
    var_counts = hist_counts

    # handle additional value
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, bin_edges, hist_counts, var_counts

def residuals(parameters, bin_edges, hist_count, var_count, func_target):
    '''
        chi square residuals for generalized least squares
    '''
    mu, sigma = parameters

    delta = hist_count - func_target(bin_edges, mu, sigma, hist_count.sum())
    
    G = np.linalg.cholesky(var_count)
    r = np.linalg.solve(G, delta)

    return r

