import numpy as np
from scipy.stats import binom

N = 50000
k_obs = 5
mu_b = 8
p_b = mu_b / N
CL = 0.95

# Grid for the signal probability
p_s_grid = np.linspace(0, 0.002, 10000)  # up to reasonable max

# Initialize CDF
cdf_vals = np.zeros_like(p_s_grid)

# Compute cumulative probability (sum over k=0..k_obs)
for k in range(k_obs+1):
    pmf_vals = binom.pmf(k, N, p_s_grid + p_b)
    # apply flat prior: only keep valid probabilities
    pmf_vals[p_s_grid + p_b > 1] = 0
    pmf_vals[p_s_grid < 0] = 0
    cdf_vals += pmf_vals

# Normalize to 1
cdf_vals /= cdf_vals[-1]

# Find 95% upper limit
p_up = np.interp(CL, cdf_vals, p_s_grid)
print(f"Numerical {CL*100:.0f}% CL upper limit = {p_up*100:.6f}%")
