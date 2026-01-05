from helpers import IMAGES, os, np, plt, \
    least_squares, polar_method, bin_data, residuals, norm

ss = np.random.SeedSequence([42, 3])
RNG = np.random.default_rng(ss)

elems_R = {
    (1,1): 0.65,
    (1,2): 0.25,
    (1,3): 0.1,

    (2,1): 0.25,
    (2,2): 0.4,
    (2,3): 0.25,
    (2,4): 0.1,

    (9,10): 0.25,
    (9,9): 0.4,
    (9,8): 0.25,
    (9,7): 0.1,

    (10,10): 0.65,
    (10,9): 0.25,
    (10,8): 0.1,
}

# fill in the remaining elements
m = n = 10
R = np.zeros((m,n)) 
for (i, j), v in elems_R.items():
    R[i-1][j-1] = v

for i in range(m):
    for j in range(n):
        if i == j and R[i][j] == 0:
            R[i][j] = 0.4
        
    if 1 < i < 8:
        R[i][i-2] = 0.1
        R[i][i-2+1] = 0.2
        R[i][i-2+3] = 0.2
        R[i][i-2+4] = 0.1

R = 0.9*R
print(R)

'''
    Q3.a) 
    Generate a distribution with 1000 events with these properties, organize
    them in a histogram with 10 bins and “smear” it with the response matrix R: f’(x’)= Rf(x).
    Extract the parameters of the folded distribution (mean and width), by e.g. a LSQ fit.
'''
N = 1000
mu = 5
sigma = 2
bins = 10

# generate events
distribution = []
for i in polar_method(N, mu, sigma, RNG):
    distribution.append(i)
distribution = np.array(distribution)

# get one of the normal outputs
x = distribution[:,2]

# bin it
bin_centers, bin_edges, hist_counts, var_counts = bin_data(x, bins=bins)

print('\nbinned data for polar method')
print(hist_counts)
print(hist_counts.sum())

counts_folded = R @ hist_counts
var_counts_folded = R @ np.diag(var_counts) @ R.T

print('\nbinned data polar method folded')
print(counts_folded)
print(counts_folded.sum())

# compute chi squared fit
def func_target(bin_edges, mu, sigma, N):
    '''
        this function computes the expected value of a binned normal distribution
        method: compute the expected bin value of x by the CDF of x_i+1 and x_i
    '''
    cdf = norm.cdf
    delta = cdf(bin_edges[1:], mu, sigma) - cdf(bin_edges[:-1], mu, sigma)
    return N * delta

# set up generalized least squares
N0 = counts_folded.sum()
mu0 = np.sum(bin_centers * counts_folded) / N0
sigma0 = np.sqrt(np.sum(counts_folded * (bin_centers - mu0)**2) / N0)
x0 = [mu0, sigma0] # initial guess of parameters

args = (bin_edges, counts_folded, var_counts_folded, func_target)
res = least_squares(residuals, x0, args=args)
mu_chi2, sigma_chi2 = res.x
print()
print(rf'estimated parameters for folded distribution: $\mu: {mu_chi2},\sigma: {sigma_chi2}$')

# compute the curves
model_fit = func_target(bin_edges, mu_chi2, sigma_chi2, N0)

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.plot(bin_centers, hist_counts, label=f'binned polar method. N={hist_counts.sum():.1f}')
ax.plot(bin_centers, model_fit, label=rf"LSQ Fit: $X \sim \mathcal{{N}}({mu_chi2:.4f}, {sigma_chi2:.4f}^2)$")
ax.bar(bin_centers, counts_folded, label=rf"folded data $f^\prime(x) = R f(x)$. N={counts_folded.sum():.1f}", color='grey', alpha=0.5)
ax.legend()
ax.set_ylabel("Count of events")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q3_a1.png'), dpi=300)
# plt.show()
plt.close()

'''
    Q3.b) 
    Generate 1000 new events of the variable x’, using the parameters extracted
    from the folded histogram in 3a). Note that this distribution should NOT be multiplied by
    R since it should already have the properties (i.e. mean and width) of a folded
    distribution. Hence, this new distribution “simulates” a distribution that has been distorted
    by the detector resolution and efficiency. Now use correction factor method to unfold
    this distribution to obtain an estimate of “the truth”, i.e. 𝑓̂(𝑥). Please also calculate the
    uncertainties (covariance matrix). Illustrate with histograms/plots on all levels of
    generating and unfolding.
'''
N = 1000

# generate events
distribution_folded = []
for i in polar_method(N, mu_chi2, sigma_chi2, RNG):
    distribution_folded.append(i)
distribution_folded = np.array(distribution_folded)

# get one of the normal outputs
x_folded = distribution_folded[:,2]

# plot generated distribution
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.plot(x_folded, lw=0.5, color='blue', label=rf'new distribution $f^\prime(x) \sim \mathcal{{N}}({mu_chi2:.4f}, {sigma_chi2:.4f}^2)$. N={len(x_folded):.1f}')
ax.set_ylabel("x")
ax.set_xlabel("event")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q3_b1.png'), dpi=300)
# plt.show()
plt.close()

# bin it. as they are now independent events, the var = # events in bin
# use same bin_edges as before for alignment
x_centers, x_edges, x_counts, var_x = bin_data(x_folded, bins=bin_edges)
print('\nbinned data for new folded distribution')
print(x_counts)
print(x_counts.sum())

# compute correction factor from a)
C = hist_counts / counts_folded

# plot binned distribution
fig, axs = plt.subplots(ncols=2, figsize=(10, 4), dpi=300)
axs[0].errorbar(x_centers, x_counts, yerr=var_x**0.5, label=rf'binned $f^\prime(x) \sim \mathcal{{N}}({mu_chi2:.4f}, {sigma_chi2:.4f}^2)$. N={x_counts.sum():.1f}',
            fmt='o-', markersize=6, capsize=4, capthick=1.2,
            markerfacecolor='white', markeredgecolor='black', color='blue', elinewidth=1.2,)
axs[0].legend(fontsize=9, loc='lower left')
axs[0].set_ylabel("Count of events")
axs[0].set_xlabel("x")

axs[1].plot(x_centers, C, marker='o', markerfacecolor='white', markeredgecolor='black', label='correction factor', color='black')
axs[1].legend(fontsize=9)
axs[1].set_ylabel("Correction factor")
axs[1].set_xlabel("x")

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q3_b2.png'), dpi=300)
# plt.show()
plt.close()

# compute unfolded distribution
counts = x_counts * C
covar_counts = var_x * (C ** 2)

print('\ncovariance matrix of unfolded distribution f(x)')
np.set_printoptions(precision=3, suppress=True, linewidth=100)
print(np.diag(covar_counts))

# plot generated distribution corrected
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.errorbar(x_centers, x_counts, yerr=var_x**0.5, label=rf'binned $f^\prime(x)$. N={x_counts.sum():.1f}',
            fmt='o-', markersize=6, capsize=4, capthick=1.2,
            markerfacecolor='white', markeredgecolor='black', color='blue', elinewidth=1.2,)
ax.errorbar(x_centers, counts, yerr=covar_counts**0.5, label=f'unfolded f(x). N={counts.sum():.1f}',
            fmt='o-', markersize=6, capsize=4, capthick=1.2,
            markerfacecolor='white', markeredgecolor='black', color='green', elinewidth=1.2,)
ax.set_ylabel("Count of events")
ax.set_xlabel("x")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q3_b3.png'), dpi=300)
# plt.show()
plt.close()

