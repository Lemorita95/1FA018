from helpers import IMAGES, os, np, plt, \
    least_squares, polar_method, bin_data, residuals, norm

ss = np.random.SeedSequence([42, 2])
RNG = np.random.default_rng(ss)

N = 1000
mu = 5
sigma = 2
bins = 10

distribution = []

for i in polar_method(N, mu, sigma, RNG):
    distribution.append(i)
distribution = np.array(distribution)

u, v = distribution[:,0], distribution[:,1]
x1, x2 = distribution[:,2], distribution[:,3]

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 4), dpi=300)
axs = axs.flatten() # Flatten the 2x2 array to 1D
axs[0].hist(u, bins=bins, density=True)
axs[0].set_xlabel(r'$u$')
axs[1].hist(v, bins=bins, density=True)
axs[1].set_xlabel(r'$v$')
axs[2].hist(x1, bins=bins, density=True)
axs[2].set_xlabel(r'$x_1$')
axs[3].hist(x2, bins=bins, density=True)
axs[3].set_xlabel(r'$x_2$')
fig.suptitle(f"N={N}, bins={bins}", fontsize=10)
fig.supylabel("Probability density")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q2_1.png'), dpi=300)
# plt.show()
plt.close()

'''
    Check that the
    generated distribution has the desired mean and width
'''

bin_centers, bin_edges, hist_counts, var_counts = bin_data(x1, bins=bins)

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
N0 = hist_counts.sum()
mu0 = np.sum(bin_centers * hist_counts) / N0
sigma0 = np.sqrt(np.sum(hist_counts * (bin_centers - mu0)**2) / N0)
x0 = [mu0, sigma0] # initial guess of parameters

args = (bin_edges, hist_counts, np.diag(var_counts), func_target)
res = least_squares(residuals, x0, args=args)
mu_chi2, sigma_chi2 = res.x

# compute the curves
model_fit = func_target(bin_edges, mu_chi2, sigma_chi2, N0)
model_true = func_target(bin_edges, mu, sigma, N0)

# plot results
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.plot(bin_centers, model_fit, label=rf"LSQ Fit: $X \sim \mathcal{{N}}({mu_chi2:.4f}, {sigma_chi2:.4f}^2)$")
ax.plot(bin_centers, model_true, label=rf"desired $X \sim \mathcal{{N}}({mu:.1f}, {sigma:.1f}^2)$")
ax.bar(bin_centers, hist_counts, color='grey', alpha=0.4, label=r'polar method distribution $x_1$')
ax.set_xlabel(r'$x_1$')
ax.legend()
ax.set_ylabel("Count of events")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q2_2.png'), dpi=300)
# plt.show()
plt.close()
