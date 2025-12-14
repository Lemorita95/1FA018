from matplotlib.ticker import PercentFormatter

from helpers import IMAGES, os, math, plt, np, integrate, find_cl_index



'''
    Q2.a) 
    None of these 500 people show any symptoms of a certain rare but possible side effect.
    Assume (somewhat unrealistically) that these symptoms cannot occur for any other reason
    (i.e. the background is zero). Based on this, estimate the 95% C.L. upper limit of the risk
    (quantified in %) of obtaining this side effect as a consequence of the medicine.

    set up:
    - parameter: risk `p` of side effect
    - random variable: number of observed side effects `k_obs`

    solution:
    - use frequentist approach
    - define statistical model (general binomial distribution)
    - compute CDF as functions of `p` P(k|N, `p`) from k=0 to k=`k_obs`
        - sum each CDF value for k=0 to k=`k_obs`
    - find y = 1-CL at the cdf function, get corresponding x value = p_up
'''

def pdf(k, n, p):
    ''' binomial probability density function '''
    comb = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    value = comb * (p ** k) * ((1 - p) ** (n - k))
    return value

def cdf(k, n, p_grid, plot=False):
    """
    Compute CDF of the PDF for given k and n, over p_grid
    """

    cdf_vals = np.zeros_like(p_grid)
    for i in range(k+1):
        pmf_vals = np.array(pdf(i, n, p_grid))
        if plot:
            ax.plot(p_grid, pmf_vals, label=rf'P($N_{{\rm obs}} = {i}$|{n},p)', lw=0.5)
        cdf_vals += pmf_vals

    if plot:
        fig, ax = plt.subplots(figsize=(3.5,2.5), dpi=300)
    
    if plot:

        x_up = p_grid[find_cl_index(cdf_vals, 1-CL)]
        y_up = 1-CL

        ax.plot(p_grid, cdf_vals, label='CDF', color='black', linewidth=1.5)

        ax.vlines(x=x_up, ymin=0, ymax=y_up ,color='red', linestyle='--', lw=1)
        ax.hlines(y=y_up, xmin=0, xmax=x_up ,color='red', linestyle='--', lw=1)

        ax.annotate(r'$p_{{\rm 0.95CL}}$ = {:.6f}%'.format(x_up*100),
                    xy=(x_up*1.1, y_up*1.1),
                    fontsize=8, color='red'
                    )

        ax.set_xlabel(r'Risk $p$')          # LaTeX-friendly

        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=2)) 
        
        ax.set_xlim(0, max(p_grid))
        ax.set_ylim(0, max(cdf_vals)*1.1)

        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES, f'q2_cdf_N{n}.png'), dpi=300)
        plt.show()

    return cdf_vals


N = 50000
k_obs = 0  # observed number of events
CL = 0.95

x = np.linspace(0, 0.0002, 1000)  # risk values from 0% to 1%
cdf_values = cdf(k_obs, N, x, plot=False)

idx = find_cl_index(cdf_values, 1-CL)
p_up = x[idx]
print(f"Numerical {CL*100:.0f}% CL upper limit = {p_up*100:.6f}%")

'''
    Q2.b) 
    Same as Q2.a) but N=50000
'''

'''
    Q2.c) 
    What if 5 people out of 50 000 indeed show symptoms of the side effect, but that a
    placebo study predicts that 8 out of 50 000 people should get symptoms for other
    reasons than as a side-effect from the medicine? Estimate (an approximation is
    sufficient) the 95% C.L. upper limit using the Bayesian approach.

    set up:
    k_obs: observed persons with side effect
    k_s: additional person due to medicine (INTEREST)
    k_background: background, other reasons than medicine

    solution:
    - use Bayesian approach
    - define statistical model (likelihood), poisson distribution
    - find k_s = k_obs - k_background

'''

N = 50000
k_obs = 5 # out of N (n)
k_background = 8 # out of N (nb)
CL = 0.95

def likelihood(k_obs, mu_s, mu_b):
    '''
    statistical model: poisson likelihood function
    '''
    mu = mu_s + mu_b

    return ((mu ** k_obs)/math.factorial(k_obs)) * np.exp(-mu)

def posteriori(parameter, parent):
    prior = np.where(parameter<0, 0, 1) # prior, since using medicine does not decrease side effects
    trunc_parent = parent * prior 
    normalization = integrate(parameter, trunc_parent, 0, max(parameter))[-1]
    G = trunc_parent/normalization

    return G

mu_b = k_background # since in a poisson distribution the expectation value is the number of events
mu_s = np.linspace(-5,10,20000) # additional observations due to medicine

parent = likelihood(k_obs, mu_s, mu_b) # this is P(k_obs|mu_s)

# we want the posteriori P(mu_s|k_obs)
G = posteriori(mu_s, parent)
cdf_G = np.array(integrate(mu_s, G, min(mu_s), max(mu_s)))

idx = find_cl_index(cdf_G, CL)
print(f"Numerical {CL*100:.0f}% CL upper limit = {mu_s[idx]:.2f} ({mu_s[idx]*100/N:.6f}%)")

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# --- Curves ---
ax.plot(
    mu_s, parent, lw=1.5,
    label=r'$P(k_{\mathrm{obs}} \mid \mu_{s})$',
    color='tab:blue'
)
ax.plot(
    mu_s, G, lw=1.5,
    label=r'Posterior $P(\mu_{s}\mid k_{\mathrm{obs}})$',
    color='tab:orange'
)
ax.plot(
    mu_s, cdf_G, lw=1.5,
    label=r'Posterior CDF',
    color='tab:grey'
)

# --- Confidence level lines ---
ax.axhline(CL, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax.axvline(mu_s[idx], color='red', linestyle='--', linewidth=1.5, alpha=0.8)

# --- Labels ---
ax.set_xlabel(r'$\mu_s$', fontsize=12, labelpad=6)
ax.set_ylabel('Probability', fontsize=12, labelpad=6)

# --- Grid + spines ---
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# --- Legend ---
ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor='black')

# --- Ticks ---
ax.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=10)
ax.tick_params(axis='both', which='minor', length=3, width=1.0)

# --- Title optional ---
# ax.set_title('Bayesian Posterior and CDF', fontsize=14, pad=10)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q2_bayesian.png'), dpi=300)
plt.show()

# '''
#     2.c) using binomial distribution
# '''

# def pdf2(k, n, p):
#     ''' 
#         binomial probability density function 
#         P(k|N, `p`)
#     '''
#     comb = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
#     value = comb * (p ** k) * ((1 - p) ** (n - k))
#     return value

# def cdf2(k, n, p_grid, plot=False):
#     """
#     Compute CDF of the PDF for given k and n, over p_grid
#     """
#     if plot:
#         fig, ax = plt.subplots(figsize=(3.5,2.5), dpi=300)


#     cdf_vals = np.zeros_like(p_grid)
#     for i in range(k+1):
        
#         pmf_vals = np.array(pdf(i, n, p_grid)) # sample from the pdf

#         prior = (p_grid >= 0) & (p_grid <= 1)
#         pmf_vals = np.where(prior, pmf_vals, 0) # limite the domain of the parameter
#         if plot:
#             ax.plot(p_grid, pmf_vals, label=rf'P($N_{{\rm obs}} = {i}$|{n},p)', lw=0.5)
#         cdf_vals += pmf_vals
    
#     if plot:

#         x_up = p_grid[find_cl_index(cdf_vals, 1-CL)]
#         y_up = 1-CL

#         ax.plot(p_grid, cdf_vals, label='CDF', color='black', linewidth=1.5)

#         ax.vlines(x=x_up, ymin=0, ymax=y_up ,color='red', linestyle='--', lw=1)
#         ax.hlines(y=y_up, xmin=0, xmax=x_up ,color='red', linestyle='--', lw=1)

#         ax.annotate(r'$p_{{\rm 0.95CL}}$ = {:.2f}%'.format(x_up*100),
#                     xy=(x_up*1.1, y_up*1.1),
#                     fontsize=8, color='red'
#                     )

#         ax.set_xlabel(r'Risk $p$')          # LaTeX-friendly

#         ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=2)) 
        
#         ax.set_xlim(0, max(p_grid))
#         ax.set_ylim(0, max(cdf_vals)*1.1)

#         ax.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(IMAGES, f'q2_cdf_N{n}.png'), dpi=300)
#         plt.show()

#     return cdf_vals

# p_b = mu_b/N
# p_s_grid = np.linspace(0, 0.002, 10000)

# # likelihood function
# lh = pdf(k_obs, N, p_s_grid + p_b)

# # apply pior
# prior = (p_s_grid >= 0) & (p_s_grid <= 1-p_b)
# posterior_unnorm = lh * prior

# # normalize
# from scipy.integrate import trapezoid
# integral = np.trapezoid(posterior_unnorm, p_s_grid)
# Z = trapezoid(posterior_unnorm, p_s_grid)
# posterior = posterior_unnorm / integral

# # compute cdf
# posterior_cdf = np.cumsum(posterior) * (p_s_grid[1] - p_s_grid[0])
# posterior_cdf /= posterior_cdf[-1]

# idx = find_cl_index(posterior_cdf, CL)
# p_up = p_s_grid[idx]
# print(f"Numerical {CL*100:.0f}% CL upper limit = {p_up*100:.6f}%")

# plt.plot(p_s_grid, posterior_cdf)
# plt.plot(p_s_grid, posterior)
# plt.axvline(p_up)
# plt.show()