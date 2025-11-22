from helpers import DATA, IMAGES, os, plt, np, pd, \
    likelihood, ML_estimator, find_boundaries, expectation_value

np.random.seed(42)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8
})

# define path for data file
file = os.path.join(DATA, 'data_q1.csv')

# read data file as a pandas dataframe
data = pd.read_csv(file)
samples = data['Time (s)']

'''
    Question 1.a:
        - derive the ML estimator for the exponential distribution
        - compute likelihood intervals P(𝜃𝑎 < 𝜃 < 𝜃𝑏) = gamma
'''

# compute the estimator
tau_hat = ML_estimator(samples)

# find the boundaries as a root solving problem
lower, upper = find_boundaries(likelihood, samples, tau_hat)

x = [x/100 for x in range(1, 1000, 1)]
y = [np.exp(likelihood(s, samples)) for s in x]

fig, ax = plt.subplots(figsize=(3.5,2.5), dpi=300)
ax.plot(x, y, c='black', lw=1, markersize=4)
ax.axvline(lower, ls='--', c='red', lw=0.5, markersize=4)
ax.axvline(tau_hat, c='red', lw=0.5, markersize=4)
ax.axvline(upper, ls='--', c='red', lw=0.5, markersize=4)

ax.set_xlabel(r"lifetime $\tau$ [s]")
ax.set_ylabel("Probability density")
ax.tick_params(axis='both', which='major', labelsize=8)
ax.minorticks_on()
ax.legend([r'Likelihood function L($\tau$)', 'Confidence interval', r'Estimator $\hat{\tau}$'], fontsize=6, loc='upper right')
ax.grid(True, linestyle='--', linewidth=0.25, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "q1_likelihood.pdf"))  # vector graphics
plt.show()

print(f"68.3% Confidence Interval: [{tau_hat:.2f} - {(tau_hat-lower):.2f}, {tau_hat:.2f} + {(upper-tau_hat):.2f}]")

'''
    Question 1.b:
        - generate N events with computed tau_hat from 1.a
        - check for consistency
        - calculate uncertainties
'''

N_array = [10, 100, 1000, 10000]

fig, axes = plt.subplots(2, 2, figsize=(3.5,2.5), dpi=300)  # 2x2 grid
axes = axes.flatten()  # flatten to 1D for easy indexing

tau_true = tau_hat

for ax, n in zip(axes, N_array):
    samples = np.random.exponential(tau_true, n)

    # compute the estimator
    tau_hat = ML_estimator(samples)

    # find the boundaries as a root solving problem
    lower, upper = find_boundaries(likelihood, samples, tau_hat)

    ax.plot(samples, lw=1, markersize=4, c='black')
    ax.axhline(tau_true, c='green', label='true value', lw=0.5, markersize=4)
    ax.axhline(tau_hat, c='red', label='estimator', lw=0.5, markersize=4)
    ax.axhline(lower, c='red', ls='--', label='lower', lw=0.5, markersize=4)
    ax.axhline(upper, c='red', ls='--', label='upper', lw=0.5, markersize=4)
    # ax.set_title(f"n={n} [{tau_hat:.2f} - {(tau_hat-lower):.2f}, {tau_hat:.2f} + {(upper-tau_hat):.2f}]")

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.minorticks_on()
    # ax.legend(['Likelihood function'], fontsize=6, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.25, alpha=0.7)
    ax.set_title(rf'bias $b={(tau_hat - tau_true):.4f}$', fontsize=5)

fig.supxlabel("Samples")
fig.supylabel(r"Lifetime $\tau$ [s]")
# legend only on one chart
axes[0].legend(['Exponential PDF', r'$\tau_{true}$', r'Estimator $\hat{\tau}$'], fontsize=3, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "q1_consistency.pdf"))  # vector graphics
plt.show()

'''
    Question 1.c:
        - bias for N=50
'''

estimator_array = []
for seed in range(0,100):
    np.random.seed(seed) # set individual seed for each experiment
    experiment = np.random.exponential(tau_true, 50)
    tau_hat = ML_estimator(experiment)
    estimator_array.append(tau_hat)

fig, ax = plt.subplots(figsize=(3.5,2.5), dpi=300)
ax.plot(estimator_array, lw=1, markersize=4, c='black')
ax.axhline(expectation_value(estimator_array), c='red', ls='--', lw=0.5, markersize=4)
ax.axhline(tau_true, c='green', lw=0.5, markersize=4)

ax.set_xlabel("Samples")
ax.set_ylabel(r"Lifetime $\tau$ [s]")

ax.tick_params(axis='both', which='major', labelsize=8)
ax.minorticks_on()
ax.legend(['Exponential PDF', r'E[$\hat{\tau}$]', r'$\tau_{true}$'], fontsize=6, loc='upper right')
ax.grid(True, linestyle='--', linewidth=0.25, alpha=0.7)

plt.title(rf'bias $b={(expectation_value(estimator_array) - tau_true):.4f}$', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "q1_bias.pdf"))  # vector graphics
plt.show()
