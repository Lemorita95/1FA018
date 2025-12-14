from helpers import DATA, IMAGES, os, plt, np, pd

np.random.seed(42)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8
})

# define path for data file
file = os.path.join(DATA, 'data_q2.csv')

# read data file as a pandas dataframe
data = pd.read_csv(file)
data.set_index('id')

# assume unreported systematic uncertainties as 0, i.e. included in the statistical
data.fillna(0.0, inplace=True)

data['sigma_squred'] = data['statistical']**2 + data['systematic']**2
measurements = data['value']
uncertanties = data['sigma_squred']

'''
    2.a combine uncertainties
    assume experiments are independent
    y(x): function we want to propagate the uncertaities
    x: array of measurements each with its uncertainties
    i want: the average of all experiments
    y(x) = 1/N*sum(x) --> average
    J (jacobian) = dy/dx --> (1/N)
    N: number of experiments
'''

def weighted_mean(measurements, uncertanties):
    w_array = 1/uncertanties
    return sum([x * w for x, w in zip(measurements, w_array)]) / sum(w_array)

wmean = weighted_mean(measurements, uncertanties)

# create a Nx1 Jacobian matrix
J = np.array([w/sum(1/uncertanties) for w in 1/uncertanties]).T

# an NxN uncertanties matrix
Sigma = pd.DataFrame(np.diag(uncertanties),index=uncertanties.index,columns=uncertanties.index)
V_x = Sigma.to_numpy()

V_y = J @ V_x @ J.T
sigma_y = V_y**0.5
print(f"{wmean:.4g} +- {sigma_y:.4g}") # 1 significant figure due to uncertainty

'''
    2.b)
    
    z(90%) = 1.645
    z(95%) = 1.960
    
    Frequentist approach: sampling distribution approaches the parent
        - assume a gaussian PDF
        - find alpha from confidence interval (alpha = beta)
        - compute CDF
        - values of x where CDF(x)=alpha and CDF(x)=1 - alpha, are the C.I. boundaries

'''
mean = 0.26
sigma = 0.34

def pdf(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

def integrate(x, y, lower_limit, upper_limit):
    ''''
    compute integral by trapezoidal method
    '''

    integral = [0] # assume initial value is 0

    for i in range(1, len(y)):
        if x[i] < lower_limit or x[i] > upper_limit:
            continue
        dt = (x[i] - x[i-1])
        value = 0.5 * (i - (i-1)) * (y[i] + y[i-1]) * dt
        value += integral[-1]

        integral.append(value)

    return integral

x = np.linspace(mean - 4*sigma, mean + 4*sigma, 5000)
parent = pdf(x, mean, sigma)

confidence_level = 0.90
alpha = (confidence_level) # one sided interval
cdf = np.array(integrate(x, parent, min(x), max(x)))

# find x value for 1 - alpha and alpha at the cdf
idx_upper = np.argmin(abs(cdf - alpha))

print(f"frequentist confidence interval {confidence_level}: {x[idx_upper]:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5,2.5), dpi=300)  # 1x2 grid
ax1.plot(x, parent, c='black', lw=1, markersize=4)
ax1.axvline(x[idx_upper], ls='--', c='red', lw=0.5, markersize=4)
ax2.plot(x, cdf, c='black', lw=1, markersize=4)
ax2.axhline(alpha, ls='--', c='red', lw=0.5, markersize=4)
ax2.axvline(x[idx_upper], ls='--', c='red', lw=0.5, markersize=4)

ax1.set_ylabel("Probability density function", fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.minorticks_on()
ax1.grid(True, linestyle='--', linewidth=0.25, alpha=0.7)
ax1.set_title("(a)")
ax1.legend(['function', f'{confidence_level*100:.0f}% confidence interval'], fontsize=3, loc='upper right')

ax2.set_ylabel("Cumulative density function", fontsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.minorticks_on()
ax2.grid(True, linestyle='--', linewidth=0.25, alpha=0.7)
ax2.set_title("(b)")

fig.supxlabel(r"VALUE [$eV^{2}c^{-4}$]", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "q2_frequentist.pdf"))  # vector graphics
plt.show()

'''
    2.c)
    Bayesian approach: Prior knowledge of f(x) is used.
        - assume gaussian PDF
'''

parent = pdf(x, mean, sigma)
trunc_parent = np.where(x<0, 0, parent)
normalization = integrate(x, trunc_parent, 0, max(x))[-1]
G = trunc_parent/normalization
cdf_G = np.array(integrate(x, G, min(x), max(x)))

# find x value for 1 - alpha and alpha at the cdf
idx_upper = np.argmin(abs(cdf_G - alpha))

print(f"bayesian confidence interval {confidence_level}: {x[idx_upper]:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5,2.5), dpi=300)  # 1x2 grid
ax1.plot(x, trunc_parent/normalization, c='black', lw=1, markersize=4)
ax1.axvline(x[idx_upper], ls='--', c='red', lw=0.5, markersize=4)
ax2.plot(x, cdf_G, c='black', lw=1, markersize=4)
ax2.axhline(alpha, ls='--', c='red', lw=0.5, markersize=4)
ax2.axvline(x[idx_upper], ls='--', c='red', lw=0.5, markersize=4)

ax1.set_ylabel("Probability density function", fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.minorticks_on()
ax1.grid(True, linestyle='--', linewidth=0.25, alpha=0.7)
ax1.set_title("(a)")
ax1.legend(['function', f'{confidence_level*100:.0f}% confidence interval'], fontsize=3, loc='upper right')

ax2.set_ylabel("Cumulative density function", fontsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.minorticks_on()
ax2.grid(True, linestyle='--', linewidth=0.25, alpha=0.7)
ax2.set_title("(b)")

fig.supxlabel(r"VALUE [$eV^{2}c^{-4}$]",fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "q2_bayesian.pdf"))  # vector graphics
plt.show()

