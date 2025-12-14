from helpers import IMAGES, os, np, plt, math, integrate

'''
    3.a)
    H0: the experimental results from the two experiments are compatible
        with each other at 5% and 1% significance. Please include all steps in your
        solution, in particular, how you define your test statistic and the critical value.
'''

imb_event = [80, 44, 56, 65, 33, 52, 42, 104]
kam_event = [18, 32, 30, 38]

max_val = max(imb_event + kam_event)

# define test statistic
def test_statistic(S1, S2):
    '''
    compute test statistic D_12 from 2 CDFs
    '''
    diff = np.abs(S1-S2)
    idx_max = np.argmax(diff)

    return np.max(diff), idx_max

def step_func(x, x_i):
    '''
        function for s(x_i)
        x_i is array-like
    '''
    return np.where(x_i<=x, 1, 0)

def cdf(x_array, data):
    '''
        function for S_n(x)
        x: array of values
        data: experiment data
    '''
    n = len(data)

    cdf = []
    for x in x_array:
        tmp = step_func(x, data)
        cdf.append((1/n)*sum(tmp))
    return np.array(cdf)

x_array = np.linspace(0, max_val, 1000)
cdf_imb = cdf(x_array, imb_event)
cdf_kam = cdf(x_array, kam_event)

D_12, idx_max = test_statistic(cdf_kam, cdf_imb)

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# --- Curves ---
ax.plot(
    x_array, cdf_imb, lw=1.5, color='tab:blue',
    label='CDF (IMB)'
)
ax.plot(
    x_array, cdf_kam, lw=1.5, color='tab:orange',
    label='CDF (KAM)'
)

# --- Vertical line at maximum difference ---
ax.axvline(
    x_array[idx_max], color='red', linestyle='--',
    linewidth=1.0, alpha=0.8
)

# --- Grid + spines ---
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# --- Labels ---
ax.set_xlabel('x', fontsize=12, labelpad=6)
ax.set_ylabel('CDF', fontsize=12, labelpad=6)

# --- Title ---
ax.set_title(
    rf'$D_{{12}} = {D_12:.4f} \;\; @ \;\; x = {x_array[idx_max]:.4f}$',
    fontsize=13, pad=10
)

# --- Legend ---
ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor='black')

# --- Ticks ---
ax.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=10)
ax.tick_params(axis='both', which='minor', length=3, width=1.0)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q3_a.png'), dpi=300)
plt.show()


class direct_method():
    def __init__(self, m, n, d):
        self.m = m
        self.n = n
        self.d = d

    def is_in_boundary(self, i, j):
        return np.abs(i/self.m - j/self.n) < self.d
    
    def A(self, i, j):

        if i < 0 or j < 0:
            return 0
        
        if (i==0) and (j==0):
            return 1
        
        if not self.is_in_boundary(i,j):
            return 0
        
        return self.A(i-1, j) + self.A(i, j-1)
    
    def combinatorial(self, i, j):
        return math.comb(i+j, j)
    
    def P2(self):
        '''
            P(D>=d|F=G) =  significance
        '''
        return 1-self.A(self.m, self.n)/self.combinatorial(self.m, self.n)
    

m, n, d = len(imb_event), len(kam_event), D_12
dm = direct_method(m, n, d)
print(f"H0: both dataset come from the same distribution\n")
print(f'KS statistic D={D_12}')
p_val = dm.P2()
print(f'p-value =" {p_val}')
if p_val >= 0.05:
    print(f'p-value {p_val:.3f} >= 0.05, we cannot reject the null hypothesis at this significance')
else:
    print(f'p-value {p_val:.3f} < 0.05, we can reject the null hypothesis at this significance')

if p_val >= 0.01:
    print(f'p-value {p_val:.3f} >= 0.01, we cannot reject the null hypothesis at this significance')
else:
    print(f'p-value {p_val:.3f} < 0.01, we can reject the null hypothesis at this significance')

# print("\nusing scipy")
# from scipy import stats
# D_stat, p_value = stats.ks_2samp(imb_event, kam_event)
# print("KS statistic D =", D_stat)
# print("p-value =", p_value)

'''
    3.b)
    H0:  the experimental results are compatible with the expected
    angular distribution at 5% and 1% significance, treating all the data as coming
    from the same source (i.e. forming one common sample out of the two).
'''

def angular_distribution(costheta, alpha=0.1):
    return 1 + alpha * costheta

events = np.concatenate([imb_event, kam_event])
cosEvents = np.cos(np.radians(events))

x_array = np.linspace(-1, 1, 360)
cdf_events = cdf(x_array, cosEvents)
cdf_distribution = integrate(x_array, angular_distribution(x_array), -np.inf, np.inf)
cdf_distribution /= cdf_distribution[-1]

D_12, idx_max = test_statistic(cdf_events, cdf_distribution)

'''
    from table data P(D_n <= d_alpha) ~ 1 - alpha
        significance alpha: 5% and n = 12 -> d_alpha = 0.3754
        significance alpha: 1% and n = 12 -> d_alpha = 0.4491
'''
d_alpha_5 = 0.3754
d_alpha_1 = 0.4491

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# --- Curves ---
ax.plot(
    x_array, cdf_events, lw=1.5, color='tab:blue',
    label='CDF (events)'
)
ax.plot(
    x_array, cdf_distribution, lw=1.5, color='tab:orange',
    label='CDF (distribution)'
)

# --- Vertical line at max difference ---
ax.axvline(
    x_array[idx_max], color='red', linestyle='--',
    linewidth=1.0, alpha=0.8
)

# --- Grid + spines ---
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# --- Labels ---
ax.set_xlabel('x', fontsize=12, labelpad=6)
ax.set_ylabel('CDF', fontsize=12, labelpad=6)

# --- Title ---
ax.set_title(
    rf'$D_{{12}} = {D_12:.4f} \;\; @ \;\; x = {x_array[idx_max]:.4f}$',
    fontsize=13, pad=10
)

# --- Legend ---
ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor='black')

# --- Ticks ---
ax.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=10)
ax.tick_params(axis='both', which='minor', length=3, width=1.0)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'q3_b.png'), dpi=300)
plt.show()


print(f"\nH0: The experimental results are compatible with the expected angular distribution\n")
print(f'test statistic = {D_12}')
print(f'd_alpha(alpha=5%, n=12) = {d_alpha_5}')
print(f'd_alpha(alpha=1%, n=12) = {d_alpha_1}')

if D_12 < d_alpha_5:
    print(f'we cannot reject the null hypothesis at 5% significance')
else:
    print(f'we can reject the null hypothesis at 5% significance')

if D_12 < d_alpha_1:
    print(f'we cannot reject the null hypothesis at 1% significance')
else:
    print(f'we can reject the null hypothesis at 1% significance')


# def angular_generator(N, alpha=0.1):
#     '''
#         hit or miss method
#     '''
#     RNG = np.random.default_rng(42)

#     x_min, x_max = -1, 1
#     y_max = 1 + alpha
    
#     i=0
#     while i < N:
#         r1 = RNG.uniform(0, 1)
#         # r1 -> x
#         costheta = x_min + r1 * (x_max - x_min)

#         r2 = RNG.uniform(0, 1)
#         y = r2 * y_max

#         W = angular_distribution(costheta, alpha)

#         if y <= W:
#             i += 1
#             yield costheta


# N = 10000
# alpha = +0.1

# costheta_distribution = []

# for i in angular_generator(N, alpha):
#     costheta_distribution.append(i)

# costheta_distribution = np.array(costheta_distribution)

# W_cos_theta = np.array([angular_distribution(x, alpha) for x in costheta_distribution])
# cdf_MC = cdf(x_array, costheta_distribution)
# cdf_MC /= cdf_MC[-1]

# idx = np.argsort(x_array)
# x = x_array[idx]
# y = cdf_MC[idx]

# plt.plot(x_array, cdf_events)
# plt.plot(x, y)
# plt.show()