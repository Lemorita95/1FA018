import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

''' constants and parameters '''
X = 12.0 # amount of time
N = 200 # steps
dt = X/(N-1)

C = 1.2e7 # thermal mass (J/°C)
U = 3e6 / X # heat loss
h = 1.2e7 # heating per kg wood

T0 = 20
T_min = 18.0
T_max = 24.0
r_max = 3.0 # kg/h

t = np.linspace(0,X,N)

''' measured variables'''
T_target = 18 + 6*(np.sin(np.pi*(t-2)/8)**2)  # target curve
T_out = -5 + 5*np.cos(2*np.pi*t/24)          # outdoor temp

'''parametrization of r(t), mass rate'''
# piecewise constant r(t) over M intervals
M = 20
interval_len = N // M
# initial guess
r_init = np.ones(M) * 1.0

'''indoors temperature'''
def simulate_T(r_params):
    '''
        this function hold the constraints of the lagrangian function
    '''
    T_sim = np.zeros(N)
    T_sim[0] = T0
    for k in range(N-1):
        idx = k // interval_len
        r_k = np.clip(r_params[idx], 0, r_max)  # enforce bounds
        T_sim[k+1] = T_sim[k] + dt*(h*r_k - U*(T_sim[k]-T_out[k]))/C # constraint
    return T_sim

'''function to minize'''
def lagrangian_cost(r_params):
    T_sim = simulate_T(r_params)
    # chi-squared fit
    chi2 = np.sum((T_sim - T_target)**2)
    # wood cost
    wood = np.sum(r_params)*dt*interval_len
    return chi2 + 0.01*wood  # apply penalty on wood use

'''minization result, this will find the optimal parameters for r(t)'''
res = minimize(lagrangian_cost, r_init, method='L-BFGS-B', bounds=[(0,r_max)]*M)

r_opt_params = res.x
T_opt = simulate_T(r_opt_params)

'''create the r(t) step function and apply the boundaries for r(t)'''
r_opt = np.zeros(N)
for k in range(N):
    idx = k // interval_len
    r_opt[k] = np.clip(r_opt_params[idx], 0, r_max)

'''final results'''
wood_used = np.trapezoid(r_opt, dx=dt)
print(f"Total wood used [kg]: {wood_used:.3f}")

# ===================== PLOTS =====================
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(8, 8), dpi=300)
axs[0].plot(t, T_opt, label="Fitted Interior Temp", color='black')
axs[0].plot(t, T_target, '--', label="Target Temp", color='red')
axs[0].plot(t, T_out, ':', label="Outdoor Temp", color='blue')
# plt.axhline(T_min, color='orange', linestyle='--', label='T_min')
# plt.axhline(T_max, color='red', linestyle='--', label='T_max')
# axs[0].set_title("Interior Temperature Tracking via Parameterized r(t)")
axs[0].set_xlabel("Hours")
axs[0].set_ylabel("°C")
axs[0].legend()

axs[1].plot(t, r_opt, label="Optimal Burn Rate", color='green')
# axs[1].set_title("Optimized Burn Rate (Piecewise Constant)")
axs[1].set_xlabel("Hours")
axs[1].set_ylabel("kg/h")
axs[1].legend()

plt.tight_layout()
plt.savefig('result.png', bbox_inches='tight', dpi=300)
# plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.plot(t, T_target, '--', label="Target Temp", color='red')
plt.savefig('target.png', bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.plot(t, r_opt, label="Optimal Burn Rate r(t)", color='green')
plt.savefig('burn_rate.png', bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.plot(t, T_out, ':', label="Outdoor Temp", color='blue')
plt.savefig('outdoors.png', bbox_inches='tight', dpi=300)
plt.close()