import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from matplotlib.patches import Ellipse

IMAGES = os.path.join(os.path.dirname(__file__), 'images')

def differentiate(x, y):
    ''''
    compute derivative by finite differences
    '''

    derivative = []

    for i in range(1, len(y)):
        dy = y[i] - y[i-1]
        dx = x[i] - x[i-1]
        value = dy/dx

        derivative.append(value)

    return derivative

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

def discrete_integrate(x, y, lower_limit=0, upper_limit=None):
    ''''
    compute discrete integration
    '''

    dx = x[1] - x[0]  # assume uniform spacing
    cdf_vals = np.zeros_like(y)
    cdf_vals[0] = y[0] * dx
    for i in range(1, len(y)):
        cdf_vals[i] = cdf_vals[i-1] + 0.5 * (y[i] + y[i-1]) * dx
    return cdf_vals

def find_cl_index(y, cl):
    '''
    find index corresponding to confidence level cl
    '''
    idx = np.argmin(np.abs(y - cl))
    return idx

def plot_fit(parameter_hat, x_array, y_array, sigma_x_array, sigma_y_array, chi_squared, model, label=''):
    y_fit = model(parameter_hat, x_array)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Plot data with error bars
    ax.errorbar(
        x_array, y_array, xerr=sigma_x_array, yerr=sigma_y_array,
        fmt='o', markersize=6, capsize=4, capthick=1.2,
        markerfacecolor='white', markeredgecolor='black',
        ecolor='gray', elinewidth=1.2,
    )

    # Plot model fit
    ax.plot(
        x_array, y_fit,
        lw=2, color='tab:blue', label=rf'{label} $\chi^2={chi_squared:.4f}$'
    )

    # Grid and spines
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Axis labels
    ax.set_xlabel('x', fontsize=12, labelpad=6)
    ax.set_ylabel('y', fontsize=12, labelpad=6)

    # Legend
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor='black')

    # Ticks
    ax.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=10)
    ax.tick_params(axis='both', which='minor', length=3, width=1.0)

    return fig

def prob_of_fit(chi2, dof):
    """
    @brief Chi-square probability density function (PDF).

    Evaluates the chi-square distribution PDF for a given chi-square value and
    number of degrees of freedom:

        f(chi2; dof) = 1 / ( 2^(dof/2) Γ(dof/2) ) * chi2^(dof/2 - 1) * exp(-chi2 / 2)

    This function can be used as the integrand for computing the cumulative
    or tail probabilities of the chi-square distribution.

    @param chi2  Chi-square value (scalar).
    @param dof   Degrees of freedom of the chi-square distribution.

    @return Value of the chi-square PDF at chi2.
    """

    numerator = chi2**((dof/2 - 1)) * np.exp(-chi2 / 2)
    denominator = 2**(dof/2) * gamma(dof / 2)
    probability = numerator / denominator
    return probability

# define Jacobian for error propagation
def jacobian(a1, a2, w0=1, cap=0.02e-6):
    '''
    Jacobian matrix for
        L = R.a1/w0 = a1/(w0^2.a2.C)
        R = 1/(w0.a2.C)

    '''

    A = 1 / ((w0**2) * a2 * cap)
    B = -a1 / ((w0**2) * cap * (a2**2))
    C = 0
    D = -1 / (w0 * cap * (a2**2))

    L_array = np.array([A, B])
    R_array = np.array([C, D])

    return np.vstack([L_array, R_array])

def error_propagation(parameter_hat, V_parameter):

    a1 = parameter_hat[0]
    a2 = parameter_hat[1]
    w0 = 1
    cap = 0.02e-6
    L = a1 / ((w0**2) * a2 * cap)
    R = 1 / (w0 * a2 * cap)

    # error propagation
    J = jacobian(*tuple(parameter_hat), w0, cap)

    V_LR = J @ V_parameter @ J.T

    return R, L, V_LR

def plot_ellipsis(R, L, V_LR, title_label='Ellipse'):
    # Compute ellipse properties
    eigenvalues, eigenvectors = np.linalg.eig(V_LR)
    # Major axis direction
    vx, vy = eigenvectors[:, 0]
    angle = np.degrees(np.arctan2(vy, vx))
    width, height = 2 * np.sqrt(eigenvalues)  # 1-sigma ellipse

    # Correlation coefficient
    v_0 = np.sqrt(V_LR[0,0])
    v_1 = np.sqrt(V_LR[1,1])
    rho = V_LR[0, 1] / (v_0 * v_1)

    # Conditional uncertainties
    delta_R = np.sqrt(1 - rho**2) * v_1  # sigma_R|L
    delta_L = np.sqrt(1 - rho**2) * v_0  # sigma_L|R

    # Plot setup
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Ellipse
    ellipse1sigma = Ellipse(
        xy=(L, R),
        width=width,
        height=height,
        angle=angle,
        fc='none', ec='brown', lw=2.5, ls='-'
    )
    ax.add_patch(ellipse1sigma)

    # Center point
    ax.scatter(L, R, color='black', s=30, zorder=3)

    # Reference lines
    ax.axvline(x=L, color='gray', alpha=0.3, linestyle='--', lw=1)
    ax.axhline(y=R, color='gray', alpha=0.3, linestyle='--', lw=1)
    ax.hlines(y=R, xmin=L, xmax=L+delta_L, color='gray', linestyle='--', lw=1)
    ax.vlines(x=L, ymin=R, ymax=R+delta_R, color='gray', linestyle='--', lw=1)

    # Labels
    ax.set_xlabel('L', fontsize=12, labelpad=6)
    ax.set_ylabel('R', fontsize=12, labelpad=6)

    ax.tick_params(axis='x', labelrotation=45)

    # Grid and spines
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Title with LaTeX
    ax.set_title(rf'$\Delta_L={delta_L:.4f}, \Delta_R={delta_R:.4f}$', fontsize=12, pad=6)
    plt.suptitle(rf'{title_label}, $p={rho:.4f}$', fontsize=14, y=1.02)

    return fig, delta_R, delta_L