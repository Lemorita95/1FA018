from helpers import IMAGES, os, np, plt, plot_fit, prob_of_fit, error_propagation, plot_ellipsis

from matplotlib.patches import Ellipse
from scipy.integrate import quad
from scipy.optimize import minimize

y_array = np.array([-4.017, -2.742, -1.1478, 1.491, 6.873]).astype(float)
sigma_y_array = np.array([0.5, 0.25, 0.08, 0.09, 1.90]).astype(float)
x_array = np.array([22000, 22930, 23880, 25130, 26390]).astype(float)
sigma_x_array = np.array([440, 470, 500, 530, 540]).astype(float)

# plt.plot(x_array, y_array)
# plt.show()

def model(beta, x):
    a1, a2 = beta
    return a1*x - a2/x

def main(show):

    ''' 
        Q4.a)
        Determine the values of L and R, and their uncertainties, of
        "Little Henry" neglecting the uncertainties in x. What is the χ2 of the fit?

        assumptions:
        - uncertainties of x_1 are small compared to y_i
    '''
    print('\nQuestion Q4.a)')

    V_y = np.diag(sigma_y_array**2)
    A = np.vstack((x_array, -x_array**-1)).T 

    def LSF_method(A, V, y):

        '''
        chi2 ~ degrees of freedom, chi2/v ~ 1 for a good fit
        '''

        S = np.linalg.inv(A.T @ np.linalg.inv(V) @ A) @ A.T @ np.linalg.inv(V)
        parameter_hat = S @ y
        chi_squared = ((y - A @ parameter_hat).T @ np.linalg.inv(V) @ (y - A @ parameter_hat))

        V_parameter = np.linalg.inv(A.T @ np.linalg.inv(V) @ A)
        v_0 = V_parameter[0,0] ** 0.5
        v_1 = V_parameter[1,1] ** 0.5

        rho = V_parameter[0, 1] / (v_0 * v_1)
        
        dof = len(y) - len(parameter_hat)
        s2 = chi_squared/dof

        cov_matrix = V_parameter #* s2

        return parameter_hat, chi_squared, rho, cov_matrix

    parameter_hat_y, chi_squared_y, rho_y, V_a1a2_y = LSF_method(A, V_y, y_array)

    print(f"optimal solution found with chi2: {chi_squared_y} @ parameter: {parameter_hat_y}")
    print("Covariance matrix of a1 and a2")
    print(V_a1a2_y)

    fig = plot_fit(
        parameter_hat=parameter_hat_y, 
        x_array=x_array, 
        y_array=y_array, 
        sigma_x_array=None, 
        sigma_y_array=sigma_y_array, 
        chi_squared=chi_squared_y,
        model=model)
    filename = 'q4_fit_y'
    fig.savefig(os.path.join(IMAGES, f'{filename}.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # computes the goodness-of-fit probability (p-value) numerically 
    dof = len(y_array) - len(parameter_hat_y)  # Degrees of freedom
    probability_three, _ = quad(prob_of_fit, chi_squared_y, np.inf, args = (dof,))
    print(f"Goodness-of-fit probability (p-value): {probability_three}")

    R, L, V_LR = error_propagation(parameter_hat_y, V_a1a2_y)
    print(f"L = {L} +- {np.sqrt(V_LR[0][0])}")
    print(f"R = {R} +- {np.sqrt(V_LR[1][1])}")
    print("Covariance matrix of L and R")
    print(V_LR)

    ''' 
        Q4.b.1)
        Plot the covariance ellipse and extract the correlation
        coefficient using the intersect method.
    '''

    print('\nQuestion Q4.b.1)')

    fig, delta_Ly, delta_Ry = plot_ellipsis(R, L, V_LR)
    filename = 'q4_ellipsis_y'
    fig.savefig(os.path.join(IMAGES, f'{filename}.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    ''' 
        Q4.b.2)
        Determine the values of L and R (with uncertainties) of "Little
        Henry" neglecting the errors in y. What is the χ2 of the fit? Plot the covariance
        ellipse to extract the uncertainties and covariance.
        Hint. This problem is non-linear, which requires a numerical solution. You
        can use standard software packages for this.
    '''
    print('\nQuestion Q4.b.2)')

    def eff_variance(df_dx, sigma_x, sigma_y):
        # If sigma_x is None → no uncertainty in x → treat as zero
        if sigma_x is None:
            sigma_x = 0.0

        # If sigma_y is None → no uncertainty in y → treat as zero
        if sigma_y is None:
            sigma_y = 0.0
        return np.maximum((df_dx**2)*(sigma_x**2) + (sigma_y)**2, 1e-12)

    def chi2(beta, x, y, eff_variance):
        return sum(((y - model(beta, x))**2)/eff_variance)

    def func_dfdx(i, x, beta, h):
        return (model(beta, x[i]+h) - model(beta, x[i]-h)) / (2*h)

    def OLS_eff(x, y, sigma_x, sigma_y):
        
        beta = np.array([1e-4, 1e-4], dtype=float) # initial parameter guess
        n = len(x)
        p = len(beta)
        h = 1e-8 # derivative step
        tol = 1e-8 # convergence tolerance

        dfdx = np.zeros_like(x) # initial value for iteration

        # iteratively calculate the parameters
        for _ in range(10):
            
            # calculate effective variance
            delta2 = eff_variance(dfdx, sigma_x, sigma_y)
        
            # minimize chi2 respect to beta
            res = minimize(chi2, beta, args=(x, y, delta2)) # min chi2 w.r.t. parameters
            beta_opt = res.x

            # compute new dfdx
            for i in range(n):
                dfdx[i] = func_dfdx(i, x, beta_opt, h)

            if np.all(np.abs(beta_opt - beta) < tol * (1 + np.abs(beta))):
                # compute chi2 value
                chi_squared = chi2(beta_opt, x, y, delta2)
                break

            # if no convergence, update optimal parameters
            beta = beta_opt

        # after convergence, compute jacobian w.r.t. parameter

        J = np.zeros((n, p))
        f0 = model(beta, x) # function avaluated at optimal beta (min chi2)
        for j in range (p):
            b = beta.copy()
            b[j] += h # variation at f due only to j
            f1 = model(b, x)
            J[:, j] = (f1 - f0) / h

        W = np.diag(delta2)

        # compute variance of residuals scale factor
        dof = n - p # degree of freedom
        s2 = chi_squared/dof # reduced chi2

        cov_matrix = np.linalg.inv(J.T @ W @ J) * s2

        # compute correlation length between the TWO parameters
        v_0 = cov_matrix[0,0] ** 0.5
        v_1 = cov_matrix[1,1] ** 0.5
        rho = cov_matrix[0, 1] / (v_0 * v_1)

        return beta_opt, chi_squared, rho, cov_matrix

    parameter_hat_x, chi_squared_x, rho_x, V_a1a2_x = OLS_eff(x_array, y_array, sigma_x_array, None)

    print(f"optimal solution found with chi2: {chi_squared_x} @ parameter: {parameter_hat_x}")
    print("Covariance matrix of a1 and a2")
    print(V_a1a2_x)

    fig = plot_fit(
        parameter_hat=parameter_hat_x, 
        x_array=x_array, 
        y_array=y_array, 
        sigma_x_array=sigma_x_array, 
        sigma_y_array=None, 
        chi_squared=chi_squared_x,
        model=model)
    filename = 'q4_fit_x'
    fig.savefig(os.path.join(IMAGES, f'{filename}.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # computes the goodness-of-fit probability (p-value) numerically 
    dof = len(y_array) - len(parameter_hat_x)  # Degrees of freedom
    probability_three, _ = quad(prob_of_fit, chi_squared_x, np.inf, args = (dof,))
    print(f"Goodness-of-fit probability (p-value): {probability_three}")

    R, L, V_LR = error_propagation(parameter_hat_x, V_a1a2_x)
    print(f"L = {L} +- {np.sqrt(V_LR[0][0])}")
    print(f"R = {R} +- {np.sqrt(V_LR[1][1])}")
    print("Covariance matrix of L and R")
    print(V_LR)

    fig, delta_Lx, delta_Rx = plot_ellipsis(R, L, V_LR)
    filename = 'q4_ellipsis_x'
    fig.savefig(os.path.join(IMAGES, f'{filename}.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)


    ''' 
        Q4.c)
        Determine the values of L and R (with uncertainties) of
        "Little Henry", taking into account both the uncertainties in x and y,
        using the method of effective variance. What is the χ2 of the fit?
    '''
    print('\nQuestion Q4.c)')


    parameter_hat_xy, chi_squared_xy, rho_xy, V_a1a2_xy = OLS_eff(x_array, y_array, sigma_x_array, sigma_y_array)

    print(f"optimal solution found with chi2: {chi_squared_xy} @ parameter: {parameter_hat_xy}")
    print("Covariance matrix of a1 and a2")
    print(V_a1a2_xy)

    fig = plot_fit(
        parameter_hat=parameter_hat_xy, 
        x_array=x_array, 
        y_array=y_array, 
        sigma_x_array=sigma_x_array, 
        sigma_y_array=sigma_y_array, 
        chi_squared=chi_squared_xy,
        model=model)
    filename = 'q4_fit_xy'
    fig.savefig(os.path.join(IMAGES, f'{filename}.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # computes the goodness-of-fit probability (p-value) numerically 
    dof = len(y_array) - len(parameter_hat_xy)  # Degrees of freedom
    probability_three, _ = quad(prob_of_fit, chi_squared_xy, np.inf, args = (dof,))
    print(f"Goodness-of-fit probability (p-value): {probability_three}")

    R, L, V_LR = error_propagation(parameter_hat_xy, V_a1a2_xy)
    print(f"L = {L} +- {np.sqrt(V_LR[0][0])}")
    print(f"R = {R} +- {np.sqrt(V_LR[1][1])}")
    print("Covariance matrix of L and R")
    print(V_LR)

    fig, delta_Lxy, delta_Rxy = plot_ellipsis(R, L, V_LR)
    filename = 'q4_ellipsis_xy'
    fig.savefig(os.path.join(IMAGES, f'{filename}.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    ''' 
        Q4.d)
        Plot the results of the fits together with the data. Do you observe any
        trend in the uncertainties and the χ2 for the cases a-c? Is this expected?
    '''
    print('\nQuestion Q4.d)')

    # all fits vs data
    # include information of conditional uncertainties and chi2

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)  # Slightly larger aspect ratio

    # Plot data with error bars
    ax.errorbar(
        x_array, y_array, xerr=sigma_x_array, yerr=sigma_y_array,
        fmt='o', markersize=6, capsize=4, capthick=1.2,
        markerfacecolor='white', markeredgecolor='black', ecolor='gray', elinewidth=1.2,
    )

    # Plot model fits
    ax.plot(
        x_array, model(parameter_hat_y, x_array),
        label=rf'$\chi_{{y}}^{{2}}={chi_squared_y:.4f}, \Delta_{{L_{{y}}}}={delta_Ly:.4f}, \Delta_{{R_{{y}}}}={delta_Ry:.4f}$', lw=2, color='tab:blue'
    )
    ax.plot(
        x_array, model(parameter_hat_x, x_array),
        label=rf'$\chi_{{x}}^{{2}}={chi_squared_x:.4f}, \Delta_{{L_{{x}}}}={delta_Lx:.4f}, \Delta_{{R_{{x}}}}={delta_Rx:.4f}$', lw=2, color='tab:orange', ls='--'
    )
    ax.plot(
        x_array, model(parameter_hat_xy, x_array),
        label=rf'$\chi_{{xy}}^{{2}}={chi_squared_xy:.4f}, \Delta_{{L_{{xy}}}}={delta_Lxy:.4f}, \Delta_{{R_{{xy}}}}={delta_Rxy:.4f}$', lw=2, color='tab:green', ls=':'
    )

    # Grid and spines
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Axis labels and title
    ax.set_xlabel('x', fontsize=12, labelpad=6)
    ax.set_ylabel('y', fontsize=12, labelpad=6)
    # ax.set_title('Comparison of OLS Fits with Different Uncertainties', fontsize=14, pad=10)

    # Legend
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor='black')

    # Ticks
    ax.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=10)
    ax.tick_params(axis='both', which='minor', length=3, width=1.0)

    # Save figure
    filename = 'q4_fit_all_cases'
    fig.savefig(os.path.join(IMAGES, f'{filename}.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main(show=False)