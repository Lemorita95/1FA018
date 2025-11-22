from helpers import DATA, IMAGES, os, plt, np, pd

np.random.seed(42)

# define path for data file
file = os.path.join(DATA, 'data_q3.csv')

# read data file as a pandas dataframe
data = pd.read_csv(file)

'''
    3.a) error propagation
        - assume measurements x, y, z are independent > V() = diagonal
        p2 = decay, p1 = production
        V0 = p2 - p1
        dV0/dp1 = -1 for x=x
        dV0/dp2 = 1
        dV0/dp1 = -1 for x<>x
        J = 
            [dV0_x/dp1_x dV0_x/dp1_y dV0_x/dp1_z dV0_x/dp2_x dV0_x/dp2_y dV0_x/dp2_z]
            [dV0_y/dp1_x dV0_y/dp1_y dV0_y/dp1_z dV0_y/dp2_x dV0_y/dp2_y dV0_y/dp2_z]
            [dV0_z/dp1_x dV0_z/dp1_y dV0_z/dp1_z dV0_z/dp2_x dV0_z/dp2_y dV0_z/dp2_z]

        =
            [-1  0  0  1  0  0]
            [ 0 -1  0  0  1  0]
            [ 0  0 -1  0  0  1]
'''

production_means = np.array([x[1] for x in data.values if x[0] in ['x', 'y', 'z']])
production_sigmas = np.array([x[1] for x in data.values if x[0] not in ['x', 'y', 'z']])
decay_means = np.array([x[2] for x in data.values if x[0] in ['x', 'y', 'z']])
decay_sigmas = np.array([x[2] for x in data.values if x[0] not in ['x', 'y', 'z']])

p1_std = production_sigmas**2
p2_std = decay_sigmas**2
V_x = np.diag(np.concat((p1_std, p2_std), axis=0))

J = np.concat((-np.eye(len(production_sigmas)), np.eye(len(decay_sigmas))), axis=1)

V0 = J @ V_x @ J.T
sigma_V0 = V0**0.5

mean_V0 = decay_means - production_means
std_V0 = np.diagonal(sigma_V0)

for i, v in enumerate(['x', 'y', 'z']):
    print(f"{v}: {mean_V0[i]:.2f} +- {std_V0[i]:.2f}") # 1 significant figure due to uncertainty

# in spherical

def cartesian_to_spherical(x, y, z):
    r = (x**2 + y**2 + z**2)**0.5
    rho = (x**2 + y**2)**0.5
    theta = np.degrees(np.arctan2(y, x))
    phi = np.degrees(np.arctan2(rho, z))

    return r, theta, phi, rho

def jacobian(x, y, z):
    '''
    Jacobian matrix for
        r2 = x2 + y2 + z2
        theta = arctan(y/x)
        phi = arctan(sqrt(x2+y2)/z)

    '''
    r, _, _, rho = cartesian_to_spherical(x, y, z)

    r = (x**2 + y**2 + z**2)**0.5
    rho = (y**2 + x**2)**0.5

    A = x / r
    B = y / r
    C = z / r
    D = -y / rho**2
    E = x / rho**2
    F = 0
    G = x * z / ((r**2) * (rho))
    H = y * z / ((r**2) * (rho))
    I = - (rho) / (r**2)

    r_array = np.array([A, B, C])
    theta_array = np.array([D, E, F])
    phi_array = np.array([G, H, I])

    return np.vstack([r_array, theta_array, phi_array])

J = jacobian(*tuple(mean_V0))

V0 = J @ np.diag(std_V0**2) @ J.T

r, theta, phi, rho = cartesian_to_spherical(*mean_V0)

print(f"r={r:.2f}, θ={theta:.2f}°, ϕ={phi:.2f}°")
print(f"uncertainties:")
print(f"\tr\tθ\tϕ")
for i, v in enumerate(["r", "θ", "ϕ"]):
    print(f"{v}   ", end='')
    print({", ".join(f"{x:.1g}" for x in V0[i])})