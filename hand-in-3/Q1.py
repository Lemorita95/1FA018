
T_calibration = 20 # C
T_actual = 16 # C

L_measured = 99.6 # cm

alpha = 0.0005 # cm/cm/C, expansion coefficient
uncertainty_alpha = 0.0001 # cm/cm/C, expansion coefficient

ruler_marker = 1 # mm

delta_L = L_measured * alpha * (T_actual - T_calibration)
print()
print(f'delta_L: {delta_L}')
print(f'L_prime: L_measured + delta_L')

uncertainty_L = ruler_marker / 10 / 2 # cm

partial_L = 1 + alpha*(T_actual-T_calibration)
partial_alpha = (T_actual-T_calibration) * L_measured

print()
print(f'sigma_L: {uncertainty_L}')
print(f'sigma_alpha: {uncertainty_alpha}')
print(f'partial_L: {partial_L}')
print(f'partial_alpha: {partial_alpha}')

uncertainty_total = partial_L * uncertainty_L + partial_alpha * uncertainty_alpha
uncertainty_total_uncorr = (partial_L**2 * uncertainty_L**2 + partial_alpha**2 * uncertainty_alpha**2)**0.5

print()
print(f'total uncertainty: {uncertainty_total}')
print(f'total uncertainty uncorrelated: {uncertainty_total_uncorr}')