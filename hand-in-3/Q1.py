
T_calibration = 20 # C
T_actual = 16 # C

L_measured = 99.6 # cm

alpha = 0.0005 # cm/cm/C, expansion coefficient
alpha_uncertainty = 0.0001 # cm/cm/C, expansion coefficient

ruler_marker = 1 # mm

delta_L = L_measured * alpha * (T_actual - T_calibration)
print(delta_L)
print(L_measured + delta_L)

uncertainty_grading = ruler_marker / 10 # cm
uncertainty_alpha = alpha_uncertainty * L_measured * abs(T_actual - T_calibration)
uncertainty_total = uncertainty_grading + uncertainty_alpha

print(uncertainty_grading)
print(uncertainty_alpha)
print(uncertainty_total)