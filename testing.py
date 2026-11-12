import numpy as np
from information_volume_computing import compute_info_volume
from information_volume_computing import estimate_domain_bounds
from non_normalized_learning_solver import simulate_non_normalized
from normalized_learning_solver import simulate_normalized

M, D, alpha = estimate_domain_bounds(base_amount=20.0, complexity=1.5, n_harmonics=3)
I0 = compute_info_volume(base_amount=20.0, complexity=1.5, phi=np.pi / 4, n_harmonics=3)

# Plug straight into the solvers from earlier:
res = simulate_non_normalized(steps=50, I0=I0, M=M, D=D)
res_norm = simulate_normalized(steps=50, xi0=I0 / D, alpha=alpha)

print(res)
print(res_norm)