import numpy as np
from information_volume_computing import compute_info_volume
from information_volume_computing import estimate_domain_bounds
from non_normalized_learning_solver import simulate_non_normalized
from normalized_learning_solver import simulate_normalized
from automated_multiverse_solver import simulate_multiverse_auto
from information_volume_computing import compute_info_volume
from timeline_solver import TimelineSpec
from shock_event_handler import ShockEvent

M, D, alpha = estimate_domain_bounds(base_amount=20.0, complexity=1.5, n_harmonics=3)
I0 = compute_info_volume(base_amount=20.0, complexity=1.5, phi=np.pi / 4, n_harmonics=3)

# Plug straight into the solvers from earlier:
res = simulate_non_normalized(steps=50, I0=I0, M=M, D=D)
res_norm = simulate_normalized(steps=50, xi0=I0 / D, alpha=alpha)
sim_res = simulate_multiverse_auto(
    steps=50,
    timelines=[
        TimelineSpec(
            name="Timeline A",
            tau=0.5,
            phi0=np.pi / 4,
            v0=0.5,
            n0=0.5,
            sigma=0.1,
            lam=0.05,
            k=0.02,
            p_tilde=0.01,
        ),
        TimelineSpec(
            name="Timeline B",
            tau=0.6,
            phi0=np.pi / 3,
            v0=0.6,
            n0=0.6,
            sigma=0.15,
            lam=0.07,
            k=0.03,
            p_tilde=0.02,
        ),
    ],
    shocks=[
        ShockEvent(step=10, branch=0, dv=0.1, dxi=0.05, dn=0.02, label="Shock A"),
        ShockEvent(step=20, branch=1, dv=-0.05, dxi=-0.02, dn=-0.01, label="Shock B"),
    ],
)

print(res)
print(res_norm)
print(sim_res)

json_output = {
    "non_normalized": {k: v.tolist() for k, v in res.items()},
    "normalized": {k: v.tolist() for k, v in res_norm.items()},
    "multiverse": {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in sim_res.items()
    },
}