import numpy as np
from cognitive_dynamics import (
    TimelineSpec,
    DomainSpec,
    simulate,
    plot_interactive_run,
    plot_multiverse_4d,
    plot_energy_burnout,
)

timelines = [
    TimelineSpec("Night Owl (Spite)", tau=0.60, phi0=np.pi / 4, v0=0.46, n0=0.55, sigma=0.015, lam=0.04, k=0.12, p_tilde=0.14, is_night_owl=True),
    TimelineSpec("Early Bird (Steady)", tau=0.60, phi0=0.0, v0=0.52, n0=0.55, sigma=0.015, lam=0.01, k=0.08, p_tilde=0.10, is_night_owl=False),
    TimelineSpec("Bricked Chaos", tau=0.55, phi0=np.pi / 6, v0=0.15, n0=0.55, sigma=0.025, lam=0.05, k=0.10, p_tilde=0.15, is_night_owl=True),
]

domains = [
    DomainSpec("Functional Analysis", base_amount=18.0, complexity=1.3, t_exam=18.0, v_offset=0.05),
    DomainSpec("GR Hydrodynamics", base_amount=25.0, complexity=1.8, t_exam=22.0, v_offset=-0.10),
    DomainSpec("Quantum Computing", base_amount=30.0, complexity=2.0, t_exam=26.0, v_offset=0.06),
]

J_timelines = np.array([
    [0.00, 0.03, 0.02],
    [0.03, 0.00, 0.04],
    [0.02, 0.04, 0.00],
])

W_domains = np.array([
    [0.00, 0.15, 0.20],
    [0.75, 0.00, 0.15],
    [0.65, 0.20, 0.00],
])

res = simulate(
    steps=55,
    timelines=timelines,
    domains=domains,
    n_runs=500,
    coupling_J=J_timelines,
    domain_transfer_W=W_domains,
    seed=21,
)

print("Exam Pass Probability Matrix (%):\n", np.round(res["stats"]["pass_matrix"] * 100, 1))
plot_multiverse_4d(res, save_path="./Figures/Multiverse_4D.png")
plot_energy_burnout(res, domain_idx=0, save_path="./Figures/Energy_Burnout.png")
plot_interactive_run(res, run_idx=0, domain_idx=0, interactive=True)