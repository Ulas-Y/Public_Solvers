import numpy as np
from archived_old_files.multiverse_4D_Wenergy import DomainSpec, simulate_multiverse_4d_energy , plot_energy_burnout_diagnostics , TimelineSpec, build_graph_laplacian , estimate_domain_bounds, compute_info_volume

laplacian = build_graph_laplacian(np.array([[0, 0.04, 0.02], [0.04, 0, 0.05], [0.02, 0.05, 0]]))  # Test the function

# Rows = target domain receiving insight, Columns = source domain
W_domains = np.array([
    [0.00, 0.15, 0.20],  # Functional Analysis <- mild intuition from GR & Quantum
    [0.75, 0.00, 0.15],  # GR Hydrodynamics    <- heavily unlocked by Functional Analysis
    [0.65, 0.20, 0.00],  # Quantum Computing   <- unlocked by Functional Analysis + tensor/GR math
])

J_timelines = np.array([
    [0.00, 0.03, 0.02],
    [0.03, 0.00, 0.04],
    [0.02, 0.04, 0.00],
])

sim_domains = [
    DomainSpec("Functional Analysis", base_amount=18.0, complexity=1.3, t_exam=9.0, v_offset=0.05),
    DomainSpec("GR Hydrodynamics", base_amount=25.0, complexity=1.8, t_exam=13.0, v_offset=-0.12),
    DomainSpec("Quantum Computing", base_amount=30.0, complexity=2.0, t_exam=15.0, v_offset=0.08),
]



simulation_results = simulate_multiverse_4d_energy(
    steps=50,
    timelines=[
        TimelineSpec(
            name="Timeline A",
            tau=0.5,
            sigma=0.1,
            lam=0.05,
            k=0.1,
            p_tilde=0.2,
            phi0=0.5,
            v0=0.12,
            n0=0.55,
        ),
        TimelineSpec(
            name="Timeline B",
            tau=0.5,
            sigma=0.1,
            lam=0.05,
            k=0.1,
            p_tilde=0.2,
            phi0=0.6,
            v0=0.15,
            n0=0.50,
        ),
        TimelineSpec(
            name="Timeline C",
            tau=0.5,
            sigma=0.1,
            lam=0.05,
            k=0.1,
            p_tilde=0.2,
            phi0=0.4,
            v0=0.10,
            n0=0.60,
        ),
    ],
    coupling_J=J_timelines,
    domains = sim_domains,
    domain_transfer_W = W_domains,
    seed=21
)

plot_energy_burnout_diagnostics(
    simulation_results,
    save_path="./Figures/energy_burnout_diagnostics.png",
)