import numpy as np
from archived_old_files.vectorized_solver import build_graph_laplacian, plot_monte_carlo_ensemble, simulate_multiverse_mc
from archived_old_files.timeline_solver import TimelineSpec

laplacian = build_graph_laplacian(np.array([[0, 0.04, 0.02], [0.04, 0, 0.05], [0.02, 0.05, 0]]))  # Test the function

resulting_multiverse = simulate_multiverse_mc(
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
    coupling_J=laplacian,
    seed=21
)

plot_monte_carlo_ensemble(
    resulting_multiverse,
    save_path="./Figures/Monte_Carlo_Ensemble.png"
)