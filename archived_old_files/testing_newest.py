from archived_old_files.complete_solver_wPlots import simulate_multiverse_auto, TimelineSpec, ShockEvent, estimate_domain_bounds, compute_info_volume, plot_multiverse_diagnostics

simulation_results = simulate_multiverse_auto(
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
    ],
)

plot_multiverse_diagnostics(
    simulation_results,
    save_path="./Figures/Multiverse_Diagnostics.png",
)
