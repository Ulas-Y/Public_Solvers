import numpy as np
from .specs import TimelineSpec, DomainSpec, ShockEvent
from .physics import compute_info_volume, estimate_domain_bounds, build_graph_laplacian


def simulate(
    steps: int,
    timelines: list[TimelineSpec],
    domains: list[DomainSpec] | None = None,
    n_runs: int = 500,
    shocks: list[ShockEvent] | None = None,
    coupling_J: np.ndarray | None = None,
    domain_transfer_W: np.ndarray | None = None,
    beta_alpha: float = 0.85,
    beta_v: float = 0.25,
    Deadline_Panic: bool = True,
    Boredom_Side_Quest: bool = True,
    Spite_Taunt: bool = True,
    Energy_Burnout: bool = True,
    taunt_rate: float = 0.07,
    panic_window: float = 4.0,
    panic_mastery_floor: float = 0.45,
    boredom_threshold: float = 0.94,
    init_jitter: float = 0.02,
    seed: int | None = 42,
) -> dict:
    """
    Unified 4D Reverse-Spacetime Tensor Solver of shape (steps + 1, R, B, G).
    Supports n_runs=1 (single trajectory / interactive scrubbing) up to arbitrary R Monte Carlo runs.
    """
    if domains is None:
        domains = [DomainSpec("Default Domain")]

    rng = np.random.default_rng(seed)
    R, B, G = n_runs, len(timelines), len(domains)
    jitter_scale = 0.0 if R == 1 else init_jitter

    bounds = np.array([
        estimate_domain_bounds(dom.base_amount, dom.complexity, dom.n_harmonics)
        for dom in domains
    ])
    _, D_vec, alpha0 = bounds[:, 0], bounds[:, 1], bounds[:, 2]

    t_exam_vec = np.array([dom.t_exam for dom in domains], dtype=float)
    u_pass_vec = np.array([dom.u_pass for dom in domains], dtype=float)
    v_off_vec = np.array([dom.v_offset for dom in domains], dtype=float)

    def bcast_tl(attr: str, dtype=float) -> np.ndarray:
        arr = np.array([getattr(tl, attr) for tl in timelines], dtype=dtype)[:, None]
        return np.broadcast_to(arr, (R, B, G)).copy()

    tau = bcast_tl("tau")
    sigma = bcast_tl("sigma")
    lam = bcast_tl("lam")
    k = bcast_tl("k")
    p_tilde = bcast_tl("p_tilde")

    is_night_owl = bcast_tl("is_night_owl", dtype=bool)
    phase_shift = bcast_tl("phase_shift_hours")
    e_base = bcast_tl("e_base")
    k_energy = bcast_tl("k_energy")
    k_fatigue = bcast_tl("k_fatigue")
    k_recovery = bcast_tl("k_recovery")
    t_peak_dur = bcast_tl("t_peak")

    t = np.zeros((steps + 1, R, B, G))
    u = np.zeros((steps + 1, R, B, G))
    xi = np.zeros((steps + 1, R, B, G))
    v = np.zeros((steps + 1, R, B, G))
    n = np.zeros((steps + 1, R, B, G))
    alpha_eff_hist = np.zeros((steps + 1, R, B, G))
    v_eff_hist = np.zeros((steps + 1, R, B, G))

    awake_hist = np.zeros((steps + 1, R, B, G))
    energy_hist = np.zeros((steps + 1, R, B, G))
    fatigue_hist = np.zeros((steps + 1, R, B, G))
    eta_cog_hist = np.zeros((steps + 1, R, B, G))

    xi_init_bg = np.zeros((B, G))
    for b, tl in enumerate(timelines):
        for g, dom in enumerate(domains):
            xi_init_bg[b, g] = (
                compute_info_volume(dom.base_amount, dom.complexity, tl.phi0, dom.n_harmonics)
                / D_vec[g]
            )

    v_init_bg = np.array([tl.v0 for tl in timelines])[:, None] + v_off_vec[None, :]
    n_init_bg = np.broadcast_to(np.array([tl.n0 for tl in timelines])[:, None], (B, G))

    xi[0] = np.clip(xi_init_bg[None, :, :] + rng.normal(0.0, jitter_scale * 0.5, size=(R, B, G)), alpha0, 1.0)
    v[0] = np.clip(v_init_bg[None, :, :] + rng.normal(0.0, jitter_scale, size=(R, B, G)), 0.0, 1.0)
    n[0] = np.clip(n_init_bg[None, :, :] + rng.normal(0.0, jitter_scale, size=(R, B, G)), 0.0, 1.0)

    e_curr = e_base.copy()
    f_curr = np.zeros((R, B, G))
    impulse_i = np.ones((R, B, G))
    peak_timer = np.zeros((R, B, G))

    L_graph = build_graph_laplacian(coupling_J) if coupling_J is not None else None
    W = np.asarray(domain_transfer_W, dtype=float).copy() if domain_transfer_W is not None else None
    if W is not None:
        np.fill_diagonal(W, 0.0)

    shock_map: dict[int, list[ShockEvent]] = {}
    for sh in (shocks or []):
        shock_map.setdefault(sh.step, []).append(sh)

    fired_panic = np.zeros((R, B, G), dtype=bool)
    fired_boredom = np.zeros((R, B, G), dtype=bool)
    fired_spite = np.zeros((R, B, G), dtype=bool)
    run0_markers: list[tuple[int, int, int, str]] = []

    for s in range(steps):
        u_s, xi_s, v_s, n_s, t_s = u[s], xi[s], v[s], n[s], t[s]

        # 0A. Optional manual shocks
        if s in shock_map:
            for sh in shock_map[s]:
                b = sh.branch
                g_slice = slice(None) if sh.domain is None else sh.domain
                v_s[:, b, g_slice] = np.clip(v_s[:, b, g_slice] + sh.dv, 0.0, 1.0)
                xi_s[:, b, g_slice] = np.clip(xi_s[:, b, g_slice] + sh.dxi, alpha0[g_slice], 1.0)
                n_s[:, b, g_slice] = np.clip(n_s[:, b, g_slice] + sh.dn, 0.0, 1.0)
                e_curr[:, b, g_slice] = np.clip(e_curr[:, b, g_slice] + sh.de, 0.0, 1.0)
                if sh.new_regime is not None:
                    nr = sh.new_regime
                    tau[:, b, g_slice] = nr.tau
                    sigma[:, b, g_slice] = nr.sigma
                    lam[:, b, g_slice] = nr.lam
                    k[:, b, g_slice] = nr.k
                    p_tilde[:, b, g_slice] = nr.p_tilde
                g_idx = 0 if sh.domain is None else sh.domain
                run0_markers.append((s, b, g_idx, "Manual"))

        local_m = u_s / xi_s

        # 0B. Circadian awakeness wave (24h period -> pi radians inside sin^2 / cos^2)
        theta = (np.pi / 24.0) * (t_s - phase_shift)
        awake = np.where(is_night_owl, np.sin(theta) ** 2, np.cos(theta) ** 2)

        # 1. Boredom_Side_Quest
        if Boredom_Side_Quest:
            m_bored = (~fired_boredom) & (local_m >= boredom_threshold) & (xi_s < 0.90)
            if np.any(m_bored):
                for b in range(B):
                    for g in range(G):
                        if m_bored[0, b, g]:
                            run0_markers.append((s, b, g, "Boredom"))
                fired_boredom |= m_bored
                xi_s = np.where(m_bored, np.clip(xi_s + 0.18, alpha0, 1.0), xi_s)
                lam = np.where(m_bored, np.maximum(lam, 0.11), lam)
                k = np.where(m_bored, k * 1.35, k)
                p_tilde = np.where(m_bored, p_tilde * 1.35, p_tilde)
                e_curr = np.where(m_bored, np.maximum(e_curr, 0.70), e_curr)
                impulse_i = np.where(m_bored, np.maximum(impulse_i, 1.5), impulse_i)
                peak_timer = np.where(m_bored, np.maximum(peak_timer, 0.6 * t_peak_dur), peak_timer)

        # 2. Deadline_Panic
        if Deadline_Panic:
            time_left = t_exam_vec[None, None, :] - t_s
            m_panic = (
                (~fired_panic)
                & (time_left > 0.0)
                & (time_left <= panic_window)
                & (u_s < panic_mastery_floor)
            )
            if np.any(m_panic):
                for b in range(B):
                    for g in range(G):
                        if m_panic[0, b, g]:
                            run0_markers.append((s, b, g, "Panic"))
                fired_panic |= m_panic
                v_s = np.where(m_panic, np.clip(v_s + 0.16, 0.0, 1.0), v_s)
                xi_s = np.where(m_panic, np.clip(xi_s - 0.20, alpha0, 1.0), xi_s)
                lam = np.where(m_panic, -0.06, lam)
                tau = np.where(m_panic, np.maximum(tau, 0.75), tau)
                sigma = np.where(m_panic, sigma * 0.25, sigma)
                e_curr = np.where(m_panic, 1.0, e_curr)
                impulse_i = np.where(m_panic, np.maximum(impulse_i, 2.2), impulse_i)
                peak_timer = np.where(m_panic, t_peak_dur, peak_timer)

        # 3. Stochastic Spite_Taunt
        if Spite_Taunt:
            p_hit = 1.0 - np.exp(-taunt_rate * tau)
            m_spite = (~fired_spite) & (rng.random((R, B, G)) < p_hit)
            if np.any(m_spite):
                for b in range(B):
                    for g in range(G):
                        if m_spite[0, b, g]:
                            run0_markers.append((s, b, g, "Spite"))
                fired_spite |= m_spite
                v_s = np.where(m_spite, np.clip(v_s + 0.22, 0.0, 1.0), v_s)
                n_s = np.where(m_spite, np.clip(n_s + 0.15, 0.0, 1.0), n_s)
                tau = np.where(m_spite, np.maximum(tau, 0.85), tau)
                sigma = np.where(m_spite, sigma * 0.15, sigma)
                lam = np.where(m_spite, np.maximum(lam, 0.10), lam)
                k = np.where(m_spite, k * 1.50, k)
                p_tilde = np.where(m_spite, p_tilde * 1.50, p_tilde)
                e_curr = np.where(m_spite, 1.0, e_curr)
                impulse_i = np.where(m_spite, np.maximum(impulse_i, 3.0), impulse_i)
                peak_timer = np.where(m_spite, 1.25 * t_peak_dur, peak_timer)

        xi[s], v[s], n[s] = xi_s, v_s, n_s

        # 4. Cognitive Energy & Burnout Multiplier
        eta_cog = awake * np.maximum(0.0, e_curr - f_curr) if Energy_Burnout else np.ones((R, B, G))
        awake_hist[s] = awake
        energy_hist[s] = e_curr
        fatigue_hist[s] = f_curr
        eta_cog_hist[s] = eta_cog

        # 5. Cross-Domain Prerequisite Transfer (Axis G)
        if W is not None:
            prereq_signal = u_s @ W.T
            alpha_eff = alpha0[None, None, :] * np.exp(-beta_alpha * prereq_signal)
            v_eff = np.minimum(1.0, v_s + beta_v * prereq_signal)
        else:
            alpha_eff = np.broadcast_to(alpha0, (R, B, G))
            v_eff = v_s

        alpha_eff_hist[s] = alpha_eff
        v_eff_hist[s] = v_eff

        # 6. Ingestion + Graph Laplacian Cross-Timeline Diffusion (Axis B)
        raw_gain = np.where(
            v_eff >= alpha_eff,
            np.minimum(v_eff, xi_s) * n_s * np.maximum(0.0, 1.0 - u_s / xi_s),
            0.0,
        )
        u_gain = raw_gain * eta_cog
        u_tilde = (u_s + u_gain) * np.exp(-sigma * tau)

        if L_graph is not None:
            diffusion = np.einsum("bc,rcg->rbg", L_graph, u_tilde)
            u_next = np.clip(u_tilde - diffusion * tau, 0.0, 1.0)
        else:
            u_next = u_tilde

        local_mastery_next = u_next / xi_s

        # 7. Meta-learning & scope updates
        t[s + 1] = t_s + tau
        u[s + 1] = u_next
        n[s + 1] = np.minimum(1.0, n_s * np.exp(k * local_mastery_next * eta_cog * tau))
        v[s + 1] = np.minimum(1.0, v_s * np.exp(p_tilde * local_mastery_next * eta_cog * tau))
        xi[s + 1] = np.clip(xi_s * np.exp(lam * tau), alpha0, 1.0)

        # 8. Advance Fatigue & Energy Reserve
        if Energy_Burnout:
            df_build = np.maximum(0.0, awake - 0.5) * e_curr * k_fatigue * tau
            df_recov = np.maximum(0.0, 0.5 - awake) * k_recovery * tau
            f_curr = np.clip(f_curr + df_build - df_recov, 0.0, 1.0)

            holding = peak_timer > 0.0
            e_decayed = e_base + (e_curr - e_base) * np.exp(-(k_energy / impulse_i) * tau)
            e_curr = np.where(holding, e_curr, e_decayed)
            peak_timer = np.maximum(0.0, peak_timer - tau)

    theta_end = (np.pi / 24.0) * (t[-1] - phase_shift)
    awake_hist[-1] = np.where(is_night_owl, np.sin(theta_end) ** 2, np.cos(theta_end) ** 2)
    energy_hist[-1] = e_curr
    fatigue_hist[-1] = f_curr
    eta_cog_hist[-1] = awake_hist[-1] * np.maximum(0.0, e_curr - f_curr) if Energy_Burnout else 1.0

    if W is not None:
        prereq_end = u[-1] @ W.T
        alpha_eff_hist[-1] = alpha0[None, None, :] * np.exp(-beta_alpha * prereq_end)
        v_eff_hist[-1] = np.minimum(1.0, v[-1] + beta_v * prereq_end)
    else:
        alpha_eff_hist[-1] = np.broadcast_to(alpha0, (R, B, G))
        v_eff_hist[-1] = v[-1]

    exam_u = np.array([
        [[np.interp(t_exam_vec[g], t[:, r, b, g], u[:, r, b, g]) for g in range(G)] for b in range(B)]
        for r in range(R)
    ])

    return {
        "timeline_names": [tl.name for tl in timelines],
        "domain_names": [dom.name for dom in domains],
        "bounds": bounds,
        "t_exam": t_exam_vec,
        "u_pass": u_pass_vec,
        "t": t,
        "u": u,
        "xi": xi,
        "v": v,
        "v_eff": v_eff_hist,
        "n": n,
        "alpha_eff": alpha_eff_hist,
        "awake": awake_hist,
        "energy": energy_hist,
        "fatigue": fatigue_hist,
        "eta_cog": eta_cog_hist,
        "run0_markers": run0_markers,
        "stats": {
            "pass_matrix": np.mean(exam_u >= u_pass_vec[None, None, :], axis=0),
            "sat_matrix": np.mean(u[-1] >= 0.95, axis=0),
            "panic_matrix": np.mean(fired_panic, axis=0),
            "boredom_matrix": np.mean(fired_boredom, axis=0),
            "spite_matrix": np.mean(fired_spite, axis=0),
            "burnout_matrix": np.mean(np.any((energy_hist - fatigue_hist) <= 0.0, axis=0), axis=0),
        },
    }