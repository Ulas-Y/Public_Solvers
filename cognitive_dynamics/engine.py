import numpy as np
from .specs import TimelineSpec, DomainSpec, ShockEvent
from .physics import compute_info_volume, estimate_domain_bounds, build_graph_laplacian
from .backend import TensorBackend


def simulate(
    steps: int,
    timelines: list[TimelineSpec],
    domains: list[DomainSpec] | None = None,
    n_runs: int = 500,
    backend: str = "auto",
    device: str | None = None,
    dtype: str | None = None,
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
    Supports backend='auto' | 'torch' | 'numpy' and auto-falls back if PyTorch/CUDA is absent.
    """
    if domains is None:
        domains = [DomainSpec("Default Domain")]

    ops = TensorBackend(backend=backend, device=device, seed=seed, dtype=dtype)
    R, B, G = n_runs, len(timelines), len(domains)
    jitter_scale = 0.0 if R == 1 else init_jitter

    bounds_np = np.array([
        estimate_domain_bounds(dom.base_amount, dom.complexity, dom.n_harmonics)
        for dom in domains
    ])
    _, D_vec_np, alpha0_np = bounds_np[:, 0], bounds_np[:, 1], bounds_np[:, 2]

    alpha0 = ops.as_tensor(alpha0_np)
    t_exam_vec = ops.as_tensor([dom.t_exam for dom in domains])
    u_pass_vec = ops.as_tensor([dom.u_pass for dom in domains])
    v_off_np = np.array([dom.v_offset for dom in domains], dtype=float)

    tau = ops.broadcast_timeline([tl.tau for tl in timelines], R, B, G)
    sigma = ops.broadcast_timeline([tl.sigma for tl in timelines], R, B, G)
    lam = ops.broadcast_timeline([tl.lam for tl in timelines], R, B, G)
    k = ops.broadcast_timeline([tl.k for tl in timelines], R, B, G)
    p_tilde = ops.broadcast_timeline([tl.p_tilde for tl in timelines], R, B, G)

    is_night_owl = ops.broadcast_timeline([tl.is_night_owl for tl in timelines], R, B, G, is_bool=True)
    phase_shift = ops.broadcast_timeline([tl.phase_shift_hours for tl in timelines], R, B, G)
    e_base = ops.broadcast_timeline([tl.e_base for tl in timelines], R, B, G)
    k_energy = ops.broadcast_timeline([tl.k_energy for tl in timelines], R, B, G)
    k_fatigue = ops.broadcast_timeline([tl.k_fatigue for tl in timelines], R, B, G)
    k_recovery = ops.broadcast_timeline([tl.k_recovery for tl in timelines], R, B, G)
    t_peak_dur = ops.broadcast_timeline([tl.t_peak for tl in timelines], R, B, G)

    t = ops.zeros((steps + 1, R, B, G))
    u = ops.zeros((steps + 1, R, B, G))
    xi = ops.zeros((steps + 1, R, B, G))
    v = ops.zeros((steps + 1, R, B, G))
    n = ops.zeros((steps + 1, R, B, G))
    alpha_eff_hist = ops.zeros((steps + 1, R, B, G))
    v_eff_hist = ops.zeros((steps + 1, R, B, G))

    awake_hist = ops.zeros((steps + 1, R, B, G))
    energy_hist = ops.zeros((steps + 1, R, B, G))
    fatigue_hist = ops.zeros((steps + 1, R, B, G))
    eta_cog_hist = ops.zeros((steps + 1, R, B, G))

    xi_init_bg = np.zeros((B, G))
    for b, tl in enumerate(timelines):
        for g, dom in enumerate(domains):
            xi_init_bg[b, g] = (
                compute_info_volume(dom.base_amount, dom.complexity, tl.phi0, dom.n_harmonics)
                / D_vec_np[g]
            )

    v_init_bg = np.array([tl.v0 for tl in timelines])[:, None] + v_off_np[None, :]
    n_init_bg = np.broadcast_to(np.array([tl.n0 for tl in timelines])[:, None], (B, G))

    xi[0] = ops.clamp(ops.as_tensor(xi_init_bg)[None, :, :] + ops.randn((R, B, G)) * (jitter_scale * 0.5), alpha0, 1.0)
    v[0] = ops.clamp(ops.as_tensor(v_init_bg)[None, :, :] + ops.randn((R, B, G)) * jitter_scale, 0.0, 1.0)
    n[0] = ops.clamp(ops.as_tensor(n_init_bg)[None, :, :] + ops.randn((R, B, G)) * jitter_scale, 0.0, 1.0)

    e_curr = e_base.copy() if not ops.is_torch else e_base.clone()
    f_curr = ops.zeros((R, B, G))
    impulse_i = ops.zeros((R, B, G)) + 1.0
    peak_timer = ops.zeros((R, B, G))

    L_graph = ops.as_tensor(build_graph_laplacian(coupling_J)) if coupling_J is not None else None
    if domain_transfer_W is not None:
        W_np = np.asarray(domain_transfer_W, dtype=float).copy()
        np.fill_diagonal(W_np, 0.0)
        W = ops.as_tensor(W_np)
    else:
        W = None

    shock_map: dict[int, list[ShockEvent]] = {}
    for sh in (shocks or []):
        shock_map.setdefault(sh.step, []).append(sh)

    fired_panic = ops.zeros((R, B, G), is_bool=True)
    fired_boredom = ops.zeros((R, B, G), is_bool=True)
    fired_spite = ops.zeros((R, B, G), is_bool=True)
    run0_markers: list[tuple[int, int, int, str]] = []

    pi_over_24 = float(np.pi / 24.0)

    for s in range(steps):
        u_s, xi_s, v_s, n_s, t_s = u[s], xi[s], v[s], n[s], t[s]

        # 0A. Optional manual shocks
        if s in shock_map:
            for sh in shock_map[s]:
                b = sh.branch
                g_slice = slice(None) if sh.domain is None else sh.domain
                v_s[:, b, g_slice] = ops.clamp(v_s[:, b, g_slice] + sh.dv, 0.0, 1.0)
                xi_s[:, b, g_slice] = ops.clamp(xi_s[:, b, g_slice] + sh.dxi, alpha0[g_slice], 1.0)
                n_s[:, b, g_slice] = ops.clamp(n_s[:, b, g_slice] + sh.dn, 0.0, 1.0)
                e_curr[:, b, g_slice] = ops.clamp(e_curr[:, b, g_slice] + sh.de, 0.0, 1.0)
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

        # 0B. Circadian awakeness wave
        theta = pi_over_24 * (t_s - phase_shift)
        awake = ops.where(is_night_owl, ops.sin(theta) ** 2, ops.cos(theta) ** 2)

        # 1. Boredom_Side_Quest
        if Boredom_Side_Quest:
            m_bored = (~fired_boredom) & (local_m >= boredom_threshold) & (xi_s < 0.90)
            if ops.any(m_bored):
                m0 = ops.to_numpy(m_bored[0])
                for b in range(B):
                    for g in range(G):
                        if m0[b, g]:
                            run0_markers.append((s, b, g, "Boredom"))
                fired_boredom |= m_bored
                xi_s = ops.where(m_bored, ops.clamp(xi_s + 0.18, alpha0, 1.0), xi_s)
                lam = ops.where(m_bored, ops.maximum(lam, 0.11), lam)
                k = ops.where(m_bored, k * 1.35, k)
                p_tilde = ops.where(m_bored, p_tilde * 1.35, p_tilde)
                e_curr = ops.where(m_bored, ops.maximum(e_curr, 0.70), e_curr)
                impulse_i = ops.where(m_bored, ops.maximum(impulse_i, 1.5), impulse_i)
                peak_timer = ops.where(m_bored, ops.maximum(peak_timer, 0.6 * t_peak_dur), peak_timer)

        # 2. Deadline_Panic
        if Deadline_Panic:
            time_left = t_exam_vec[None, None, :] - t_s
            m_panic = (
                (~fired_panic)
                & (time_left > 0.0)
                & (time_left <= panic_window)
                & (u_s < panic_mastery_floor)
            )
            if ops.any(m_panic):
                m0 = ops.to_numpy(m_panic[0])
                for b in range(B):
                    for g in range(G):
                        if m0[b, g]:
                            run0_markers.append((s, b, g, "Panic"))
                fired_panic |= m_panic
                v_s = ops.where(m_panic, ops.clamp(v_s + 0.16, 0.0, 1.0), v_s)
                xi_s = ops.where(m_panic, ops.clamp(xi_s - 0.20, alpha0, 1.0), xi_s)
                lam = ops.where(m_panic, -0.06, lam)
                tau = ops.where(m_panic, ops.maximum(tau, 0.75), tau)
                sigma = ops.where(m_panic, sigma * 0.25, sigma)
                e_curr = ops.where(m_panic, 1.0, e_curr)
                impulse_i = ops.where(m_panic, ops.maximum(impulse_i, 2.2), impulse_i)
                peak_timer = ops.where(m_panic, t_peak_dur, peak_timer)

        # 3. Stochastic Spite_Taunt
        if Spite_Taunt:
            p_hit = 1.0 - ops.exp(-taunt_rate * tau)
            m_spite = (~fired_spite) & (ops.rand((R, B, G)) < p_hit)
            if ops.any(m_spite):
                m0 = ops.to_numpy(m_spite[0])
                for b in range(B):
                    for g in range(G):
                        if m0[b, g]:
                            run0_markers.append((s, b, g, "Spite"))
                fired_spite |= m_spite
                v_s = ops.where(m_spite, ops.clamp(v_s + 0.22, 0.0, 1.0), v_s)
                n_s = ops.where(m_spite, ops.clamp(n_s + 0.15, 0.0, 1.0), n_s)
                tau = ops.where(m_spite, ops.maximum(tau, 0.85), tau)
                sigma = ops.where(m_spite, sigma * 0.15, sigma)
                lam = ops.where(m_spite, ops.maximum(lam, 0.10), lam)
                k = ops.where(m_spite, k * 1.50, k)
                p_tilde = ops.where(m_spite, p_tilde * 1.50, p_tilde)
                e_curr = ops.where(m_spite, 1.0, e_curr)
                impulse_i = ops.where(m_spite, ops.maximum(impulse_i, 3.0), impulse_i)
                peak_timer = ops.where(m_spite, 1.25 * t_peak_dur, peak_timer)

        xi[s], v[s], n[s] = xi_s, v_s, n_s

        # 4. Cognitive Energy & Burnout Multiplier
        eta_cog = awake * ops.maximum(0.0, e_curr - f_curr) if Energy_Burnout else (ops.zeros((R, B, G)) + 1.0)
        awake_hist[s] = awake
        energy_hist[s] = e_curr
        fatigue_hist[s] = f_curr
        eta_cog_hist[s] = eta_cog

        # 5. Cross-Domain Prerequisite Transfer (Axis G)
        if W is not None:
            prereq_signal = u_s @ W.T
            alpha_eff = alpha0[None, None, :] * ops.exp(-beta_alpha * prereq_signal)
            v_eff = ops.minimum(1.0, v_s + beta_v * prereq_signal)
        else:
            alpha_eff = alpha0[None, None, :] + ops.zeros((R, B, G))
            v_eff = v_s

        alpha_eff_hist[s] = alpha_eff
        v_eff_hist[s] = v_eff

        # 6. Ingestion + Graph Laplacian Cross-Timeline Diffusion (Axis B)
        raw_gain = ops.where(
            v_eff >= alpha_eff,
            ops.minimum(v_eff, xi_s) * n_s * ops.maximum(0.0, 1.0 - u_s / xi_s),
            0.0,
        )
        u_gain = raw_gain * eta_cog
        u_tilde = (u_s + u_gain) * ops.exp(-sigma * tau)

        if L_graph is not None:
            diffusion = ops.einsum("bc,rcg->rbg", L_graph, u_tilde)
            u_next = ops.clamp(u_tilde - diffusion * tau, 0.0, 1.0)
        else:
            u_next = u_tilde

        local_mastery_next = u_next / xi_s

        # 7. Meta-learning & scope updates
        t[s + 1] = t_s + tau
        u[s + 1] = u_next
        n[s + 1] = ops.minimum(1.0, n_s * ops.exp(k * local_mastery_next * eta_cog * tau))
        v[s + 1] = ops.minimum(1.0, v_s * ops.exp(p_tilde * local_mastery_next * eta_cog * tau))
        xi[s + 1] = ops.clamp(xi_s * ops.exp(lam * tau), alpha0, 1.0)

        # 8. Advance Fatigue & Energy Reserve
        if Energy_Burnout:
            df_build = ops.maximum(0.0, awake - 0.5) * e_curr * k_fatigue * tau
            df_recov = ops.maximum(0.0, 0.5 - awake) * k_recovery * tau
            f_curr = ops.clamp(f_curr + df_build - df_recov, 0.0, 1.0)

            holding = peak_timer > 0.0
            e_decayed = e_base + (e_curr - e_base) * ops.exp(-(k_energy / impulse_i) * tau)
            e_curr = ops.where(holding, e_curr, e_decayed)
            peak_timer = ops.maximum(0.0, peak_timer - tau)

    theta_end = pi_over_24 * (t[-1] - phase_shift)
    awake_hist[-1] = ops.where(is_night_owl, ops.sin(theta_end) ** 2, ops.cos(theta_end) ** 2)
    energy_hist[-1] = e_curr
    fatigue_hist[-1] = f_curr
    eta_cog_hist[-1] = awake_hist[-1] * ops.maximum(0.0, e_curr - f_curr) if Energy_Burnout else (ops.zeros((R, B, G)) + 1.0)

    if W is not None:
        prereq_end = u[-1] @ W.T
        alpha_eff_hist[-1] = alpha0[None, None, :] * ops.exp(-beta_alpha * prereq_end)
        v_eff_hist[-1] = ops.minimum(1.0, v[-1] + beta_v * prereq_end)
    else:
        alpha_eff_hist[-1] = alpha0[None, None, :] + ops.zeros((R, B, G))
        v_eff_hist[-1] = v[-1]

    exam_u_np = ops.to_numpy(ops.interp_exam(t_exam_vec, t, u))
    u_final_np = ops.to_numpy(u[-1])
    u_pass_np = ops.to_numpy(u_pass_vec)
    net_energy_np = ops.to_numpy(energy_hist - fatigue_hist)

    return {
        "backend": ops.name,
        "device": ops.device,
        "timeline_names": [tl.name for tl in timelines],
        "domain_names": [dom.name for dom in domains],
        "bounds": bounds_np,
        "t_exam": ops.to_numpy(t_exam_vec),
        "u_pass": u_pass_np,
        "t": ops.to_numpy(t),
        "u": ops.to_numpy(u),
        "xi": ops.to_numpy(xi),
        "v": ops.to_numpy(v),
        "v_eff": ops.to_numpy(v_eff_hist),
        "n": ops.to_numpy(n),
        "alpha_eff": ops.to_numpy(alpha_eff_hist),
        "awake": ops.to_numpy(awake_hist),
        "energy": ops.to_numpy(energy_hist),
        "fatigue": ops.to_numpy(fatigue_hist),
        "eta_cog": ops.to_numpy(eta_cog_hist),
        "run0_markers": run0_markers,
        "stats": {
            "pass_matrix": np.mean(exam_u_np >= u_pass_np[None, None, :], axis=0),
            "sat_matrix": np.mean(u_final_np >= 0.95, axis=0),
            "panic_matrix": np.mean(ops.to_numpy(fired_panic), axis=0),
            "boredom_matrix": np.mean(ops.to_numpy(fired_boredom), axis=0),
            "spite_matrix": np.mean(ops.to_numpy(fired_spite), axis=0),
            "burnout_matrix": np.mean(np.any(net_energy_np <= 0.0, axis=0), axis=0),
        },
    }