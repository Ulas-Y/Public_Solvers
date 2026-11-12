from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


def compute_info_volume(
    base_amount: float,
    complexity: float,
    phi: float,
    n_harmonics: int = 3,
    a_coeffs: tuple[float, ...] | None = None,
    b_coeffs: tuple[float, ...] | None = None,
    decay_power: float = 1.0,
) -> float:
    if n_harmonics > 3 and decay_power <= 0.0:
        raise ValueError("ZIP BOMB DETECTED: Unbounded harmonics without spectral decay.")
    if a_coeffs is None:
        a_coeffs = tuple(0.40 / (m**decay_power) for m in range(1, n_harmonics + 1))
    if b_coeffs is None:
        b_coeffs = tuple(0.25 / (m**decay_power) for m in range(1, n_harmonics + 1))

    harmonic_sum = 1.0
    for m in range(1, n_harmonics + 1):
        a_m = a_coeffs[m - 1] if m - 1 < len(a_coeffs) else 0.0
        b_m = b_coeffs[m - 1] if m - 1 < len(b_coeffs) else 0.0
        harmonic_sum += a_m * (1.0 - np.cos(m * phi)) + b_m * np.sin(m * phi)

    return float(base_amount * complexity * max(1e-6, harmonic_sum))


def estimate_domain_bounds(
    base_amount: float,
    complexity: float,
    n_harmonics: int = 3,
    **kwargs,
) -> tuple[float, float, float]:
    phis = np.linspace(0.0, np.pi, 500)
    vols = [compute_info_volume(base_amount, complexity, p, n_harmonics, **kwargs) for p in phis]
    M = float(min(vols))
    D = float(max(vols))
    return M, D, M / D


@dataclass
class TimelineSpec:
    name: str
    tau: float
    phi0: float
    v0: float
    n0: float
    sigma: float
    lam: float
    k: float
    p_tilde: float
    is_night_owl: bool = True
    phase_shift_hours: float = 8.0   # Early bird peaks at 08:00 (cos^2), Night owl at 20:00 (sin^2)
    e_base: float = 0.25             # Baseline cognitive energy reserve (0..1)
    k_energy: float = 0.45           # Post-spike decay constant
    k_fatigue: float = 0.32          # Burnout buildup rate when awake > 0.5 & energy is high
    k_recovery: float = 0.22         # Fatigue recovery rate during circadian trough (awake < 0.5)
    t_peak: float = 2.5              # Duration (hours) an adrenaline/spite spike holds at peak


@dataclass
class DomainSpec:
    name: str
    base_amount: float = 20.0
    complexity: float = 1.5
    n_harmonics: int = 3
    t_exam: float = 18.0
    u_pass: float = 0.60
    v_offset: float = 0.0


def build_graph_laplacian(coupling_J: np.ndarray) -> np.ndarray:
    M = np.asarray(coupling_J, dtype=float).copy()
    off_diag = M - np.diag(np.diag(M))
    if np.all(np.diag(M) >= 0.0) and np.all(off_diag <= 0.0) and np.allclose(np.sum(M, axis=1), 0.0):
        return M
    np.fill_diagonal(M, 0.0)
    D_deg = np.diag(np.sum(M, axis=1))
    return D_deg - M


def simulate_multiverse_4d_energy(
    steps: int,
    timelines: list[TimelineSpec],
    domains: list[DomainSpec],
    n_runs: int = 500,
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
    4D Reverse-Spacetime Solver (steps + 1, R, B, G) with Circadian Awakeness,
    Adrenaline/Spite Energy Spikes (t_peak hold + 1/i decay), and Post-Spike Burnout.
    """
    rng = np.random.default_rng(seed)
    R, B, G = n_runs, len(timelines), len(domains)

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

    xi[0] = np.clip(xi_init_bg[None, :, :] + rng.normal(0.0, init_jitter * 0.5, size=(R, B, G)), alpha0, 1.0)
    v[0] = np.clip(v_init_bg[None, :, :] + rng.normal(0.0, init_jitter, size=(R, B, G)), 0.0, 1.0)
    n[0] = np.clip(n_init_bg[None, :, :] + rng.normal(0.0, init_jitter, size=(R, B, G)), 0.0, 1.0)

    e_curr = e_base.copy()
    f_curr = np.zeros((R, B, G))
    impulse_i = np.ones((R, B, G))
    peak_timer = np.zeros((R, B, G))

    L_graph = build_graph_laplacian(coupling_J) if coupling_J is not None else None
    W = np.asarray(domain_transfer_W, dtype=float).copy() if domain_transfer_W is not None else None
    if W is not None:
        np.fill_diagonal(W, 0.0)

    fired_panic = np.zeros((R, B, G), dtype=bool)
    fired_boredom = np.zeros((R, B, G), dtype=bool)
    fired_spite = np.zeros((R, B, G), dtype=bool)

    for s in range(steps):
        u_s, xi_s, v_s, n_s, t_s = u[s], xi[s], v[s], n[s], t[s]
        local_m = u_s / xi_s

        # 0. Circadian Awakeness wave (24h period -> pi radians inside sin^2 / cos^2)
        theta = (np.pi / 24.0) * (t_s - phase_shift)
        awake = np.where(is_night_owl, np.sin(theta) ** 2, np.cos(theta) ** 2)

        # 1. Boredom_Side_Quest (moderate curiosity spike)
        if Boredom_Side_Quest:
            m_bored = (~fired_boredom) & (local_m >= boredom_threshold) & (xi_s < 0.90)
            if np.any(m_bored):
                fired_boredom |= m_bored
                xi_s = np.where(m_bored, np.clip(xi_s + 0.18, alpha0, 1.0), xi_s)
                lam = np.where(m_bored, np.maximum(lam, 0.11), lam)
                k = np.where(m_bored, k * 1.35, k)
                p_tilde = np.where(m_bored, p_tilde * 1.35, p_tilde)
                e_curr = np.where(m_bored, np.maximum(e_curr, 0.70), e_curr)
                impulse_i = np.where(m_bored, np.maximum(impulse_i, 1.5), impulse_i)
                peak_timer = np.where(m_bored, np.maximum(peak_timer, 0.6 * t_peak_dur), peak_timer)

        # 2. Deadline_Panic (adrenaline surge pinned at 1.0 for t_peak)
        if Deadline_Panic:
            time_left = t_exam_vec[None, None, :] - t_s
            m_panic = (
                (~fired_panic)
                & (time_left > 0.0)
                & (time_left <= panic_window)
                & (u_s < panic_mastery_floor)
            )
            if np.any(m_panic):
                fired_panic |= m_panic
                v_s = np.where(m_panic, np.clip(v_s + 0.16, 0.0, 1.0), v_s)
                xi_s = np.where(m_panic, np.clip(xi_s - 0.20, alpha0, 1.0), xi_s)
                lam = np.where(m_panic, -0.06, lam)
                tau = np.where(m_panic, np.maximum(tau, 0.75), tau)
                sigma = np.where(m_panic, sigma * 0.25, sigma)
                e_curr = np.where(m_panic, 1.0, e_curr)
                impulse_i = np.where(m_panic, np.maximum(impulse_i, 2.2), impulse_i)
                peak_timer = np.where(m_panic, t_peak_dur, peak_timer)

        # 3. Stochastic Spite_Taunt (spite overdrive pinned at 1.0 with extended tail)
        if Spite_Taunt:
            p_hit = 1.0 - np.exp(-taunt_rate * tau)
            m_spite = (~fired_spite) & (rng.random((R, B, G)) < p_hit)
            if np.any(m_spite):
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

        # 4. Net Cognitive Multiplier: eta_cog = awake * max(0, E - F)
        eta_cog = awake * np.maximum(0.0, e_curr - f_curr) if Energy_Burnout else np.ones((R, B, G))

        awake_hist[s] = awake
        energy_hist[s] = e_curr
        fatigue_hist[s] = f_curr
        eta_cog_hist[s] = eta_cog

        # 5. Cross-Domain Prerequisite Transfer (contraction over axis G)
        if W is not None:
            prereq_signal = u_s @ W.T
            alpha_eff = alpha0[None, None, :] * np.exp(-beta_alpha * prereq_signal)
            v_eff = np.minimum(1.0, v_s + beta_v * prereq_signal)
        else:
            alpha_eff = np.broadcast_to(alpha0, (R, B, G))
            v_eff = v_s

        alpha_eff_hist[s] = alpha_eff
        v_eff_hist[s] = v_eff

        # 6. Ingestion modulated by eta_cog + Graph Laplacian diffusion
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

        # 8. Fatigue buildup/recovery & post-peak energy decay
        if Energy_Burnout:
            df_build = np.maximum(0.0, awake - 0.5) * e_curr * k_fatigue * tau
            df_recov = np.maximum(0.0, 0.5 - awake) * k_recovery * tau
            f_curr = np.clip(f_curr + df_build - df_recov, 0.0, 1.0)

            holding = peak_timer > 0.0
            e_decayed = e_base + (e_curr - e_base) * np.exp(-(k_energy / impulse_i) * tau)
            e_curr = np.where(holding, e_curr, e_decayed)
            peak_timer = np.maximum(0.0, peak_timer - tau)

    # Final step recording
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
        "stats": {
            "pass_matrix": np.mean(exam_u >= u_pass_vec[None, None, :], axis=0),
            "sat_matrix": np.mean(u[-1] >= 0.95, axis=0),
            "burnout_rate": np.mean(np.any((energy_hist - fatigue_hist) <= 0.0, axis=0), axis=0),
        },
    }


def plot_energy_burnout_diagnostics(res: dict, domain_idx: int = 0, save_path: str | None = None) -> None:
    """
    Visualizes Circadian Awakeness, Adrenaline Energy Spikes vs. Fatigue Burnout,
    Total Cognitive Multiplier eta_cog, and Resulting Mastery Plateaus for a chosen domain.
    """
    t = res["t"][:, :, :, domain_idx]
    u = res["u"][:, :, :, domain_idx]
    awake = res["awake"][:, :, :, domain_idx]
    energy = res["energy"][:, :, :, domain_idx]
    fatigue = res["fatigue"][:, :, :, domain_idx]
    eta_cog = res["eta_cog"][:, :, :, domain_idx]

    _, R, B = u.shape
    tl_names = res["timeline_names"]
    dom_name = res["domain_names"][domain_idx]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), dpi=105)
    fig.suptitle(
        f"Circadian Energy & Post-Spite Burnout Dynamics — {dom_name} ({R} Runs)",
        fontsize=13.5,
        fontweight="bold",
        y=0.965,
    )
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.06, right=0.97, hspace=0.36, wspace=0.20)

    colors = plt.cm.plasma(np.linspace(0.22, 0.88, B))
    ax_circ, ax_ef, ax_eta, ax_u = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for b in range(B):
        c = colors[b]
        t_med = np.median(t[:, :, b], axis=1)

        ax_circ.plot(t_med, np.median(awake[:, :, b], axis=1), color=c, lw=2.2, label=tl_names[b])

        ax_ef.plot(t_med, np.median(energy[:, :, b], axis=1), color=c, lw=2.2, label=f"{tl_names[b]} Energy")
        ax_ef.plot(t_med, np.median(fatigue[:, :, b], axis=1), color=c, lw=1.8, ls="--", alpha=0.85, label=f"{tl_names[b]} Fatigue")

        eta10, eta50, eta90 = np.percentile(eta_cog[:, :, b], [10, 50, 90], axis=1)
        ax_eta.fill_between(t_med, eta10, eta90, color=c, alpha=0.16)
        ax_eta.plot(t_med, eta50, color=c, lw=2.2, label=tl_names[b])

        u10, u50, u90 = np.percentile(u[:, :, b], [10, 50, 90], axis=1)
        ax_u.fill_between(t_med, u10, u90, color=c, alpha=0.16)
        ax_u.plot(t_med, u50, color=c, lw=2.3, label=tl_names[b])

    ax_circ.axhline(0.5, color="white", ls=":", lw=1.2, alpha=0.6, label="Burnout / Recovery Pivot (0.5)")
    ax_circ.set_title("1. Circadian Awakeness Wave: Night Owl sin² vs. Early Bird cos²", pad=8)
    ax_circ.set_xlabel("Time of Day / Proper Time (hours)")
    ax_circ.set_ylabel("Awakeness Coefficient")
    ax_circ.set_ylim(-0.04, 1.06)
    ax_circ.grid(alpha=0.2, linestyle="--")
    ax_circ.legend(loc="upper right", fontsize=8)

    ax_ef.set_title("2. Energy Spikes (Solid) vs. Fatigue Buildup (Dashed)", pad=8)
    ax_ef.set_xlabel("Time (hours)")
    ax_ef.set_ylabel("Normalized Level (0–1)")
    ax_ef.set_ylim(-0.04, 1.06)
    ax_ef.grid(alpha=0.2, linestyle="--")
    ax_ef.legend(loc="upper right", fontsize=7.5, ncol=2)

    ax_eta.set_title("3. Total Cognitive Efficiency η_cog = awake · max(0, E − F)", pad=8)
    ax_eta.set_xlabel("Time (hours)")
    ax_eta.set_ylabel("Effective Multiplier η_cog")
    ax_eta.set_ylim(-0.04, 1.06)
    ax_eta.grid(alpha=0.2, linestyle="--")
    ax_eta.legend(loc="upper right", fontsize=8)

    ax_u.axvline(res["t_exam"][domain_idx], color="#ffbe0b", ls="--", lw=1.4, label=f"Exam (t={res['t_exam'][domain_idx]}h)")
    ax_u.axhline(res["u_pass"][domain_idx], color="#8ac926", ls=":", lw=1.3, label="Pass Floor (60%)")
    ax_u.set_title("4. Domain Mastery u(t) Under Burnout Throttling", pad=8)
    ax_u.set_xlabel("Time (hours)")
    ax_u.set_ylabel("Domain Mastery (u)")
    ax_u.set_ylim(-0.04, 1.06)
    ax_u.grid(alpha=0.2, linestyle="--")
    ax_u.legend(loc="lower right", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()