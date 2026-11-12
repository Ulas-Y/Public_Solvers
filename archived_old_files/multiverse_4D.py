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


@dataclass
class DomainSpec:
    name: str
    base_amount: float = 20.0
    complexity: float = 1.5
    n_harmonics: int = 3
    t_exam: float = 10.5
    u_pass: float = 0.60
    v_offset: float = 0.0  # Domain-specific bandwidth shift (e.g., <0 for harder advanced fields)


def build_graph_laplacian(coupling_J: np.ndarray) -> np.ndarray:
    """
    Constructs L = D_deg - J from adjacency matrix J (B x B).
    Safely passes through if a valid graph Laplacian is already provided.
    """
    M = np.asarray(coupling_J, dtype=float).copy()
    off_diag = M - np.diag(np.diag(M))
    if np.all(np.diag(M) >= 0.0) and np.all(off_diag <= 0.0) and np.allclose(np.sum(M, axis=1), 0.0):
        return M
    np.fill_diagonal(M, 0.0)
    D_deg = np.diag(np.sum(M, axis=1))
    return D_deg - M


def simulate_multiverse_4d(
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
    taunt_rate: float = 0.07,
    panic_window: float = 4.0,
    panic_mastery_floor: float = 0.45,
    boredom_threshold: float = 0.94,
    init_jitter: float = 0.02,
    seed: int | None = 42,
) -> dict:
    """
    4D Reverse-Spacetime Tensor Solver of shape (steps + 1, R, B, G):
      - Axis 0: Simulation step s
      - Axis 1: Monte Carlo realization r in [0, R)
      - Axis 2: Parallel timeline branch b in [0, B)
      - Axis 3: Knowledge domain g in [0, G)
    """
    rng = np.random.default_rng(seed)
    R, B, G = n_runs, len(timelines), len(domains)

    bounds = np.array([
        estimate_domain_bounds(dom.base_amount, dom.complexity, dom.n_harmonics)
        for dom in domains
    ])  # (G, 3) -> [M_g, D_g, alpha0_g]
    _, D_vec, alpha0 = bounds[:, 0], bounds[:, 1], bounds[:, 2]

    t_exam_vec = np.array([dom.t_exam for dom in domains], dtype=float)
    u_pass_vec = np.array([dom.u_pass for dom in domains], dtype=float)
    v_off_vec = np.array([dom.v_offset for dom in domains], dtype=float)

    def bcast_tl(attr: str) -> np.ndarray:
        arr = np.array([getattr(tl, attr) for tl in timelines], dtype=float)[:, None]
        return np.broadcast_to(arr, (R, B, G)).copy()

    tau = bcast_tl("tau")
    sigma = bcast_tl("sigma")
    lam = bcast_tl("lam")
    k = bcast_tl("k")
    p_tilde = bcast_tl("p_tilde")

    t = np.zeros((steps + 1, R, B, G))
    u = np.zeros((steps + 1, R, B, G))
    xi = np.zeros((steps + 1, R, B, G))
    v = np.zeros((steps + 1, R, B, G))
    n = np.zeros((steps + 1, R, B, G))
    alpha_eff_hist = np.zeros((steps + 1, R, B, G))
    v_eff_hist = np.zeros((steps + 1, R, B, G))

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

        # 1. Vectorized Boredom_Side_Quest (R, B, G)
        if Boredom_Side_Quest:
            m_bored = (~fired_boredom) & (local_m >= boredom_threshold) & (xi_s < 0.90)
            if np.any(m_bored):
                fired_boredom |= m_bored
                xi_s = np.where(m_bored, np.clip(xi_s + 0.18, alpha0, 1.0), xi_s)
                lam = np.where(m_bored, np.maximum(lam, 0.11), lam)
                k = np.where(m_bored, k * 1.35, k)
                p_tilde = np.where(m_bored, p_tilde * 1.35, p_tilde)

        # 2. Vectorized Deadline_Panic (R, B, G)
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

        # 3. Vectorized Stochastic Spite_Taunt (R, B, G)
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

        xi[s], v[s], n[s] = xi_s, v_s, n_s

        # 4. Cross-Domain Prerequisite Transfer (contraction over axis G)
        if W is not None:
            prereq_signal = u_s @ W.T  # (R, B, G)
            alpha_eff = alpha0[None, None, :] * np.exp(-beta_alpha * prereq_signal)
            v_eff = np.minimum(1.0, v_s + beta_v * prereq_signal)
        else:
            alpha_eff = np.broadcast_to(alpha0, (R, B, G))
            v_eff = v_s

        alpha_eff_hist[s] = alpha_eff
        v_eff_hist[s] = v_eff

        # 5. Ingestion + Graph Laplacian Cross-Timeline Diffusion (contraction over axis B)
        u_gain = np.where(
            v_eff >= alpha_eff,
            np.minimum(v_eff, xi_s) * n_s * np.maximum(0.0, 1.0 - u_s / xi_s),
            0.0,
        )
        u_tilde = (u_s + u_gain) * np.exp(-sigma * tau)

        if L_graph is not None:
            diffusion = np.einsum("bc,rcg->rbg", L_graph, u_tilde)
            u_next = np.clip(u_tilde - diffusion * tau, 0.0, 1.0)
        else:
            u_next = u_tilde

        local_mastery_next = u_next / xi_s

        # 6. Meta-learning & scope updates
        t[s + 1] = t_s + tau
        u[s + 1] = u_next
        n[s + 1] = np.minimum(1.0, n_s * np.exp(k * local_mastery_next * tau))
        v[s + 1] = np.minimum(1.0, v_s * np.exp(p_tilde * local_mastery_next * tau))
        xi[s + 1] = np.clip(xi_s * np.exp(lam * tau), alpha0, 1.0)

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
        "stats": {
            "pass_matrix": np.mean(exam_u >= u_pass_vec[None, None, :], axis=0),  # (B, G)
            "sat_matrix": np.mean(u[-1] >= 0.95, axis=0),                          # (B, G)
            "panic_matrix": np.mean(fired_panic, axis=0),
            "spite_matrix": np.mean(fired_spite, axis=0),
        },
    }


def plot_multiverse_4d(res: dict, save_path: str | None = None) -> None:
    t, u, xi, v_eff, alpha_eff = res["t"], res["u"], res["xi"], res["v_eff"], res["alpha_eff"]
    _, R, B, G = u.shape
    tl_names, dom_names = res["timeline_names"], res["domain_names"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), dpi=105)
    fig.suptitle(
        f"4D Reverse-Spacetime Multiverse ({R} Runs × {B} Timelines × {G} Coupled Domains)",
        fontsize=13.5,
        fontweight="bold",
        y=0.965,
    )
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.06, right=0.95, hspace=0.36, wspace=0.27)

    colors = plt.cm.plasma(np.linspace(0.22, 0.88, B))
    linestyles = ["-", "--", ":", "-."]
    ax_u, ax_barrier, ax_xi, ax_heat = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for b in range(B):
        c = colors[b]
        for g in range(G):
            ls = linestyles[g % len(linestyles)]
            t_med = np.median(t[:, :, b, g], axis=1)

            ax_u.plot(t_med, np.median(u[:, :, b, g], axis=1), color=c, ls=ls, lw=2.1,
                      label=f"{tl_names[b]} | {dom_names[g]}")
            ax_barrier.plot(t_med, np.median(v_eff[:, :, b, g], axis=1), color=c, ls=ls, lw=2.0)
            if b == 0:
                ax_barrier.plot(t_med, np.median(alpha_eff[:, :, b, g], axis=1), color="#ff595e",
                                ls=ls, lw=1.6, alpha=0.85, label=f"α_eff ({dom_names[g]})")
            ax_xi.plot(t_med, np.median(xi[:, :, b, g], axis=1), color=c, ls=ls, lw=2.0)

    ax_u.set_title("1. Multi-Domain Mastery u(t) — Median Trajectories", pad=8)
    ax_u.set_xlabel("Median Proper Time (t)")
    ax_u.set_ylabel("Domain Mastery (u)")
    ax_u.set_ylim(-0.04, 1.06)
    ax_u.grid(alpha=0.2, linestyle="--")
    ax_u.legend(loc="lower right", fontsize=7.2, ncol=2)

    ax_barrier.set_title("2. Effective Bandwidth v_eff(t) vs. Lowered Barrier α_eff(t)", pad=8)
    ax_barrier.set_xlabel("Median Proper Time (t)")
    ax_barrier.set_ylabel("Normalized Rate / Barrier")
    ax_barrier.set_ylim(0.0, 1.06)
    ax_barrier.grid(alpha=0.2, linestyle="--")
    ax_barrier.legend(loc="lower right", fontsize=8)

    ax_xi.axhline(1.0, color="#8ac926", ls="--", lw=1.2, label="Domain Ceiling (1.0)")
    ax_xi.set_title("3. Active Scope Expansion ξ(t) across Domains", pad=8)
    ax_xi.set_xlabel("Median Proper Time (t)")
    ax_xi.set_ylabel("Normalized Scope (ξ)")
    ax_xi.set_ylim(0.35, 1.06)
    ax_xi.grid(alpha=0.2, linestyle="--")
    ax_xi.legend(loc="lower right", fontsize=8)

    pass_mat = res["stats"]["pass_matrix"] * 100.0
    im = ax_heat.imshow(pass_mat, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax_heat.set_title("4. Exam Pass Probability Matrix (%)", pad=8)
    ax_heat.set_xticks(np.arange(G))
    ax_heat.set_xticklabels(dom_names, fontsize=9)
    ax_heat.set_yticks(np.arange(B))
    ax_heat.set_yticklabels(tl_names, fontsize=9)
    for b in range(B):
        for g in range(G):
            val = pass_mat[b, g]
            txt_col = "black" if val > 55 else "white"
            ax_heat.text(g, b, f"{val:.1f}%", ha="center", va="center", color=txt_col, fontweight="bold", fontsize=11)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Pass Probability (%)", fontsize=8.5)

    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()