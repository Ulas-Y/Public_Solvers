from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from timeline_solver import TimelineSpec
from timeline_solver import estimate_domain_bounds, compute_info_volume

def build_graph_laplacian(coupling_J: np.ndarray) -> np.ndarray:
    """
    Constructs L = D_deg - J from coupling adjacency matrix J (B x B).
    - (U @ L.T) computes exact diffusive inflow sum_j J_ij * (u_j - u_i).
    """
    J = np.asarray(coupling_J, dtype=float).copy()
    np.fill_diagonal(J, 0.0)
    D_deg = np.diag(np.sum(J, axis=1))
    return D_deg - J


def simulate_multiverse_mc(
    steps: int,
    timelines: list[TimelineSpec],
    n_runs: int = 500,
    Deadline_Panic: bool = True,
    Boredom_Side_Quest: bool = True,
    Spite_Taunt: bool = True,
    taunt_rate: float = 0.07,
    t_exam: float = 10.5,
    u_pass: float = 0.60,
    panic_window: float = 4.0,
    panic_mastery_floor: float = 0.45,
    boredom_threshold: float = 0.94,
    base_amount: float = 20.0,
    complexity: float = 1.5,
    n_harmonics: int = 3,
    coupling_J: np.ndarray | None = None,
    init_jitter: float = 0.02,
    seed: int | None = 42,
) -> dict:
    """
    Fully vectorized 3D tensor solver of shape (steps + 1, n_runs, B).
    Zero inner Python loops over runs or timeline branches.
    """
    rng = np.random.default_rng(seed)
    M, D, alpha = estimate_domain_bounds(base_amount, complexity, n_harmonics)
    B = len(timelines)
    R = n_runs

    # Broadcast branch parameters to (R, B)
    tau = np.broadcast_to(np.array([tl.tau for tl in timelines]), (R, B)).copy()
    sigma = np.broadcast_to(np.array([tl.sigma for tl in timelines]), (R, B)).copy()
    lam = np.broadcast_to(np.array([tl.lam for tl in timelines]), (R, B)).copy()
    k = np.broadcast_to(np.array([tl.k for tl in timelines]), (R, B)).copy()
    p_tilde = np.broadcast_to(np.array([tl.p_tilde for tl in timelines]), (R, B)).copy()

    # 3D state tensors: (steps + 1, R, B)
    t = np.zeros((steps + 1, R, B))
    u = np.zeros((steps + 1, R, B))
    xi = np.zeros((steps + 1, R, B))
    v = np.zeros((steps + 1, R, B))
    n = np.zeros((steps + 1, R, B))

    xi_base = np.array([
        compute_info_volume(base_amount, complexity, tl.phi0, n_harmonics) / D
        for tl in timelines
    ])
    v_base = np.array([tl.v0 for tl in timelines], dtype=float)
    n_base = np.array([tl.n0 for tl in timelines], dtype=float)

    # Perturb initial conditions across Monte Carlo realizations
    xi[0] = np.clip(xi_base + rng.normal(0.0, init_jitter * 0.5, size=(R, B)), alpha, 1.0)
    v[0] = np.clip(v_base + rng.normal(0.0, init_jitter, size=(R, B)), 0.0, 1.0)
    n[0] = np.clip(n_base + rng.normal(0.0, init_jitter, size=(R, B)), 0.0, 1.0)

    L_graph = build_graph_laplacian(coupling_J) if coupling_J is not None else None

    fired_panic = np.zeros((R, B), dtype=bool)
    fired_boredom = np.zeros((R, B), dtype=bool)
    fired_spite = np.zeros((R, B), dtype=bool)

    for s in range(steps):
        u_s, xi_s, v_s, n_s, t_s = u[s], xi[s], v[s], n[s], t[s]
        local_m = u_s / xi_s

        # 1. Vectorized Boredom_Side_Quest mask (R, B)
        if Boredom_Side_Quest:
            m_bored = (~fired_boredom) & (local_m >= boredom_threshold) & (xi_s < 0.90)
            if np.any(m_bored):
                fired_boredom |= m_bored
                xi_s = np.where(m_bored, np.clip(xi_s + 0.18, alpha, 1.0), xi_s)
                lam = np.where(m_bored, np.maximum(lam, 0.11), lam)
                k = np.where(m_bored, k * 1.35, k)
                p_tilde = np.where(m_bored, p_tilde * 1.35, p_tilde)

        # 2. Vectorized Deadline_Panic mask (R, B)
        if Deadline_Panic:
            time_left = t_exam - t_s
            m_panic = (
                (~fired_panic)
                & (time_left > 0.0)
                & (time_left <= panic_window)
                & (u_s < panic_mastery_floor)
            )
            if np.any(m_panic):
                fired_panic |= m_panic
                v_s = np.where(m_panic, np.clip(v_s + 0.16, 0.0, 1.0), v_s)
                xi_s = np.where(m_panic, np.clip(xi_s - 0.20, alpha, 1.0), xi_s)
                lam = np.where(m_panic, -0.06, lam)
                tau = np.where(m_panic, np.maximum(tau, 0.75), tau)
                sigma = np.where(m_panic, sigma * 0.25, sigma)

        # 3. Vectorized stochastic Poisson Spite_Taunt mask (R, B)
        if Spite_Taunt:
            p_hit = 1.0 - np.exp(-taunt_rate * tau)
            m_spite = (~fired_spite) & (rng.random((R, B)) < p_hit)
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

        # 4. Tensorized ingestion + Graph Laplacian cross-timeline diffusion
        u_gain = np.where(
            v_s >= alpha,
            np.minimum(v_s, xi_s) * n_s * np.maximum(0.0, 1.0 - u_s / xi_s),
            0.0,
        )
        u_tilde = (u_s + u_gain) * np.exp(-sigma * tau)

        if L_graph is not None:
            u_next = np.clip(u_tilde - (u_tilde @ L_graph.T) * tau, 0.0, 1.0)
        else:
            u_next = u_tilde

        local_mastery_next = u_next / xi_s

        # 5. Tensorized meta-learning & scope updates
        t[s + 1] = t_s + tau
        u[s + 1] = u_next
        n[s + 1] = np.minimum(1.0, n_s * np.exp(k * local_mastery_next * tau))
        v[s + 1] = np.minimum(1.0, v_s * np.exp(p_tilde * local_mastery_next * tau))
        xi[s + 1] = np.clip(xi_s * np.exp(lam * tau), alpha, 1.0)

    # Interpolate mastery at exact exam timestamp t_exam per realization
    exam_u = np.array([
        [np.interp(t_exam, t[:, r, b], u[:, r, b]) for b in range(B)]
        for r in range(R)
    ])

    stats = {
        "pass_prob": np.mean(exam_u >= u_pass, axis=0),
        "saturation_prob": np.mean(u[-1] >= 0.95, axis=0),
        "panic_rate": np.mean(fired_panic, axis=0),
        "boredom_rate": np.mean(fired_boredom, axis=0),
        "spite_rate": np.mean(fired_spite, axis=0),
        "exam_u_median": np.median(exam_u, axis=0),
    }

    return {
        "names": [tl.name for tl in timelines],
        "bounds": np.array([M, D, alpha]),
        "t_exam": t_exam,
        "u_pass": u_pass,
        "t": t,
        "u": u,
        "xi": xi,
        "v": v,
        "n": n,
        "stats": stats,
    }


def plot_monte_carlo_ensemble(res: dict, save_path: str | None = None) -> None:
    t, u, xi, v = res["t"], res["u"], res["xi"], res["v"]
    _, _, alpha = res["bounds"]
    _, R, B = u.shape
    stats = res["stats"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=105)
    fig.suptitle(
        f"Tensorized Monte Carlo Multiverse ({R} Realizations × {B} Timelines)",
        fontsize=13.5,
        fontweight="bold",
        y=0.965,
    )
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.06, right=0.98, hspace=0.35, wspace=0.18)

    colors = plt.cm.plasma(np.linspace(0.22, 0.88, B))
    ax_u, ax_xi, ax_v, ax_bar = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for b in range(B):
        c = colors[b]
        t_med = np.median(t[:, :, b], axis=1)

        u10, u50, u90 = np.percentile(u[:, :, b], [10, 50, 90], axis=1)
        ax_u.fill_between(t_med, u10, u90, color=c, alpha=0.18)
        ax_u.plot(t_med, u50, color=c, lw=2.3, label=f"{res['names'][b]} (med)")

        xi10, xi50, xi90 = np.percentile(xi[:, :, b], [10, 50, 90], axis=1)
        ax_xi.fill_between(t_med, xi10, xi90, color=c, alpha=0.18)
        ax_xi.plot(t_med, xi50, color=c, lw=2.3)

        v10, v50, v90 = np.percentile(v[:, :, b], [10, 50, 90], axis=1)
        ax_v.fill_between(t_med, v10, v90, color=c, alpha=0.18)
        ax_v.plot(t_med, v50, color=c, lw=2.3)

    ax_u.axvline(res["t_exam"], color="#ffbe0b", ls="--", lw=1.4, label=f"Exam (t={res['t_exam']})")
    ax_u.axhline(res["u_pass"], color="#8ac926", ls=":", lw=1.3, label=f"Pass Floor ({res['u_pass']:.0%})")
    ax_u.set_title("1. Domain Mastery u(t) — Median & 10–90% Fan", pad=8)
    ax_u.set_xlabel("Median Proper Time (t)")
    ax_u.set_ylabel("Domain Mastery (u)")
    ax_u.set_ylim(-0.04, 1.06)
    ax_u.grid(alpha=0.2, linestyle="--")
    ax_u.legend(loc="lower right", fontsize=8)

    ax_xi.axhline(alpha, color="#ff595e", ls=":", lw=1.5, label=f"Axiom Floor α = {alpha:.2f}")
    ax_xi.axhline(1.0, color="#8ac926", ls="--", lw=1.2, label="Domain Ceiling (1.0)")
    ax_xi.set_title("2. Active Scope ξ(t) — Median & 10–90% Fan", pad=8)
    ax_xi.set_xlabel("Median Proper Time (t)")
    ax_xi.set_ylabel("Normalized Scope (ξ)")
    ax_xi.set_ylim(alpha - 0.08, 1.06)
    ax_xi.grid(alpha=0.2, linestyle="--")
    ax_xi.legend(loc="lower right", fontsize=8)

    ax_v.axhline(alpha, color="#ff595e", ls="--", lw=1.8, label=f"Ignition Barrier α = {alpha:.2f}")
    ax_v.fill_between([0, np.max(t)], 0, alpha, color="#ff595e", alpha=0.10, label="Brick Mode (v < α)")
    ax_v.set_title("3. Bandwidth v(t) — Median & 10–90% Fan", pad=8)
    ax_v.set_xlabel("Median Proper Time (t)")
    ax_v.set_ylabel("Normalized Bandwidth (v)")
    ax_v.set_ylim(0.0, 1.06)
    ax_v.grid(alpha=0.2, linestyle="--")
    ax_v.legend(loc="lower right", fontsize=8)

    x = np.arange(B)
    w = 0.18
    ax_bar.bar(x - 1.5 * w, stats["pass_prob"] * 100, width=w, color="#8ac926", label="Exam Pass %")
    ax_bar.bar(x - 0.5 * w, stats["saturation_prob"] * 100, width=w, color="#00f5d4", label="Full Domain (u≥0.95) %")
    ax_bar.bar(x + 0.5 * w, stats["panic_rate"] * 100, width=w, color="#ffbe0b", label="Panic Trigger %")
    ax_bar.bar(x + 1.5 * w, stats["spite_rate"] * 100, width=w, color="#ff006e", label="Spite Taunt %")
    ax_bar.set_title("4. Monte Carlo Outcome & Event Probabilities", pad=8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(res["names"], fontsize=9)
    ax_bar.set_ylabel("Probability (%)")
    ax_bar.set_ylim(0, 128)
    ax_bar.grid(axis="y", alpha=0.2, linestyle="--")
    ax_bar.legend(loc="upper right", fontsize=7.5, ncol=2)

    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()