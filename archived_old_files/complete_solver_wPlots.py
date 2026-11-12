from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

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
class ShockEvent:
    step: int
    branch: int
    new_regime: TimelineSpec | None = None
    dv: float = 0.0
    dxi: float = 0.0
    dn: float = 0.0
    label: str = "Shock"


def simulate_multiverse_auto(
    steps: int,
    timelines: list[TimelineSpec],
    shocks: list[ShockEvent] | None = None,
    Deadline_Panic: bool = True,
    Boredom_Side_Quest: bool = True,
    Spite_Taunt: bool = True,
    taunt_rate: float = 0.07,
    t_exam: float = 10.5,
    panic_window: float = 4.0,
    panic_mastery_floor: float = 0.45,
    boredom_threshold: float = 0.94,
    base_amount: float = 20.0,
    complexity: float = 1.5,
    n_harmonics: int = 3,
    coupling_J: np.ndarray | None = None,
    seed: int | None = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    M, D, alpha = estimate_domain_bounds(base_amount, complexity, n_harmonics)
    B = len(timelines)

    tau = np.array([tl.tau for tl in timelines], dtype=float)
    sigma = np.array([tl.sigma for tl in timelines], dtype=float)
    lam = np.array([tl.lam for tl in timelines], dtype=float)
    k = np.array([tl.k for tl in timelines], dtype=float)
    p_tilde = np.array([tl.p_tilde for tl in timelines], dtype=float)

    t = np.zeros((steps + 1, B))
    u = np.zeros((steps + 1, B))
    xi = np.zeros((steps + 1, B))
    v = np.zeros((steps + 1, B))
    n = np.zeros((steps + 1, B))

    xi[0] = np.array([
        compute_info_volume(base_amount, complexity, tl.phi0, n_harmonics) / D
        for tl in timelines
    ])
    v[0] = np.array([tl.v0 for tl in timelines], dtype=float)
    n[0] = np.array([tl.n0 for tl in timelines], dtype=float)

    shock_map: dict[int, list[ShockEvent]] = {}
    for sh in (shocks or []):
        shock_map.setdefault(sh.step, []).append(sh)

    current_names = [tl.name for tl in timelines]
    event_log: list[str] = []
    markers: list[tuple[int, int, str]] = []  # (step, branch, event_type)

    fired_panic = np.zeros(B, dtype=bool)
    fired_boredom = np.zeros(B, dtype=bool)
    fired_spite = np.zeros(B, dtype=bool)

    for s in range(steps):
        # 0A. Optional manual shocks
        if s in shock_map:
            for sh in shock_map[s]:
                b = sh.branch
                v[s, b] = np.clip(v[s, b] + sh.dv, 0.0, 1.0)
                xi[s, b] = np.clip(xi[s, b] + sh.dxi, alpha, 1.0)
                n[s, b] = np.clip(n[s, b] + sh.dn, 0.0, 1.0)
                if sh.new_regime is not None:
                    nr = sh.new_regime
                    tau[b], sigma[b], lam[b], k[b], p_tilde[b] = (
                        nr.tau, nr.sigma, nr.lam, nr.k, nr.p_tilde
                    )
                    current_names[b] = f"{nr.name} [{sh.label}]"
                event_log.append(f"Step {s:02d} | Branch {b}: Manual [{sh.label}]")
                markers.append((s, b, "Manual"))

        # 0B. Deterministic + stochastic state-triggered shocks
        for b in range(B):
            local_m = u[s, b] / xi[s, b]

            # 1. Boredom_Side_Quest (local syllabus saturation -> rabbit-hole explosion)
            if (
                Boredom_Side_Quest
                and not fired_boredom[b]
                and local_m >= boredom_threshold
                and xi[s, b] < 0.90
            ):
                fired_boredom[b] = True
                xi[s, b] = np.clip(xi[s, b] + 0.18, alpha, 1.0)
                lam[b] = max(lam[b], 0.11)
                k[b] *= 1.35
                p_tilde[b] *= 1.35
                current_names[b] += " -> [Boredom]"
                event_log.append(
                    f"Step {s:02d} (t={t[s, b]:.1f}) | Branch {b} ({timelines[b].name}): Boredom_Side_Quest"
                )
                markers.append((s, b, "Boredom"))

            # 2. Deadline_Panic (exam proximity + low mastery -> adrenaline triage)
            time_left = t_exam - t[s, b]
            if (
                Deadline_Panic
                and not fired_panic[b]
                and 0.0 < time_left <= panic_window
                and u[s, b] < panic_mastery_floor
            ):
                fired_panic[b] = True
                v[s, b] = np.clip(v[s, b] + 0.16, 0.0, 1.0)
                xi[s, b] = np.clip(xi[s, b] - 0.20, alpha, 1.0)
                lam[b] = -0.06
                tau[b] = max(tau[b], 0.75)
                sigma[b] *= 0.25
                current_names[b] += " -> [Panic]"
                event_log.append(
                    f"Step {s:02d} (t={t[s, b]:.1f}) | Branch {b} ({timelines[b].name}): Deadline_Panic"
                )
                markers.append((s, b, "Panic"))

            # 3. Spite_Taunt (stochastic Poisson arrival: P = 1 - exp(-rate * tau))
            if Spite_Taunt and not fired_spite[b]:
                p_hit = 1.0 - np.exp(-taunt_rate * tau[b])
                if rng.random() < p_hit:
                    fired_spite[b] = True
                    v[s, b] = np.clip(v[s, b] + 0.22, 0.0, 1.0)
                    n[s, b] = np.clip(n[s, b] + 0.15, 0.0, 1.0)
                    tau[b] = max(tau[b], 0.85)
                    sigma[b] *= 0.15
                    lam[b] = max(lam[b], 0.10)
                    k[b] *= 1.50
                    p_tilde[b] *= 1.50
                    current_names[b] += " -> [SPITE]"
                    event_log.append(
                        f"Step {s:02d} (t={t[s, b]:.1f}) | Branch {b} ({timelines[b].name}): Spite_Taunt (p={p_hit:.2%})!"
                    )
                    markers.append((s, b, "Spite"))

        # 1. Ingestion with per-timeline ignition check (v >= alpha)
        active = v[s] >= alpha
        u_gain = np.where(
            active,
            np.minimum(v[s], xi[s]) * n[s] * np.maximum(0.0, 1.0 - u[s] / xi[s]),
            0.0,
        )

        # 2. Retention + optional spatial-time Laplacian coupling
        u_tilde = (u[s] + u_gain) * np.exp(-sigma * tau)
        if coupling_J is not None:
            diff_matrix = u_tilde[None, :] - u_tilde[:, None]
            leakage = np.sum(coupling_J * diff_matrix, axis=1) * tau
            u_next = np.clip(u_tilde + leakage, 0.0, 1.0)
        else:
            u_next = u_tilde

        local_mastery = u_next / xi[s]

        # 3. Meta-learning & clamped scope creep
        n_next = np.minimum(1.0, n[s] * np.exp(k * local_mastery * tau))
        v_next = np.minimum(1.0, v[s] * np.exp(p_tilde * local_mastery * tau))
        xi_next = np.clip(xi[s] * np.exp(lam * tau), alpha, 1.0)

        t[s + 1] = t[s] + tau
        u[s + 1] = u_next
        xi[s + 1] = xi_next
        v[s + 1] = v_next
        n[s + 1] = n_next

    return {
        "names": current_names,
        "event_log": event_log,
        "markers": markers,
        "bounds": np.array([M, D, alpha]),
        "t": t,
        "u": u,
        "xi": xi,
        "v": v,
        "n": n,
    }


def plot_multiverse_diagnostics(
    res: dict, save_path: str | None = None, interactive: bool = True
) -> None:
    t, u, xi, v, n = res["t"], res["u"], res["xi"], res["v"], res["n"]
    _, _, alpha = res["bounds"]
    total_steps, B = u.shape[0] - 1, u.shape[1]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.8), dpi=130)
    fig.suptitle(
        "Coupled Nonlinear Cognitive Dynamics — Multiverse Phase Diagnostics",
        fontsize=14,
        fontweight="bold",
        y=0.972,
    )

    # Dedicated margins for top telemetry pill, subplot titles, and bottom scrubber
    bottom_margin = 0.125 if interactive else 0.07
    fig.subplots_adjust(
        top=0.865,
        bottom=bottom_margin,
        left=0.06,
        right=0.98,
        hspace=0.38,
        wspace=0.18,
    )

    colors = plt.cm.plasma(np.linspace(0.22, 0.88, B))
    marker_styles = {
        "Boredom": ("^", "#00f5d4", "Boredom Side-Quest"),
        "Panic": ("s", "#ffbe0b", "Deadline Panic"),
        "Spite": ("*", "#ff006e", "Spite Taunt (Stochastic)"),
        "Manual": ("D", "#8338ec", "Manual Shock"),
    }

    ax_u, ax_xi = axes[0, 0], axes[0, 1]
    ax_v, ax_phase = axes[1, 0], axes[1, 1]

    lines_u, lines_xi, lines_v, lines_phase = [], [], [], []
    heads_u, heads_xi, heads_v, heads_phase = [], [], [], []

    for b in range(B):
        c = colors[b]
        # Faint full-trajectory ghost traces
        ax_u.plot(t[:, b], u[:, b], color=c, lw=1.0, alpha=0.22)
        ax_xi.plot(t[:, b], xi[:, b], color=c, lw=1.0, alpha=0.22)
        ax_v.plot(t[:, b], v[:, b], color=c, lw=1.0, alpha=0.22)
        ax_phase.plot(xi[:, b], u[:, b], color=c, lw=1.0, alpha=0.22)

        # Active foreground curves
        (lu,) = ax_u.plot(t[:, b], u[:, b], color=c, lw=2.4, label=res["names"][b])
        (lxi,) = ax_xi.plot(t[:, b], xi[:, b], color=c, lw=2.4)
        (lv,) = ax_v.plot(t[:, b], v[:, b], color=c, lw=2.4)
        (lph,) = ax_phase.plot(xi[:, b], u[:, b], color=c, lw=2.4)
        lines_u.append(lu)
        lines_xi.append(lxi)
        lines_v.append(lv)
        lines_phase.append(lph)

        # Start marker (o) & live moving state head
        ax_phase.scatter(xi[0, b], u[0, b], color=c, edgecolor="white", s=45, zorder=5)
        heads_u.append(ax_u.scatter([t[-1, b]], [u[-1, b]], color=c, edgecolor="white", s=55, zorder=8))
        heads_xi.append(ax_xi.scatter([t[-1, b]], [xi[-1, b]], color=c, edgecolor="white", s=55, zorder=8))
        heads_v.append(ax_v.scatter([t[-1, b]], [v[-1, b]], color=c, edgecolor="white", s=55, zorder=8))
        heads_phase.append(
            ax_phase.scatter([xi[-1, b]], [u[-1, b]], color=c, marker="X", edgecolor="white", s=80, zorder=8)
        )

    # Panel 1: Domain Mastery u(t)
    ax_u.set_title("1. Normalized Domain Mastery u(t) = L/D", pad=8)
    ax_u.set_xlabel("Subjective Proper Time (t)", labelpad=4)
    ax_u.set_ylabel("Domain Mastery (u)")
    ax_u.set_xlim(-0.5, np.max(t) * 1.03)
    ax_u.set_ylim(-0.04, 1.06)
    ax_u.grid(alpha=0.2, linestyle="--")

    # Panel 2: Active Scope xi(t)
    ax_xi.axhline(alpha, color="#ff595e", ls=":", lw=1.5, label=f"Axiom Floor α = {alpha:.2f}")
    ax_xi.axhline(1.0, color="#8ac926", ls="--", lw=1.2, label="Domain Ceiling (1.0)")
    ax_xi.set_title("2. Active Scope Expansion ξ(t) = I/D", pad=8)
    ax_xi.set_xlabel("Subjective Proper Time (t)", labelpad=4)
    ax_xi.set_ylabel("Normalized Scope (ξ)")
    ax_xi.set_xlim(-0.5, np.max(t) * 1.03)
    ax_xi.set_ylim(alpha - 0.08, 1.06)
    ax_xi.grid(alpha=0.2, linestyle="--")
    ax_xi.legend(loc="lower right", fontsize=8)

    # Panel 3: Bandwidth v(t) & Brick Mode Zone
    ax_v.axhline(alpha, color="#ff595e", ls="--", lw=1.8, label=f"Ignition Barrier (v = α = {alpha:.2f})")
    ax_v.fill_between([0, np.max(t)], 0, alpha, color="#ff595e", alpha=0.10, label="Brick Mode Zone (v < α)")
    ax_v.set_title("3. Ingestion Bandwidth v(t) = V/D vs. Ignition Barrier", pad=8)
    ax_v.set_xlabel("Subjective Proper Time (t)", labelpad=4)
    ax_v.set_ylabel("Normalized Bandwidth (v)")
    ax_v.set_xlim(-0.5, np.max(t) * 1.03)
    ax_v.set_ylim(0.0, 1.06)
    ax_v.grid(alpha=0.2, linestyle="--")
    ax_v.legend(loc="lower right", fontsize=8)

    # Panel 4: Phase Portrait in (xi, u) space
    diag = np.linspace(alpha, 1.0, 100)
    ax_phase.plot(diag, diag, color="white", ls="--", alpha=0.5, lw=1.3, label="100% Local Mastery (u = ξ)")
    ax_phase.fill_between(diag, diag, 1.0, color="gray", alpha=0.12)
    ax_phase.set_title("4. Phase Portrait: Scope (ξ) vs. Mastery (u)", pad=8)
    ax_phase.set_xlabel("Normalized Scope (ξ)", labelpad=4)
    ax_phase.set_ylabel("Domain Mastery (u)")
    ax_phase.set_xlim(alpha - 0.04, 1.04)
    ax_phase.set_ylim(-0.04, 1.06)
    ax_phase.grid(alpha=0.2, linestyle="--")

    # Shock markers across all 4 panels
    plotted_labels = set()
    marker_artists: list[tuple[int, list]] = []
    for step_idx, b, ev_type in res["markers"]:
        mk, col, lbl = marker_styles[ev_type]
        show_lbl = lbl if lbl not in plotted_labels else None
        if show_lbl:
            plotted_labels.add(lbl)
        sz = 135 if ev_type == "Spite" else 75
        m1 = ax_u.scatter(t[step_idx, b], u[step_idx, b], marker=mk, color=col, s=sz, zorder=7, label=show_lbl)
        m2 = ax_xi.scatter(t[step_idx, b], xi[step_idx, b], marker=mk, color=col, s=sz, zorder=7)
        m3 = ax_v.scatter(t[step_idx, b], v[step_idx, b], marker=mk, color=col, s=sz, zorder=7)
        m4 = ax_phase.scatter(xi[step_idx, b], u[step_idx, b], marker=mk, color=col, s=sz, zorder=7)
        marker_artists.append((step_idx, [m1, m2, m3, m4]))

    ax_u.legend(loc="lower right", fontsize=7.5, framealpha=0.82)
    ax_phase.legend(loc="upper left", fontsize=8)

    if interactive:
        status_text = fig.text(
            0.50,
            0.922,
            "",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#e0fbfc",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#1b263b", edgecolor="#415a77", alpha=0.92),
        )

        def render_step(s_idx: float) -> None:
            idx = int(np.clip(s_idx, 0, total_steps))
            for b in range(B):
                lines_u[b].set_data(t[: idx + 1, b], u[: idx + 1, b])
                lines_xi[b].set_data(t[: idx + 1, b], xi[: idx + 1, b])
                lines_v[b].set_data(t[: idx + 1, b], v[: idx + 1, b])
                lines_phase[b].set_data(xi[: idx + 1, b], u[: idx + 1, b])

                heads_u[b].set_offsets([[t[idx, b], u[idx, b]]])
                heads_xi[b].set_offsets([[t[idx, b], xi[idx, b]]])
                heads_v[b].set_offsets([[t[idx, b], v[idx, b]]])
                heads_phase[b].set_offsets([[xi[idx, b], u[idx, b]]])

            for m_step, artists in marker_artists:
                vis = m_step <= idx
                for art in artists:
                    art.set_visible(vis)

            summary = "  |  ".join(
                f"B{b}: u={u[idx, b]:.2f}, ξ={xi[idx, b]:.2f}, v={v[idx, b]:.2f}, n={n[idx, b]:.2f}"
                for b in range(B)
            )
            status_text.set_text(f"Step {idx:02d}/{total_steps:02d}  ::  {summary}")
            fig.canvas.draw_idle()

        ax_slider = fig.add_axes([0.16, 0.032, 0.62, 0.025], facecolor="#1b263b") # pyright: ignore[reportArgumentType]
        step_slider = Slider(
            ax=ax_slider,
            label="Step (s) ",
            valmin=0,
            valmax=total_steps,
            valinit=total_steps,
            valstep=1,
            color="#00f5d4",
        )
        step_slider.label.set_color("white")
        step_slider.valtext.set_color("#00f5d4")

        ax_btn = fig.add_axes([0.82, 0.029, 0.10, 0.032]) # pyright: ignore[reportArgumentType, reportCallIssue]
        play_btn = Button(ax_btn, "▶ Replay", color="#1b263b", hovercolor="#415a77")
        play_btn.label.set_color("white")

        is_playing = {"active": False}
        timer = fig.canvas.new_timer(interval=90)

        def on_tick() -> None:
            if not is_playing["active"]:
                return
            nxt = int(step_slider.val) + 1
            if nxt > total_steps:
                is_playing["active"] = False
                play_btn.label.set_text("▶ Replay")
                timer.stop()
                return
            step_slider.set_val(nxt)

        timer.add_callback(on_tick)

        def on_play_click(_event) -> None:
            if is_playing["active"]:
                is_playing["active"] = False
                play_btn.label.set_text("▶ Replay")
                timer.stop()
            else:
                if int(step_slider.val) >= total_steps:
                    step_slider.set_val(0)
                is_playing["active"] = True
                play_btn.label.set_text("❚❚ Pause")
                timer.start()

        step_slider.on_changed(render_step)
        play_btn.on_clicked(on_play_click)

        def on_hover(event) -> None:
            if event.inaxes not in (ax_u, ax_xi, ax_v) or event.xdata is None:
                return
            b_closest, s_closest, best_dist = 0, int(step_slider.val), float("inf")
            for b in range(B):
                idx = int(np.argmin(np.abs(t[:, b] - event.xdata)))
                dist = abs(t[idx, b] - event.xdata)
                if dist < best_dist:
                    best_dist = dist
                    b_closest, s_closest = b, idx
            status_text.set_text(
                f"Hover [Branch {b_closest} @ Step {s_closest:02d}, t={t[s_closest, b_closest]:.2f}]  ::  "
                f"u={u[s_closest, b_closest]:.3f} | ξ={xi[s_closest, b_closest]:.3f} | "
                f"v={v[s_closest, b_closest]:.3f} | n={n[s_closest, b_closest]:.3f}"
            )
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        # Keep widget references attached to fig so Python's GC does not kill callbacks
        fig._widgets = (step_slider, play_btn, timer) # pyright: ignore[reportAttributeAccessIssue]
        render_step(total_steps)

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    timelines = [
        TimelineSpec("Steady Grinder", tau=0.50, phi0=0.0, v0=0.52, n0=0.50, sigma=0.010, lam=0.002, k=0.04, p_tilde=0.03),
        TimelineSpec("Procrastinator", tau=0.35, phi0=0.0, v0=0.42, n0=0.50, sigma=0.035, lam=0.025, k=0.05, p_tilde=0.04),
        TimelineSpec("Chill Explorer", tau=0.45, phi0=np.pi/6, v0=0.45, n0=0.55, sigma=0.015, lam=0.030, k=0.07, p_tilde=0.06),
    ]

    res = simulate_multiverse_auto(
        steps=32,
        timelines=timelines,
        Deadline_Panic=True,
        Boredom_Side_Quest=True,
        Spite_Taunt=True,
        taunt_rate=0.07,
        t_exam=10.5,
        seed=7,
    )

    for line in res["event_log"]:
        print(line)
    plot_multiverse_diagnostics(res)