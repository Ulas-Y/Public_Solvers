import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def plot_interactive_run(
    res: dict, run_idx: int = 0, domain_idx: int = 0, save_path: str | None = None, interactive: bool = True
) -> None:
    """Interactive step-scrubber and phase portrait for a single realization & domain."""
    t = res["t"][:, run_idx, :, domain_idx]
    u = res["u"][:, run_idx, :, domain_idx]
    xi = res["xi"][:, run_idx, :, domain_idx]
    v = res["v_eff"][:, run_idx, :, domain_idx]
    n = res["n"][:, run_idx, :, domain_idx]
    alpha = res["bounds"][domain_idx, 2]
    dom_name = res["domain_names"][domain_idx]
    tl_names = res["timeline_names"]
    total_steps, B = u.shape[0] - 1, u.shape[1]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6), dpi=115)
    fig.suptitle(f"Phase Diagnostics — {dom_name} (Run #{run_idx})", fontsize=14, fontweight="bold", y=0.972)
    fig.subplots_adjust(
        top=0.865, bottom=0.125 if interactive else 0.07, left=0.06, right=0.98, hspace=0.38, wspace=0.18
    )

    colors = plt.cm.plasma(np.linspace(0.22, 0.88, B))
    marker_styles = {
        "Boredom": ("^", "#00f5d4", "Boredom Side-Quest"),
        "Panic": ("s", "#ffbe0b", "Deadline Panic"),
        "Spite": ("*", "#ff006e", "Spite Taunt (Stochastic)"),
        "Manual": ("D", "#8338ec", "Manual Shock"),
    }

    ax_u, ax_xi, ax_v, ax_phase = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    lines_u, lines_xi, lines_v, lines_phase = [], [], [], []
    heads_u, heads_xi, heads_v, heads_phase = [], [], [], []

    for b in range(B):
        c = colors[b]
        ax_u.plot(t[:, b], u[:, b], color=c, lw=1.0, alpha=0.22)
        ax_xi.plot(t[:, b], xi[:, b], color=c, lw=1.0, alpha=0.22)
        ax_v.plot(t[:, b], v[:, b], color=c, lw=1.0, alpha=0.22)
        ax_phase.plot(xi[:, b], u[:, b], color=c, lw=1.0, alpha=0.22)

        (lu,) = ax_u.plot(t[:, b], u[:, b], color=c, lw=2.3, label=tl_names[b])
        (lxi,) = ax_xi.plot(t[:, b], xi[:, b], color=c, lw=2.3)
        (lv,) = ax_v.plot(t[:, b], v[:, b], color=c, lw=2.3)
        (lph,) = ax_phase.plot(xi[:, b], u[:, b], color=c, lw=2.3)
        lines_u.append(lu); lines_xi.append(lxi); lines_v.append(lv); lines_phase.append(lph)

        ax_phase.scatter(xi[0, b], u[0, b], color=c, edgecolor="white", s=45, zorder=5)
        heads_u.append(ax_u.scatter([t[-1, b]], [u[-1, b]], color=c, edgecolor="white", s=55, zorder=8))
        heads_xi.append(ax_xi.scatter([t[-1, b]], [xi[-1, b]], color=c, edgecolor="white", s=55, zorder=8))
        heads_v.append(ax_v.scatter([t[-1, b]], [v[-1, b]], color=c, edgecolor="white", s=55, zorder=8))
        heads_phase.append(ax_phase.scatter([xi[-1, b]], [u[-1, b]], color=c, marker="X", edgecolor="white", s=80, zorder=8))

    ax_u.set_title("1. Normalized Domain Mastery u(t) = L/D", pad=8)
    ax_u.set_xlabel("Subjective Proper Time (t)"); ax_u.set_ylabel("Domain Mastery (u)")
    ax_u.set_xlim(-0.5, np.max(t) * 1.03); ax_u.set_ylim(-0.04, 1.06); ax_u.grid(alpha=0.2, linestyle="--")

    ax_xi.axhline(alpha, color="#ff595e", ls=":", lw=1.5, label=f"Axiom Floor α = {alpha:.2f}")
    ax_xi.axhline(1.0, color="#8ac926", ls="--", lw=1.2, label="Domain Ceiling (1.0)")
    ax_xi.set_title("2. Active Scope Expansion ξ(t) = I/D", pad=8)
    ax_xi.set_xlabel("Subjective Proper Time (t)"); ax_xi.set_ylabel("Normalized Scope (ξ)")
    ax_xi.set_xlim(-0.5, np.max(t) * 1.03); ax_xi.set_ylim(alpha - 0.08, 1.06)
    ax_xi.grid(alpha=0.2, linestyle="--"); ax_xi.legend(loc="lower right", fontsize=8)

    ax_v.axhline(alpha, color="#ff595e", ls="--", lw=1.8, label=f"Base Barrier α_0 = {alpha:.2f}")
    ax_v.fill_between([0, np.max(t)], 0, alpha, color="#ff595e", alpha=0.10, label="Brick Mode Zone (v < α_0)")
    ax_v.set_title("3. Effective Bandwidth v_eff(t) vs. Ignition Barrier", pad=8)
    ax_v.set_xlabel("Subjective Proper Time (t)"); ax_v.set_ylabel("Normalized Bandwidth (v)")
    ax_v.set_xlim(-0.5, np.max(t) * 1.03); ax_v.set_ylim(0.0, 1.06)
    ax_v.grid(alpha=0.2, linestyle="--"); ax_v.legend(loc="lower right", fontsize=8)

    diag = np.linspace(alpha, 1.0, 100)
    ax_phase.plot(diag, diag, color="white", ls="--", alpha=0.5, lw=1.3, label="100% Local Mastery (u = ξ)")
    ax_phase.fill_between(diag, diag, 1.0, color="gray", alpha=0.12)
    ax_phase.set_title("4. Phase Portrait: Scope (ξ) vs. Mastery (u)", pad=8)
    ax_phase.set_xlabel("Normalized Scope (ξ)"); ax_phase.set_ylabel("Domain Mastery (u)")
    ax_phase.set_xlim(alpha - 0.04, 1.04); ax_phase.set_ylim(-0.04, 1.06); ax_phase.grid(alpha=0.2, linestyle="--")

    plotted_labels = set()
    marker_artists: list[tuple[int, list]] = []
    if run_idx == 0:
        for step_idx, b, g, ev_type in res.get("run0_markers", []):
            if g != domain_idx:
                continue
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
            0.50, 0.922, "", ha="center", va="center", fontsize=9.5, color="#e0fbfc",
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
                for art in artists:
                    art.set_visible(m_step <= idx)
            summary = "  |  ".join(
                f"B{b}: u={u[idx, b]:.2f}, ξ={xi[idx, b]:.2f}, v={v[idx, b]:.2f}, n={n[idx, b]:.2f}"
                for b in range(B)
            )
            status_text.set_text(f"Step {idx:02d}/{total_steps:02d}  ::  {summary}")
            fig.canvas.draw_idle()

        ax_slider = fig.add_axes([0.16, 0.032, 0.62, 0.025], facecolor="#1b263b")
        step_slider = Slider(ax_slider, "Step (s) ", 0, total_steps, valinit=total_steps, valstep=1, color="#00f5d4")
        step_slider.label.set_color("white"); step_slider.valtext.set_color("#00f5d4")

        ax_btn = fig.add_axes([0.82, 0.029, 0.10, 0.032])
        play_btn = Button(ax_btn, "▶ Replay", color="#1b263b", hovercolor="#415a77")
        play_btn.label.set_color("white")

        is_playing = {"active": False}
        timer = fig.canvas.new_timer(interval=90)

        def on_tick() -> None:
            if not is_playing["active"]:
                return
            nxt = int(step_slider.val) + 1
            if nxt > total_steps:
                is_playing["active"] = False; play_btn.label.set_text("▶ Replay"); timer.stop()
                return
            step_slider.set_val(nxt)

        timer.add_callback(on_tick)

        def on_play_click(_event) -> None:
            if is_playing["active"]:
                is_playing["active"] = False; play_btn.label.set_text("▶ Replay"); timer.stop()
            else:
                if int(step_slider.val) >= total_steps:
                    step_slider.set_val(0)
                is_playing["active"] = True; play_btn.label.set_text("❚❚ Pause"); timer.start()

        step_slider.on_changed(render_step)
        play_btn.on_clicked(on_play_click)
        fig._widgets = (step_slider, play_btn, timer)
        render_step(total_steps)

    if save_path:
        plt.savefig(save_path, dpi=130)
    plt.show()


def plot_multiverse_4d(res: dict, save_path: str | None = None) -> None:
    """4D ensemble overview across all timelines and coupled knowledge domains."""
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
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.06, right=0.94, hspace=0.36, wspace=0.27)

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
    ax_u.set_xlabel("Median Proper Time (t)"); ax_u.set_ylabel("Domain Mastery (u)")
    ax_u.set_ylim(-0.04, 1.06); ax_u.grid(alpha=0.2, linestyle="--")
    ax_u.legend(loc="lower right", fontsize=7.2, ncol=2)

    ax_barrier.set_title("2. Effective Bandwidth v_eff(t) vs. Lowered Barrier α_eff(t)", pad=8)
    ax_barrier.set_xlabel("Median Proper Time (t)"); ax_barrier.set_ylabel("Normalized Rate / Barrier")
    ax_barrier.set_ylim(0.0, 1.06); ax_barrier.grid(alpha=0.2, linestyle="--")
    ax_barrier.legend(loc="lower right", fontsize=8)

    ax_xi.axhline(1.0, color="#8ac926", ls="--", lw=1.2, label="Domain Ceiling (1.0)")
    ax_xi.set_title("3. Active Scope Expansion ξ(t) across Domains", pad=8)
    ax_xi.set_xlabel("Median Proper Time (t)"); ax_xi.set_ylabel("Normalized Scope (ξ)")
    ax_xi.set_ylim(0.35, 1.06); ax_xi.grid(alpha=0.2, linestyle="--")
    ax_xi.legend(loc="lower right", fontsize=8)

    pass_mat = res["stats"]["pass_matrix"] * 100.0
    im = ax_heat.imshow(pass_mat, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax_heat.set_title("4. Exam Pass Probability Matrix (%)", pad=8)
    ax_heat.set_xticks(np.arange(G)); ax_heat.set_xticklabels(dom_names, fontsize=9)
    ax_heat.set_yticks(np.arange(B)); ax_heat.set_yticklabels(tl_names, fontsize=9)
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


def plot_energy_burnout(res: dict, domain_idx: int = 0, save_path: str | None = None) -> None:
    """Circadian awakeness, energy spikes vs. fatigue buildup, and burnout plateau diagnostics."""
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
    ax_circ.set_xlabel("Time of Day / Proper Time (hours)"); ax_circ.set_ylabel("Awakeness Coefficient")
    ax_circ.set_ylim(-0.04, 1.06); ax_circ.grid(alpha=0.2, linestyle="--"); ax_circ.legend(loc="upper right", fontsize=8)

    ax_ef.set_title("2. Energy Spikes (Solid) vs. Fatigue Buildup (Dashed)", pad=8)
    ax_ef.set_xlabel("Time (hours)"); ax_ef.set_ylabel("Normalized Level (0–1)")
    ax_ef.set_ylim(-0.04, 1.06); ax_ef.grid(alpha=0.2, linestyle="--"); ax_ef.legend(loc="upper right", fontsize=7.5, ncol=2)

    ax_eta.set_title("3. Total Cognitive Efficiency η_cog = awake · max(0, E − F)", pad=8)
    ax_eta.set_xlabel("Time (hours)"); ax_eta.set_ylabel("Effective Multiplier η_cog")
    ax_eta.set_ylim(-0.04, 1.06); ax_eta.grid(alpha=0.2, linestyle="--"); ax_eta.legend(loc="upper right", fontsize=8)

    ax_u.axvline(res["t_exam"][domain_idx], color="#ffbe0b", ls="--", lw=1.4, label=f"Exam (t={res['t_exam'][domain_idx]}h)")
    ax_u.axhline(res["u_pass"][domain_idx], color="#8ac926", ls=":", lw=1.3, label="Pass Floor (60%)")
    ax_u.set_title("4. Domain Mastery u(t) Under Burnout Throttling", pad=8)
    ax_u.set_xlabel("Time (hours)"); ax_u.set_ylabel("Domain Mastery (u)")
    ax_u.set_ylim(-0.04, 1.06); ax_u.grid(alpha=0.2, linestyle="--"); ax_u.legend(loc="lower right", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()