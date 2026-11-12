import numpy as np
from timeline_solver import TimelineSpec
from shock_event_handler import ShockEvent
from information_volume_computing import compute_info_volume
from information_volume_computing import estimate_domain_bounds

def simulate_multiverse_auto(
    steps: int,
    timelines: list[TimelineSpec],
    shocks: list[ShockEvent] | None = None,
    Deadline_Panic: bool = True,
    Boredom_Side_Quest: bool = True,
    t_exam: float = 10.0,
    panic_window: float = 4.0,
    panic_mastery_floor: float = 0.45,
    boredom_threshold: float = 0.94,
    base_amount: float = 20.0,
    complexity: float = 1.5,
    n_harmonics: int = 3,
    coupling_J: np.ndarray | None = None,
) -> dict[str, np.ndarray | list[str]]:
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

    fired_panic = np.zeros(B, dtype=bool)
    fired_boredom = np.zeros(B, dtype=bool)

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
                event_log.append(
                    f"Step {s:02d} | Branch {b} ({timelines[b].name}): Manual [{sh.label}]"
                )

        # 0B. Endogenous state-triggered shocks
        for b in range(B):
            local_m = u[s, b] / xi[s, b]

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
                current_names[b] += " -> [Boredom_Side_Quest]"
                event_log.append(
                    f"Step {s:02d} (t={t[s, b]:.1f}) | Branch {b} ({timelines[b].name}): "
                    f"Boredom_Side_Quest (u/xi={local_m:.2f} >= {boredom_threshold})"
                )

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
                current_names[b] += " -> [Deadline_Panic]"
                event_log.append(
                    f"Step {s:02d} (t={t[s, b]:.1f}) | Branch {b} ({timelines[b].name}): "
                    f"Deadline_Panic (t_left={time_left:.1f}, u={u[s, b]:.2f})"
                )

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
        "bounds": np.array([M, D, alpha]),
        "t": t,
        "u": u,
        "xi": xi,
        "v": v,
        "n": n,
    }