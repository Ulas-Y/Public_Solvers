from dataclasses import dataclass
import numpy as np
from information_volume_computing import compute_info_volume
from information_volume_computing import estimate_domain_bounds
from timeline_solver import TimelineSpec

@dataclass
class ShockEvent:
    step: int
    branch: int                              # Timeline index to perturb
    new_regime: TimelineSpec | None = None   # Optional parameter swap
    dv: float = 0.0                          # Instantaneous bandwidth kick
    dxi: float = 0.0                         # Instantaneous scope shift (<0 = axiom clamp)
    dn: float = 0.0                          # Instantaneous efficiency boost
    label: str = "Shock"


def simulate_multiverse_with_shocks(
    steps: int,
    timelines: list[TimelineSpec],
    shocks: list[ShockEvent] | None = None,
    base_amount: float = 20.0,
    complexity: float = 1.5,
    n_harmonics: int = 3,
    coupling_J: np.ndarray | None = None,
) -> dict[str, np.ndarray | list[str] | list[list[str]]]:
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
    regime_history = [list(current_names)]

    for s in range(steps):
        # 0. Apply scheduled shocks / regime switches at the start of step s
        if s in shock_map:
            for sh in shock_map[s]:
                b = sh.branch
                v[s, b] = np.clip(v[s, b] + sh.dv, 0.0, 1.0)
                xi[s, b] = np.clip(xi[s, b] + sh.dxi, alpha, 1.0)
                n[s, b] = np.clip(n[s, b] + sh.dn, 0.0, 1.0)
                if sh.new_regime is not None:
                    nr = sh.new_regime
                    tau[b] = nr.tau
                    sigma[b] = nr.sigma
                    lam[b] = nr.lam
                    k[b] = nr.k
                    p_tilde[b] = nr.p_tilde
                    current_names[b] = f"{nr.name} [{sh.label}]"

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
        regime_history.append(list(current_names))

    return {
        "names": current_names,
        "regime_history": regime_history,
        "bounds": np.array([M, D, alpha]),
        "t": t,
        "u": u,
        "xi": xi,
        "v": v,
        "n": n,
    }