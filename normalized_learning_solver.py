def simulate_normalized(
    steps: int,
    tau: float = 0.5,
    u0: float = 0.0,
    xi0: float = 0.10,
    v0: float = 0.12,
    n0: float = 0.55,
    alpha: float = 0.10,
    sigma: float = 0.005,
    lam: float = 0.08,
    k: float = 0.15,
    p_tilde: float = 0.20,  # p / D
) -> dict[str, np.ndarray]:
    t = np.arange(steps + 1) * tau
    u = np.zeros(steps + 1)
    xi = np.zeros(steps + 1)
    v = np.zeros(steps + 1)
    n = np.zeros(steps + 1)

    u[0], xi[0], v[0], n[0] = u0, xi0, v0, n0

    for step in range(steps):
        # 1. Ingestion step (zero if v < alpha)
        if v[step] < alpha:
            u_gain = 0.0
        else:
            u_gain = (
                min(v[step], xi[step])
                * n[step]
                * max(0.0, 1.0 - u[step] / xi[step])
            )

        # 2. Retention & decay
        u_next = (u[step] + u_gain) * np.exp(-sigma * tau)
        local_mastery = u_next / xi[step]

        # 3. Meta-learning updates (n, v) and clamped scope creep (xi)
        n_next = min(1.0, n[step] * np.exp(k * local_mastery * tau))
        v_next = min(1.0, v[step] * np.exp(p_tilde * local_mastery * tau))
        xi_next = max(alpha, min(1.0, xi[step] * np.exp(lam * tau)))

        u[step + 1] = u_next
        xi[step + 1] = xi_next
        v[step + 1] = v_next
        n[step + 1] = n_next

    return {"t": t, "u": u, "xi": xi, "v": v, "n": n}