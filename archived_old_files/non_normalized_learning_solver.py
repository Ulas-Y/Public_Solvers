import numpy as np


def simulate_non_normalized(
    steps: int,
    tau: float = 0.5,
    L0: float = 0.0,
    I0: float = 10.0,
    V0: float = 12.0,
    n0: float = 0.55,
    M: float = 10.0,
    D: float = 100.0,
    sigma: float = 0.005,
    lam: float = 0.08,
    k: float = 0.15,
    p: float = 20.0,
) -> dict[str, np.ndarray]:
    t = np.arange(steps + 1) * tau
    L = np.zeros(steps + 1)
    I = np.zeros(steps + 1)
    V = np.zeros(steps + 1)
    n = np.zeros(steps + 1)

    L[0], I[0], V[0], n[0] = L0, I0, V0, n0

    for step in range(steps):
        # 1. Ingestion step (zero if bandwidth is below axiomatic floor M)
        if V[step] < M:
            L_gain = 0.0
        else:
            L_gain = (
                min(V[step], I[step])
                * n[step]
                * max(0.0, 1.0 - L[step] / I[step])
            )

        # 2. Retention & decay
        L_next = (L[step] + L_gain) * np.exp(-sigma * tau)
        local_mastery = L_next / I[step]

        # 3. Meta-learning updates (n, V) and clamped scope creep (I)
        n_next = min(1.0, n[step] * np.exp(k * local_mastery * tau))
        V_next = min(D, V[step] * np.exp(p * (L_next / (I[step] * D)) * tau))
        I_next = max(M, min(D, I[step] * np.exp(lam * tau)))

        L[step + 1] = L_next
        I[step + 1] = I_next
        V[step + 1] = V_next
        n[step + 1] = n_next

    return {"t": t, "L": L, "I": I, "V": V, "n": n}