import numpy as np


def compute_info_volume(
    base_amount: float,
    complexity: float,
    phi: float,
    n_harmonics: int = 3,
    a_coeffs: tuple[float, ...] | None = None,
    b_coeffs: tuple[float, ...] | None = None,
    decay_power: float = 1.0,
) -> float:
    """Computes Informational Volume I(phi) = A_0 * C * F_N(phi) via truncated Fourier coupling."""
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
    """Derives axiomatic floor M, domain ceiling D, and aspect ratio alpha = M / D."""
    phis = np.linspace(0.0, np.pi, 500)
    vols = [compute_info_volume(base_amount, complexity, p, n_harmonics, **kwargs) for p in phis]
    M = float(min(vols))
    D = float(max(vols))
    return M, D, M / D


def build_graph_laplacian(coupling_J: np.ndarray) -> np.ndarray:
    """
    Constructs graph Laplacian L = D_deg - J from adjacency matrix J (B x B).
    Safely passes through if a valid graph Laplacian is already provided.
    """
    M = np.asarray(coupling_J, dtype=float).copy()
    off_diag = M - np.diag(np.diag(M))
    if np.all(np.diag(M) >= 0.0) and np.all(off_diag <= 0.0) and np.allclose(np.sum(M, axis=1), 0.0):
        return M
    np.fill_diagonal(M, 0.0)
    D_deg = np.diag(np.sum(M, axis=1))
    return D_deg - M