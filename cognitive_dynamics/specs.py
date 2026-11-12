from dataclasses import dataclass


@dataclass
class TimelineSpec:
    """Behavioral, meta-learning, and circadian parameters for a timeline branch."""
    name: str
    tau: float
    phi0: float
    v0: float
    n0: float
    sigma: float
    lam: float
    k: float
    p_tilde: float
    # Circadian & cognitive energy parameters
    is_night_owl: bool = True
    phase_shift_hours: float = 8.0   # Night owl (sin^2) peaks at 20:00; Early bird (cos^2) at 08:00
    e_base: float = 0.25             # Baseline cognitive energy reserve (0..1)
    k_energy: float = 0.45           # Post-spike decay rate
    k_fatigue: float = 0.32          # Burnout buildup rate when awake > 0.5
    k_recovery: float = 0.22         # Fatigue recovery rate when awake < 0.5
    t_peak: float = 2.5              # Peak hold duration (hours) for adrenaline/spite spikes


@dataclass
class DomainSpec:
    """Structural Fourier parameters and exam constraints for a knowledge domain."""
    name: str
    base_amount: float = 20.0
    complexity: float = 1.5
    n_harmonics: int = 3
    t_exam: float = 12.0
    u_pass: float = 0.60
    v_offset: float = 0.0            # Domain-specific initial bandwidth shift


@dataclass
class ShockEvent:
    """Optional manual impulse/regime intervention at a specific step and branch."""
    step: int
    branch: int
    domain: int | None = None        # None applies shock across all domains on that branch
    new_regime: TimelineSpec | None = None
    dv: float = 0.0
    dxi: float = 0.0
    dn: float = 0.0
    de: float = 0.0
    label: str = "Manual Shock"