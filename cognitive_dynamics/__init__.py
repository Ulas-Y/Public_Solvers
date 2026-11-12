from .specs import TimelineSpec, DomainSpec, ShockEvent
from .physics import compute_info_volume, estimate_domain_bounds, build_graph_laplacian
from .backend import get_backend_info, resolve_backend
from .engine import simulate
from .plotting import plot_interactive_run, plot_multiverse_4d, plot_energy_burnout

__all__ = [
    "TimelineSpec",
    "DomainSpec",
    "ShockEvent",
    "compute_info_volume",
    "estimate_domain_bounds",
    "build_graph_laplacian",
    "get_backend_info",
    "resolve_backend",
    "simulate",
    "plot_interactive_run",
    "plot_multiverse_4d",
    "plot_energy_burnout",
]