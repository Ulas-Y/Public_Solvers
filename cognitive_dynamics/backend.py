import warnings
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def get_backend_info() -> dict:
    """Reports PyTorch/CUDA availability and the active default backend."""
    torch_ok = torch is not None
    cuda_ok = bool(torch_ok and torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    return {
        "torch_available": torch_ok,
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "default_backend": "torch" if cuda_ok else "numpy",
        "default_device": "cuda" if cuda_ok else "cpu",
    }


def resolve_backend(backend: str = "auto", device: str | None = None) -> tuple[str, str]:
    """
    Resolves ('numpy' | 'torch', device_str) with automatic fallback:
      - 'auto'  -> ('torch', 'cuda') if PyTorch + CUDA are available, else ('numpy', 'cpu').
      - 'torch' -> uses requested device ('cuda'/'cpu'/'mps') or auto-detects, falling back to 'numpy' if Torch is missing.
      - 'numpy' -> ('numpy', 'cpu').
    """
    mode = backend.lower().strip()

    if mode == "auto":
        if torch is not None and torch.cuda.is_available():
            return "torch", (device or "cuda")
        return "numpy", "cpu"

    if mode == "torch":
        if torch is None:
            warnings.warn("PyTorch is not installed; falling back to NumPy CPU backend.", RuntimeWarning)
            return "numpy", "cpu"
        target_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if target_dev.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA not available; falling back to PyTorch CPU execution.", RuntimeWarning)
            target_dev = "cpu"
        return "torch", target_dev

    if mode == "numpy":
        return "numpy", "cpu"

    raise ValueError(f"Unsupported backend {backend!r}. Expected 'auto', 'numpy', or 'torch'.")


class TensorBackend:
    """Unified array/tensor dispatch adapter so engine.py stays single-source."""

    def __init__(self, backend: str = "auto", device: str | None = None, seed: int | None = 42, dtype: str | None = None):
        self.name, self.device = resolve_backend(backend, device)
        self.is_torch = self.name == "torch"

        if self.is_torch:
            self.torch_dtype = torch.float64 if dtype == "float64" else torch.float32
            self.np_dtype = np.float64 if dtype == "float64" else np.float32
            self._gen = torch.Generator(device=self.device)
            if seed is not None:
                self._gen.manual_seed(seed)
        else:
            self.np_dtype = np.float32 if dtype == "float32" else np.float64
            self._rng = np.random.default_rng(seed)

    def as_tensor(self, arr, is_bool: bool = False):
        if self.is_torch:
            dt = torch.bool if is_bool else self.torch_dtype
            return torch.as_tensor(arr, device=self.device, dtype=dt)
        return np.asarray(arr, dtype=bool if is_bool else self.np_dtype)

    def zeros(self, shape: tuple[int, ...], is_bool: bool = False):
        if self.is_torch:
            dt = torch.bool if is_bool else self.torch_dtype
            return torch.zeros(shape, device=self.device, dtype=dt)
        return np.zeros(shape, dtype=bool if is_bool else self.np_dtype)

    def broadcast_timeline(self, values: list, R: int, B: int, G: int, is_bool: bool = False):
        t_col = self.as_tensor(values, is_bool=is_bool).reshape(1, B, 1)
        if self.is_torch:
            return t_col.expand(R, B, G).clone()
        return np.broadcast_to(t_col, (R, B, G)).copy()

    def rand(self, shape: tuple[int, ...]):
        if self.is_torch:
            return torch.rand(shape, generator=self._gen, device=self.device, dtype=self.torch_dtype)
        return self._rng.random(shape).astype(self.np_dtype)

    def randn(self, shape: tuple[int, ...]):
        if self.is_torch:
            return torch.randn(shape, generator=self._gen, device=self.device, dtype=self.torch_dtype)
        return self._rng.normal(0.0, 1.0, size=shape).astype(self.np_dtype)

    def clamp(self, x, lo, hi):
        if self.is_torch:
            lo_t = torch.as_tensor(lo, device=self.device, dtype=x.dtype)
            hi_t = torch.as_tensor(hi, device=self.device, dtype=x.dtype)
            return torch.clamp(x, min=lo_t, max=hi_t)
        return np.clip(x, lo, hi)

    def where(self, cond, a, b):
        if self.is_torch:
            a_t = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, device=self.device, dtype=self.torch_dtype)
            b_t = b if isinstance(b, torch.Tensor) else torch.as_tensor(b, device=self.device, dtype=self.torch_dtype)
            return torch.where(cond, a_t, b_t)
        return np.where(cond, a, b)

    def minimum(self, a, b):
        if self.is_torch:
            a_t = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, device=self.device, dtype=self.torch_dtype)
            b_t = b if isinstance(b, torch.Tensor) else torch.as_tensor(b, device=self.device, dtype=self.torch_dtype)
            return torch.minimum(a_t, b_t)
        return np.minimum(a, b)

    def maximum(self, a, b):
        if self.is_torch:
            a_t = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, device=self.device, dtype=self.torch_dtype)
            b_t = b if isinstance(b, torch.Tensor) else torch.as_tensor(b, device=self.device, dtype=self.torch_dtype)
            return torch.maximum(a_t, b_t)
        return np.maximum(a, b)

    def exp(self, x):
        return torch.exp(x) if self.is_torch else np.exp(x)

    def sin(self, x):
        return torch.sin(x) if self.is_torch else np.sin(x)

    def cos(self, x):
        return torch.cos(x) if self.is_torch else np.cos(x)

    def einsum(self, eq: str, a, b):
        return torch.einsum(eq, a, b) if self.is_torch else np.einsum(eq, a, b)

    def any(self, x) -> bool:
        return bool(torch.any(x).item()) if self.is_torch else bool(np.any(x))

    def interp_exam(self, t_exam_vec, t, u):
        """Vectorized linear interpolation of u(t) at t_exam_vec across all (R, B, G)."""
        S = t.shape[0]
        if self.is_torch:
            target = t_exam_vec.view(1, 1, -1).expand_as(t[0])
            idx_hi = torch.clamp((t <= target.unsqueeze(0)).sum(dim=0), min=1, max=S - 1)
            idx_lo = idx_hi - 1
            t_lo = torch.gather(t, 0, idx_lo.unsqueeze(0)).squeeze(0)
            t_hi = torch.gather(t, 0, idx_hi.unsqueeze(0)).squeeze(0)
            u_lo = torch.gather(u, 0, idx_lo.unsqueeze(0)).squeeze(0)
            u_hi = torch.gather(u, 0, idx_hi.unsqueeze(0)).squeeze(0)
            w = torch.clamp((target - t_lo) / torch.clamp(t_hi - t_lo, min=1e-12), min=0.0, max=1.0)
            return u_lo + w * (u_hi - u_lo)

        target = np.broadcast_to(t_exam_vec[None, None, :], t.shape[1:])
        idx_hi = np.clip(np.sum(t <= target[None, ...], axis=0), 1, S - 1)
        idx_lo = idx_hi - 1
        t_lo = np.take_along_axis(t, idx_lo[None, ...], axis=0)[0]
        t_hi = np.take_along_axis(t, idx_hi[None, ...], axis=0)[0]
        u_lo = np.take_along_axis(u, idx_lo[None, ...], axis=0)[0]
        u_hi = np.take_along_axis(u, idx_hi[None, ...], axis=0)[0]
        w = np.clip((target - t_lo) / np.maximum(t_hi - t_lo, 1e-12), 0.0, 1.0)
        return u_lo + w * (u_hi - u_lo)

    def to_numpy(self, x) -> np.ndarray:
        if self.is_torch and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)