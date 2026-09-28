"""Device selection helpers."""

import torch


def _mps_is_usable():
    """Return True only if PyTorch can actually allocate on MPS."""
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        return False
    try:
        _ = torch.zeros(1, device="mps")
        return True
    except RuntimeError:
        return False


def resolve_device(device=None):
    """Prefer MPS when available, then CUDA, then CPU."""
    if device not in (None, "auto"):
        return device

    if _mps_is_usable():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
