"""Classical patch feature extractor: amplitude and phase of the dominant harmonic."""

from __future__ import annotations

import numpy as np


def amp_phase_features(series: np.ndarray, period: int, patch_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Single-bin DFT per patch and channel.

    For x(t) = A cos(w t + phi) over an integer number of periods,
    (2/L) sum_t x(t) e^{-i w t} = A e^{i phi}.

    Returns amplitudes and phases of shape (N, n_items) where items are ordered
    channel-major within each patch: item = patch * C + channel.
    """
    n, length, channels = series.shape
    if length % patch_len or patch_len % period:
        raise ValueError("window must split into patches of whole periods")
    n_patches = length // patch_len
    t = np.arange(patch_len)
    basis = np.exp(-1j * 2 * np.pi * t / period)  # (patch_len,)
    patches = series.reshape(n, n_patches, patch_len, channels)
    coeff = (2.0 / patch_len) * np.einsum("npLc,L->npc", patches, basis)
    coeff = coeff.reshape(n, n_patches * channels)
    return np.abs(coeff), np.angle(coeff)


def amp_to_theta(amp: np.ndarray, amp_max: float) -> np.ndarray:
    """RY angle with sin(theta) = A / amp_max, so the Bloch x-y radius is proportional to A."""
    return np.arcsin(np.clip(amp / amp_max, 0.0, 1.0))
