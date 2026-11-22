"""Module A adapter over the shared ``StructuredSequenceEncoder``.

The encoder (``qss.classical.encoder``) owns the sequence-to-patch contract for both modules.
Module A consumes its per-scale ``amplitude`` and ``phase`` tensors, flattened to items.
Item order is patch-major: item = patch * C + channel.
"""

from __future__ import annotations

import numpy as np

from qss.classical.encoder import StructuredEncoderConfig, StructuredSequenceEncoder


def encode_items(train_series: np.ndarray, series: np.ndarray, period: int, patch_len: int
                 ) -> tuple[np.ndarray, np.ndarray, StructuredSequenceEncoder]:
    """Fit the shared encoder on ``train_series`` only, then encode ``series``.

    Returns amplitude and phase of shape (N, n_patches * C). Amplitudes are in units of the
    training-set input standard deviation (the encoder normalizes the series first).
    """
    encoder = StructuredSequenceEncoder(StructuredEncoderConfig(patch_lengths=(patch_len,), period=period))
    encoder.fit(train_series)
    scale = encoder.transform(series).scales[patch_len]
    n = len(series)
    return scale.amplitude.reshape(n, -1), scale.phase.reshape(n, -1), encoder


def fit_amp_max(train_amp: np.ndarray, margin: float = 1.05) -> float:
    """Amplitude scale for the RY encoding, set from training data (larger values are clipped)."""
    return float(margin * np.max(train_amp))


def amp_to_theta(amp: np.ndarray, amp_max: float) -> np.ndarray:
    """RY angle with sin(theta) = A / amp_max, so the Bloch x-y radius is proportional to A."""
    return np.arcsin(np.clip(amp / amp_max, 0.0, 1.0))
