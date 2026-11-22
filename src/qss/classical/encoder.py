"""Shared classical structured encoder for Module A and Module C.

The encoder owns the sequence-to-patch contract.  It accepts a batch of
``[batch, length, channels]`` series and exposes, per scale, both interpretable
periodic quantities for Module A and normalized embeddings for Module C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuredEncoderConfig:
    """Configuration for fixed, non-overlapping multi-scale feature patches."""

    patch_lengths: tuple[int, ...] = (24, 48)
    period: int = 24
    stride: int | None = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not self.patch_lengths or any(length < 2 for length in self.patch_lengths):
            raise ValueError("patch_lengths must contain integers of at least 2")
        if self.period < 2:
            raise ValueError("period must be at least 2")
        if self.stride is not None and self.stride < 1:
            raise ValueError("stride must be positive when specified")


@dataclass(frozen=True)
class ScaleEncoding:
    """Features and metadata for a single patch scale.

    ``periodic`` values have shape ``[batch, patches, channels]``. ``features``
    and ``embedding`` have shape ``[batch, patches, channels * 8]``.
    """

    patch_length: int
    starts: np.ndarray
    feature_names: tuple[str, ...]
    features: np.ndarray
    embedding: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray


@dataclass(frozen=True)
class StructuredEncoding:
    """Complete shared output contract for one batch of sequences."""

    scales: dict[int, ScaleEncoding]
    input_mean: np.ndarray
    input_std: np.ndarray


class StructuredSequenceEncoder:
    """Fit-on-training-data structured sequence encoder.

    For each scale it extracts per-channel mean, standard deviation, slope,
    periodic amplitude, sine/cosine phase, dominant frequency, and residual
    standard deviation.  All raw features are then standardized using training
    patch statistics to form the retrieval embedding.
    """

    _BASE_FEATURE_NAMES = (
        "mean",
        "std",
        "slope",
        "amplitude",
        "sin_phase",
        "cos_phase",
        "dominant_frequency",
        "residual_std",
    )

    def __init__(self, config: StructuredEncoderConfig | None = None) -> None:
        self.config = config or StructuredEncoderConfig()
        self._input_mean: np.ndarray | None = None
        self._input_std: np.ndarray | None = None
        self._feature_mean: dict[int, np.ndarray] = {}
        self._feature_std: dict[int, np.ndarray] = {}

    @staticmethod
    def _validate(series: np.ndarray) -> np.ndarray:
        array = np.asarray(series, dtype=float)
        if array.ndim != 3:
            raise ValueError("series must have shape [batch, length, channels]")
        if array.shape[0] == 0 or array.shape[2] == 0:
            raise ValueError("series must include at least one sample and channel")
        if not np.isfinite(array).all():
            raise ValueError("series must not contain NaN or infinite values")
        return array

    def fit(self, series: np.ndarray) -> StructuredSequenceEncoder:
        array = self._validate(series)
        self._input_mean = array.mean(axis=(0, 1), keepdims=True)
        self._input_std = np.maximum(array.std(axis=(0, 1), keepdims=True), self.config.eps)
        normalized = self._normalize(array)
        for length in self.config.patch_lengths:
            raw, *_ = self._scale_features(normalized, length)
            self._feature_mean[length] = raw.mean(axis=(0, 1), keepdims=True)
            self._feature_std[length] = np.maximum(
                raw.std(axis=(0, 1), keepdims=True), self.config.eps
            )
        return self

    def fit_transform(self, series: np.ndarray) -> StructuredEncoding:
        return self.fit(series).transform(series)

    def transform(self, series: np.ndarray) -> StructuredEncoding:
        if self._input_mean is None or self._input_std is None:
            raise RuntimeError("call fit or fit_transform before transform")
        array = self._validate(series)
        if array.shape[2] != self._input_mean.shape[2]:
            raise ValueError("channel count must match the data passed to fit")
        normalized = self._normalize(array)
        scales: dict[int, ScaleEncoding] = {}
        for length in self.config.patch_lengths:
            raw, starts, amplitude, phase, frequency = self._scale_features(normalized, length)
            if length not in self._feature_mean:
                raise RuntimeError(f"missing fitted feature statistics for patch length {length}")
            embedding = (raw - self._feature_mean[length]) / self._feature_std[length]
            channels = array.shape[2]
            names = tuple(
                f"channel_{channel}.{name}"
                for channel in range(channels)
                for name in self._BASE_FEATURE_NAMES
            )
            scales[length] = ScaleEncoding(
                patch_length=length,
                starts=starts,
                feature_names=names,
                features=raw,
                embedding=embedding,
                amplitude=amplitude,
                phase=phase,
                frequency=frequency,
            )
        return StructuredEncoding(
            scales=scales,
            input_mean=self._input_mean.squeeze((0, 1)).copy(),
            input_std=self._input_std.squeeze((0, 1)).copy(),
        )

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        assert self._input_mean is not None and self._input_std is not None
        return (array - self._input_mean) / self._input_std

    def _scale_features(
        self, normalized: np.ndarray, patch_length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch, length, channels = normalized.shape
        if patch_length > length:
            raise ValueError(f"patch length {patch_length} exceeds input length {length}")
        stride = self.config.stride or patch_length
        starts = np.arange(0, length - patch_length + 1, stride)
        patches = np.stack([normalized[:, start : start + patch_length, :] for start in starts], axis=1)
        # [B, P, L, C]
        mean = patches.mean(axis=2)
        std = patches.std(axis=2)
        time = np.arange(patch_length, dtype=float)
        centered_time = time - time.mean()
        slope = (patches * centered_time[None, None, :, None]).sum(axis=2) / np.sum(centered_time**2)

        periodic_bin = max(1, int(round(patch_length / self.config.period)))
        spectrum = np.fft.rfft(patches - mean[:, :, None, :], axis=2)
        periodic_bin = min(periodic_bin, spectrum.shape[2] - 1)
        coefficient = (2.0 / patch_length) * spectrum[:, :, periodic_bin, :]
        amplitude = np.abs(coefficient)
        phase = np.angle(coefficient)

        non_dc = np.abs(spectrum[:, :, 1:, :])
        dominant_bin = np.argmax(non_dc, axis=2) + 1
        frequency = dominant_bin / patch_length
        periodic = amplitude[:, :, None, :] * np.cos(
            2 * np.pi * frequency[:, :, None, :] * time[None, None, :, None]
            + phase[:, :, None, :]
        )
        residual = patches - mean[:, :, None, :] - slope[:, :, None, :] * centered_time[None, None, :, None]
        residual_std = (residual - periodic).std(axis=2)

        per_channel = np.stack(
            (mean, std, slope, amplitude, np.sin(phase), np.cos(phase), frequency, residual_std), axis=-1
        )
        features = per_channel.reshape(batch, len(starts), channels * len(self._BASE_FEATURE_NAMES))
        return features, starts, amplitude, phase, frequency
