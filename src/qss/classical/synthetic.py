"""Synthetic periodic tasks for Module A.

S1 (amplitude-phase interaction): C channels, each a single sinusoid
    x_c(t) = A_c cos(w t + phi_c) + eps,   y = A_1 A_2 cos(phi_1 - phi_2) + eta.

S2 (cross-period interaction): one channel whose window is split into n_patches
consecutive periods; patch k is A_k cos(w t + phi_k) + eps. The target couples
patches at the same position of consecutive "super-periods" (stride = n_patches // 2):
    y = sum_k A_k A_{k+s} cos(phi_k - phi_{k+s}) + eta,  k = 0..s-1.
With 4 patches this is A0A2cos(phi0-phi2) + A1A3cos(phi1-phi3).

S3 (structured classification): the S2 series, labelled by the sign of the noisy S2 score,
    label = 1[ sum_k A_k A_{k+s} cos(phi_k - phi_{k+s}) + eta > 0 ].
Label 1 means "patches one super-period apart are, on balance, in phase". Classes are balanced
by symmetry of the uniform phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SyntheticData:
    series: np.ndarray  # (N, L, C)
    y: np.ndarray  # (N,)
    amp: np.ndarray  # true amplitudes, (N, n_items)
    phase: np.ndarray  # true phases, (N, n_items)
    period: int
    patch_len: int  # samples per feature patch
    pairs: tuple[tuple[int, int], ...]  # item pairs entering the target
    noise_y: float


def _amp_phase(rng: np.random.Generator, n: int, k: int, amp_range: tuple[float, float]):
    amp = rng.uniform(*amp_range, size=(n, k))
    phase = rng.uniform(-np.pi, np.pi, size=(n, k))
    return amp, phase


def _target(amp: np.ndarray, phase: np.ndarray, pairs) -> np.ndarray:
    return sum(amp[:, i] * amp[:, j] * np.cos(phase[:, i] - phase[:, j]) for i, j in pairs)


def make_s1(
    n: int,
    seed: int,
    channels: int = 2,
    period: int = 24,
    n_periods: int = 4,
    noise_x: float = 0.1,
    noise_y: float = 0.02,
    amp_range: tuple[float, float] = (0.2, 1.0),
) -> SyntheticData:
    rng = np.random.default_rng(seed)
    amp, phase = _amp_phase(rng, n, channels, amp_range)
    t = np.arange(period * n_periods)
    w = 2 * np.pi / period
    series = amp[:, None, :] * np.cos(w * t[None, :, None] + phase[:, None, :])
    series = series + noise_x * rng.standard_normal(series.shape)
    pairs = ((0, 1),)
    y = _target(amp, phase, pairs) + noise_y * rng.standard_normal(n)
    return SyntheticData(series, y, amp, phase, period, period * n_periods, pairs, noise_y)


def make_s2(
    n: int,
    seed: int,
    n_patches: int = 4,
    period: int = 24,
    noise_x: float = 0.1,
    noise_y: float = 0.02,
    amp_range: tuple[float, float] = (0.2, 1.0),
) -> SyntheticData:
    if n_patches % 2:
        raise ValueError("n_patches must be even")
    rng = np.random.default_rng(seed)
    amp, phase = _amp_phase(rng, n, n_patches, amp_range)
    t = np.arange(period)
    w = 2 * np.pi / period
    # (N, n_patches, period) -> (N, n_patches * period, 1)
    patches = amp[:, :, None] * np.cos(w * t[None, None, :] + phase[:, :, None])
    series = patches.reshape(n, n_patches * period, 1)
    series = series + noise_x * rng.standard_normal(series.shape)
    s = n_patches // 2
    pairs = tuple((k, k + s) for k in range(s))
    y = _target(amp, phase, pairs) + noise_y * rng.standard_normal(n)
    return SyntheticData(series, y, amp, phase, period, period, pairs, noise_y)


def make_s3(n: int, seed: int, **kwargs) -> SyntheticData:
    """Binary version of S2: same series, y in {0, 1} from the sign of the noisy S2 target."""
    d = make_s2(n, seed, **kwargs)
    return replace(d, y=(d.y > 0).astype(np.float64))


CLASSIFICATION_TASKS = frozenset({"s3"})


def make_task(task: str, n: int, seed: int, **kwargs) -> SyntheticData:
    if task == "s1":
        return make_s1(n, seed, **kwargs)
    if task == "s2":
        return make_s2(n, seed, **kwargs)
    if task == "s3":
        return make_s3(n, seed, **kwargs)
    raise ValueError(f"unknown task {task!r}")
