from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

Array = np.ndarray


def _as_f32(x: Sequence[float] | Array) -> Array:
    return np.asarray(x, dtype=np.float32)

def _cat_f32(parts: List[Array]) -> Array:
    return np.concatenate(parts, dtype=np.float32) if parts else np.zeros(0, np.float32)

def _one_hot(idx: int, n: int) -> Array:
    v = np.zeros(n, dtype=np.float32)
    if n > 0:
        v[int(np.clip(idx, 0, n - 1))] = 1.0
    return v

def _pos_seq_voltage_mag_angle(Vmag: Array, theta: Array) -> Tuple[float, float]:
    """Positive-sequence magnitude/angle from per-phase magnitudes & angles (A,B,C order)."""
    a = np.exp(1j * 2 * np.pi / 3)  # e^{j120°}
    Va = Vmag[0] * np.exp(1j * theta[0])
    Vb = Vmag[1] * np.exp(1j * theta[1])
    Vc = Vmag[2] * np.exp(1j * theta[2])
    V1 = (Va + a * Vb + (a ** 2) * Vc) / 3.0
    return float(np.abs(V1)), float(np.angle(V1))

def _circ_mean(rad: Array) -> float:
    """Circular mean of angles (radians)."""
    rad = np.asarray(rad, dtype=np.float32)
    return float(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))

