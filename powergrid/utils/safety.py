from dataclasses import dataclass
from typing import Optional
import math

def s_over_rating(P: float, Q: float, sn_mva: Optional[float]) -> float:
    """Apparent power (S) overload as a fraction of nameplate.
    Returns 0 if sn_mva is None/<=0 or NaN.
    """
    if sn_mva is None:
        return 0.0
    try:
        if math.isnan(float(sn_mva)) or float(sn_mva) <= 0:
            return 0.0
    except Exception:
        return 0.0
    S = math.hypot(P, Q)
    return max(0.0, (S - float(sn_mva)) / float(sn_mva))


def pf_penalty(P: float, Q: float, min_pf: Optional[float]) -> float:
    """Penalty for violating minimum power factor (0 if OK or undefined)."""
    if min_pf is None:
        return 0.0
    S = math.hypot(P, Q)
    if S <= 1e-9:
        return 0.0
    return max(0.0, float(min_pf) - abs(P / S))


def voltage_deviation(V_pu: float, vmin_pu: float = 0.95, vmax_pu: float = 1.05) -> float:
    """Voltage band violation (per-unit). Linear penalty outside [vmin, vmax]."""
    if V_pu < vmin_pu:
        return (vmin_pu - V_pu) / max(1e-9, (vmin_pu - 0.0))
    if V_pu > vmax_pu:
        return (V_pu - vmax_pu) / max(1e-9, (1.5 - vmax_pu))  # assume 1.5 pu worst-case bound
    return 0.0


def soc_bounds_penalty(soc: float, min_soc: float, max_soc: float) -> float:
    """Penalty if SOC goes outside [min_soc, max_soc]. Linear overflow measure."""
    if soc > max_soc:
        return soc - max_soc
    if soc < min_soc:
        return min_soc - soc
    return 0.0


def loading_over_pct(loading_pct: float) -> float:
    """Transformer/line loading over 100% (per-unit overflow)."""
    return max(0.0, (loading_pct - 100.0) / 100.0)


def rate_of_change_penalty(prev: float, curr: float, limit: float) -> float:
    """Penalty if |curr-prev| exceeds a rate/step limit; normalized by limit."""
    delta = abs(curr - prev)
    if limit <= 0:
        return 0.0
    return max(0.0, (delta - limit) / limit)


@dataclass
class SafetySpec:
    """Weights for composing a total safety score.
    Any term with weight 0 is ignored.
    """
    s_over_rating_w: float = 1.0
    pf_w: float = 1.0
    voltage_w: float = 0.0
    soc_w: float = 1.0
    loading_w: float = 1.0
    roc_w: float = 0.0


def total_safety(
    *,
    spec: SafetySpec,
    P: float = 0.0,
    Q: float = 0.0,
    sn_mva: Optional[float] = None,
    min_pf: Optional[float] = None,
    V_pu: Optional[float] = None,
    vmin_pu: float = 0.95,
    vmax_pu: float = 1.05,
    soc: Optional[float] = None,
    min_soc: float = 0.0,
    max_soc: float = 1.0,
    loading_pct: Optional[float] = None,
    prev: Optional[float] = None,
    curr: Optional[float] = None,
    limit: float = 0.0,
) -> float:
    """Compose multiple safety terms using weights in `spec`.
    Provide only the arguments relevant to enabled weights.
    """
    s = 0.0
    if spec.s_over_rating_w:
        s += spec.s_over_rating_w * s_over_rating(P, Q, sn_mva)
    if spec.pf_w:
        s += spec.pf_w * pf_penalty(P, Q, min_pf)
    if spec.voltage_w and V_pu is not None:
        s += spec.voltage_w * voltage_deviation(V_pu, vmin_pu, vmax_pu)
    if spec.soc_w and soc is not None:
        s += spec.soc_w * soc_bounds_penalty(soc, min_soc, max_soc)
    if spec.loading_w and loading_pct is not None:
        s += spec.loading_w * loading_over_pct(loading_pct)
    if spec.roc_w and prev is not None and curr is not None:
        s += spec.roc_w * rate_of_change_penalty(prev, curr, limit)
    return s
