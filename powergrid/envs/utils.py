import numpy as np
import pandapower as pp
from typing import Dict, Tuple, Iterable, Any, Optional

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict as SpaceDict

def pp_get_idx_safe(net, table: str, name: str) -> Optional[int]:
    try:
        return int(pp.get_element_index(net, table, name))
    except Exception:
        return None

def attach_device_to_net(net, area_name: str, dev):
    """Create/find the pandapower element(s) for a device, return (table, idx)."""
    def _bus_id(bus_name):
        return pp.get_element_index(net, "bus", f"{area_name} {bus_name}")

    cls = dev.__class__.__name__
    name = f"{area_name} {dev.name}"

    if cls in ("DG", "RES"):
        # As sgen
        idx = pp_get_idx_safe(net, "sgen", name)
        if idx is None:
            bus = _bus_id(dev.bus)
            idx = pp.create_sgen(
                net, bus,
                p_mw=dev.state.P,
                q_mvar=dev.state.Q,
                sn_mva=dev.sn_mva,
                max_p_mw=dev.max_p_mw,
                min_p_mw=dev.min_p_mw, 
                max_q_mvar=dev.max_q_mvar, 
                min_q_mvar=dev.min_q_mvar, 
                name=name,
            )
        return ("sgen", idx)

    if cls == "ESS":
        idx = pp_get_idx_safe(net, "storage", name)
        if idx is None:
            bus = _bus_id(dev.bus)
            idx = pp.create_storage(
                net, bus,
                p_mw=dev.state.P,
                q_mvar=dev.state.Q,
                max_e_mwh=dev.max_e_mwh,
                min_e_mwh=dev.min_e_mwh, 
                sn_mva=dev.sn_mva, 
                soc_percent=dev.state.soc,
                max_p_mw=dev.max_p_mw, 
                min_p_mw=dev.min_p_mw, 
                max_q_mvar=dev.max_q_mvar, 
                min_q_mvar=dev.min_q_mvar,
                name=name, 
            )
        return ("storage", idx)

    if cls == "Shunt":
        idx = pp_get_idx_safe(net, "shunt", name)
        if idx is None:
            bus = _bus_id(dev.bus)
            idx = pp.create_shunt(
                net, bus, 
                q_mvar=dev.q_mvar, 
                step=dev.step, 
                max_step = dev.max_step, 
                name=name
            )
        return ("shunt", idx)

    if cls == "Transformer":
        idx = pp_get_idx_safe(net, "trafo", name)
        return ("trafo", idx)

    if cls == "Grid":
        return ("ext_grid", 0)

    # raise NotImplementedError(f"No attach rule for device type: {cls}")
