import numpy as np
import pandapower as pp
from typing import Dict, Tuple, Iterable, Any, Optional

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict as SpaceDict


class NormalizeActionWrapper(gym.Wrapper):
    """
    Map agent actions in [-1,1] for the continuous part to the env's true [low, high].
    Discrete parts pass through unchanged.

    Supports:
      - Box
      - Dict({"continuous": Box, "discrete": Discrete|MultiDiscrete})
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        act = env.action_space

        if isinstance(act, Box):
            self._low, self._high = act.low, act.high
            self.action_space = Box(low=-1.0, high=1.0, shape=act.shape, dtype=np.float32)

        elif isinstance(act, SpaceDict) and isinstance(act.spaces.get("continuous"), Box):
            box = act.spaces["continuous"]
            self._low, self._high = box.low, box.high
            # keep discrete as-is; only scale the continuous box
            self.action_space = SpaceDict({
                "continuous": Box(low=-1.0, high=1.0, shape=box.shape, dtype=np.float32),
                **{k: v for k, v in act.spaces.items() if k != "continuous"},
            })
        else:
            raise TypeError("NormalizeActionWrapper requires Box or Dict({'continuous': Box, ...}).")

    @staticmethod
    def _scale(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return low + (0.5 * (x + 1.0) * (high - low))

    def step(self, action):
        if isinstance(self.action_space, Box):
            action = self._scale(np.asarray(action, dtype=np.float32), self._low, self._high)
        else:
            c = np.asarray(action.get("continuous", []), dtype=np.float32)
            c = self._scale(c, self._low, self._high)
            action = {**action, "continuous": c}
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def pp_get_idx_safe(net, table: str, name: str) -> Optional[int]:
    try:
        return int(pp.get_element_index(net, table, name))
    except Exception:
        return None

def attach_device_to_net(net, area_name: str, dev):
    """Create/find the pandapower element(s) for a device, return (table, idx)."""
    def _bus_id(bus_name):
        if len(area_name) > 0: 
            bus_name = f"{area_name} {bus_name}"
        return pp.get_element_index(net, "bus", bus_name)

    cls = dev.__class__.__name__
    name = f"{area_name} {dev.name}" if len(area_name) > 0 else dev.name

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
