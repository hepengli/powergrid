# grid_base_env.py
import abc
import numpy as np
import pandapower as pp
from collections import OrderedDict
from typing import Dict, Tuple, Iterable, Any, Optional, TypeVar

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Discrete, Dict as SpaceDict

from powergrid.envs.utils import attach_device_to_net

ObsType = TypeVar("ObsType")

class NormalizeActionWrapper(gym.Wrapper):
    """
    Accepts agent actions in [-1, 1] for the continuous part and maps them
    to env.action_space's true [low, high]. Discrete parts pass through.

    Supports:
      - Box
      - Dict({"continuous": Box, "discrete": Discrete|MultiDiscrete})
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space

        if isinstance(action_space, Box):
            self.low, self.high = action_space.low, action_space.high

            # What the agent sees: always [-1, 1]
            self.action_space = Box(
                low=-1.0, 
                high=1.0, 
                shape=action_space.shape, 
                dtype=np.float32
            )

        elif isinstance(action_space, SpaceDict):
            # Mixed dict: expect "continuous" and optional "discrete"
            assert "continuous" in action_space.spaces, \
                "Expected a 'continuous' key in Dict action space."
            continuous_space = action_space.spaces["continuous"]
            assert isinstance(continuous_space, Box), \
                "'continuous' must be a Box space."

            # Keep discrete part as-is if present
            action_space.spaces["continuous"] = Box(
                low=-1.0, 
                high=1.0, 
                shape=continuous_space.shape, 
                dtype=np.float32
            )

        else:
            raise TypeError(
                "ScaleFromUnitAction only supports Box or \
                    Dict({'continuous': Box, ...})."
            )

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        return self.env.step(rescaled_action)


class GridBaseEnv(gym.Env, metaclass=abc.ABCMeta):
    """
    Single-agent Gymnasium env for grid control using your Device layer.

    Users only override:
      * _build_net(self) -> None
          Must set:
             - self.net : pandapower net
             - self.area : str (prefix for names, e.g., 'MG1')
             - self.devices : OrderedDict[str, Device]  (device.name -> device)
             - self.dataset : dict with at least {'load','solar','wind'} arrays
          May set:
             - self.episode_length : int

          You can build from any network template (ieee13.py etc.) and either:
             - create the PP elements yourself, or
             - just register Device objects; we’ll attach them automatically.

      * _reward_and_safety(self) -> (rewards_dict, safety_dict)
          Return per-device reward and safety (same keys as self.devices).
          The base env aggregates to a single scalar reward for Gym/RLlib.

    Optional to override:
      * _get_obs(self) -> np.ndarray   (custom observation vector)
    """

    metadata = {"render_modes": []}

    @abc.abstractmethod
    def _build_net(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _reward_and_safety(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        raise NotImplementedError

    def __init__(self, env_config: Dict[str, Any] = None):
        super().__init__()
        self.cfg = env_config or {}
        self.base_power = float(self.cfg.get("base_power", 1.0))
        self.load_scale = float(self.cfg.get("load_scale", 1.0))
        self.train = "train" if self.cfg.get("train", True) else "test"

        # will be set by _build_net
        self.net = None
        self.area = None
        self.devices: "OrderedDict[str, Any]" = OrderedDict()
        self.dataset: Dict[str, np.ndarray] = None
        self.episode_length = int(self.cfg.get("episode_length") or 24)
        self.t = 0

        # let user build everything
        self._build_net()
        assert self.net is not None and self.area is not None

        # Ensure PP elements exist for devices; record table+index
        self._pp_refs: Dict[str, Tuple[str, int]] = {}
        for name, dev in self.devices.items():
            if hasattr(dev, "reset"):
                dev.reset(rnd=self.np_random)
            tbl, idx = attach_device_to_net(self.net, self.area, dev)
            self._pp_refs[name] = (tbl, idx)

        self.action_space = self._build_action_space()
        self.observation_space = self._build_obs_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        super().reset(seed=seed)

        T = self.dataset["load"].size

        if self.cfg.get("train", False):
            self.t = self.np_random.integers(
                T // self.episode_length - 1) * self.episode_length

        for dev in self.devices.values():
            if hasattr(dev, "reset"):
                dev.reset(rnd=self.np_random)

        self._apply_dataset_scalers()

        obs = self._get_obs()
        info = {"t": self.t}

        return obs, info

    def step(self, action):
        low, high = self.action_space.low, self.action_space.high
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))
        # set actions into device objects
        self._set_action(scaled_action)
        self._push_devices_to_net()

        # solve power flow
        converged = self._solve_pf()
        self._update_device_cost_safety(converged)

        # update costs/safety
        reward_dict, safety_dict = self._reward_and_safety()

        # aggregate to single reward
        reward = np.sum(list(reward_dict.values()))
        safety = np.sum(list(safety_dict.values()))

        reward_scale = float(self.cfg.get("reward_scale", 1.0))
        safety_scale = float(self.cfg.get("safety_scale", 0.0))
        max_penalty = float(self.cfg.get("max_penalty", None))

        reward *= reward_scale
        reward -= np.clip(safety * safety_scale, 0.0, max_penalty)

        # time & termination
        self.t += 1
        terminated = (self.t % self.episode_length == 0)
        truncated = False

        self._apply_dataset_scalers()

        obs = self._get_obs()
        info = {
            "t": self.t,
            "converged": converged,
        }
        return obs, reward, terminated, truncated, info

    def _solve_pf(self) -> bool:
        try:
            pp.runpp(self.net, numba=False)
            return self.net.get("converged", False)
        except Exception:
            self.net["converged"] = False
            return False

    def _apply_dataset_scalers(self):
        load = float(self.dataset["load"][self.t])
        solar = float(self.dataset["solar"][self.t])
        wind  = float(self.dataset["wind"][self.t])
        price  = float(self.dataset["price"][self.t])

        load_scaling = load * self.load_scale
        ids = pp.get_element_index(self.net, "load", self.area, False)
        self.net.load.loc[ids, "scaling"] = load_scaling

        for name, dev in self.devices.items():
            cls = dev.__class__.__name__

            if cls == "RES":
                scaler = solar if dev.type=="solar" else wind
                dev.update_state(scaling=scaler)

            elif cls == "Grid":
                dev.update_state(price=price)

    def _push_devices_to_net(self):
        for name, dev in self.devices.items():
            tbl, idx = self._pp_refs[name]
            cls = dev.__class__.__name__
            dev.update_state()

            if cls == "DG":
                self.net.sgen.at[idx, "p_mw"] = dev.state.P
                if "q_mvar" in self.net.sgen.columns:
                    self.net.sgen.at[idx, "q_mvar"] = dev.state.Q
                self.net.sgen.at[idx, "in_service"] = bool(dev.state.on)

            elif cls == "RES":
                self.net.sgen.at[idx, "p_mw"] = dev.state.P
                if "q_mvar" in self.net.sgen.columns:
                    self.net.sgen.at[idx, "q_mvar"] = dev.state.Q

            elif cls == "ESS":
                self.net.storage.at[idx, "p_mw"] = dev.state.P
                if hasattr(dev.state, "Q"):
                    self.net.storage.at[idx, "q_mvar"] = dev.state.Q
                self.net.storage.at[idx, "soc_percent"] = dev.state.soc * 100.0
                self.net.storage.at[idx, "in_service"] = bool(dev.state.on)

            elif cls == "Shunt":
                step = int(dev.state.step)
                q = float(step) * float(dev.q_mvar)
                self.net.shunt.at[idx, "q_mvar"] = q

            elif cls == "Transformer":
                if "tap_pos" in self.net.trafo.columns:
                    self.net.trafo.at[idx, "tap_pos"] = int(dev.state.tap_position)

            elif cls == "Switch":
                if "closed" in self.net.switch.columns:
                    self.net.switch.at[idx, "closed"] = bool(dev.state.closed)

    def _update_device_cost_safety(self, converged: bool):
        # device-local cost/safety
        for dev in self.devices.values():
            if dev.__class__.__name__ == "Transformer":
                # needs loading for safety
                tbl, idx = self._pp_refs[dev.name]
                loading = 0.0
                if (
                    converged and 
                    "res_trafo" in self.net and 
                    len(self.net.res_trafo)
                ):
                    loading = self.net.res_trafo.at[idx, "loading_percent"]
                dev.update_cost_safety(loading)
            elif dev.__class__.__name__ == "Grid":
                P, Q = self.net.res_ext_grid.values[0]
                dev.update_state(P=P, Q=Q)
                dev.update_cost_safety()
            else:
                dev.update_cost_safety()

    def _device_action_slices(self):
        """
        Concatenate continuous and discrete parts across devices.
        Returns (cont_low, cont_high, disc_nvec, slices)
        slices: list of (dev, c_len, d_len)
        """
        cont_low, cont_high, disc_nvec, slices = [], [], [], []

        for dev in self.devices.values():
            if dev.action.dim_c:
                low = dev.action.range[0].ravel()
                high = dev.action.range[1].ravel()
                cont_low.extend(low.tolist())
                cont_high.extend(high.tolist())

            if dev.action.dim_d:
                ncats = int(getattr(dev.action, "ncats", 2))
                disc_nvec.extend([ncats] * dev.action.dim_d)

            slices.append((dev, dev.action.dim_c, dev.action.dim_d))

        cont_low = np.array(cont_low, dtype=np.float32) if cont_low else None
        cont_high = np.array(cont_high, dtype=np.float32) if cont_high else None
        disc_nvec = np.array(disc_nvec, dtype=np.int64) if disc_nvec else None

        return cont_low, cont_high, disc_nvec, slices

    def _build_action_space(self):
        cont_low, cont_high, disc_nvec, _ = self._device_action_slices()
        spaces = {}
        if cont_low is not None:
            spaces["continuous"] = Box(
                low=cont_low,
                high=cont_high,
                dtype=np.float32
            )

        if disc_nvec is not None:
            spaces["discrete"] = MultiDiscrete(
                disc_nvec
            ) if len(disc_nvec) > 1 else Discrete(
                disc_nvec[0]
            )

        if not spaces:
            raise "no spaces"

        if len(spaces) == 1:
            return list(spaces.values())[0]

        return SpaceDict(spaces)

    def _set_action(self, action):
        if isinstance(self.action_space, SpaceDict):
            act_c = np.array(action.get("continuous", []))
            act_d = np.array(action.get("discrete", []), dtype=np.int64)
        else:
            if isinstance(self.action_space, Box):
                act_c = np.array(action, dtype=np.float32).ravel()
                act_d = np.array([], dtype=np.int64)
            else:
                act_c = np.array([], dtype=np.float32)
                act_d = np.array(action, dtype=np.int64).ravel()

        _, _, _, slices = self._device_action_slices()
        c_ofs = d_ofs = 0
        for dev, c_len, d_len in slices:
            if c_len:
                vals = act_c[c_ofs:c_ofs + c_len]
                dev.action.c[:] = vals.astype(np.float32)
                c_ofs += c_len
            if d_len:
                vals = act_d[d_ofs:d_ofs + d_len]
                dev.action.d[:] = vals.astype(np.int32)
                d_ofs += d_len

    def _get_obs(self) -> np.ndarray:
        """Default observation: 
            concat device states + 
            local bus vm_pu +
            local bus va_degree +
            local load P, Q results.
        """
        obs = np.array([], dtype=np.float32)

        # device states
        for dev in self.devices.values():
            obs = np.concatenate([obs, dev.state.get().astype(np.float32)])

        # bus voltages (vm, va) after PF
        if hasattr(self.net, "res_bus") and len(self.net.res_bus) > 0:
            vm = self.net.res_bus["vm_pu"].values.astype(np.float32)
            va = self.net.res_bus["va_degree"].values.astype(np.float32)
        else:
            vm = np.ones(len(self.net.bus), dtype=np.float32)
            va = np.zeros(len(self.net.bus), dtype=np.float32)

        obs = np.concatenate([obs, vm, va])

        # loads P,Q (after PF)
        pq = self.net.load[["p_mw", "q_mvar"]].values
        scaling = self.net.load[['scaling']].values
        obs = np.concatenate([obs, (pq * scaling).ravel() / self.base_power])

        # line loading
        if hasattr(self.net, "res_line") and len(self.net.res_line) > 0:
            line_loading = self.net.res_line[
                "loading_percent"].values.astype(np.float32)
        else:
            line_loading = np.zeros(len(self.net.line), dtype=np.float32)

        obs = np.concatenate([obs, (line_loading / 100.)])

        # hour = np.zeros(self.episode_length, dtype=np.float32)
        # hour[self.t % self.episode_length] = 1.0
        # obs = np.concatenate([obs, hour]).astype(np.float32)

        return obs.astype(np.float32)

    def _build_obs_space(self):
        shape = self._get_obs().shape
        return Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)


