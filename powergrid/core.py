"""
Oct 12, 2020
Created by Hepeng Li

Network Core Functions.
"""

import numpy as np

# state of device
class DeviceState:
    ''' State of an Device
    '''
    def __init__(self):
        self.P = 0. # active power
        self.Q = 0. # reactive power
        self.on = 1 # operating status
        self.Pmax = 1.
        self.Pmin = -1.
        self.Qmax = 1.
        self.Qmin = -1.
    
    def get(self):
        state = np.array([self.P, self.Q], dtype=np.float32)
        if hasattr(self, 'shutting'):
            state = np.append(state, self.shutting)
        if hasattr(self, 'starting'):
            state = np.append(state, self.starting)
        if hasattr(self, 'soc'):
            state = np.append(state, self.soc)
        on_state = np.zeros(2, dtype=np.float32)
        on_state[self.on] = 1
        state = np.concatenate([state, on_state])
        
        return state

# action of the device
class Action:
    def __init__(self):
        # discreate action
        self.d = np.array([], dtype=np.int32)
        self.dim_d = 0
        # continuous action
        self.c = np.array([], dtype=np.float32)
        self.dim_c = 0

# properties of device
class Device(object):
    def __init__(self):
        super(Device, self).__init__()
        self.state = DeviceState()
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # cost and safety
        self.cost = 0
        self.safety = 0
        # action space
        self.action_space = []
        self.action_shape = 0
        # adversarial
        self.adversarial = False


# properties of distributed generator devices
class DG(Device):
    def __init__(self, name, bus, min_p_mw, max_p_mw, 
                 min_q_mvar=np.nan, 
                 max_q_mvar=np.nan, 
                 sn_mva=np.nan, 
                 min_pf=None, 
                 leading=None, 
                 cost_curve_coefs=[0,0,0], 
                 startup_time=None, 
                 shutdown_time=None, 
                 startup_cost=0, 
                 shutdown_cost=0, 
                 type='fossil', 
                 dt=1):
    
        super(DG, self).__init__()
        # properties
        self.type = type
        self.name = name
        self.bus = bus
        self.min_p_mw = min_p_mw
        self.max_p_mw = max_p_mw
        self.min_q_mvar = min_q_mvar
        self.max_q_mvar = max_q_mvar
        self.sn_mva = sn_mva
        self.leading = leading
        self.cost_curve_coefs = cost_curve_coefs
        self.dt = dt
        self.state.Pmax = max_p_mw
        self.state.Pmin = min_p_mw
        
        if not np.isnan(sn_mva):
            self.min_q_mvar = - np.sqrt(sn_mva**2 - max_p_mw**2)
            self.max_q_mvar = np.sqrt(sn_mva**2 - max_p_mw**2)
            self.state.Qmax = self.max_q_mvar
            self.state.Qmin = self.min_q_mvar
        if min_pf is not None:
            self.min_pf = min_pf
        if startup_time is not None:
            self.startup_time = startup_time
            self.shutdown_time = shutdown_time
            self.startup_cost = startup_cost
            self.shutdown_cost = shutdown_cost
            self.state.shutting = 0
            self.state.starting = 0

        self.reset()

    def set_action_space(self, rnd=None):
        rnd = np.random if rnd is None else rnd
        # action spaces
        if self.type == 'fossil':
            if not np.isnan(self.sn_mva) or not np.isnan(self.max_q_mvar):
                low = [self.min_p_mw, self.min_q_mvar]
                high = [self.max_p_mw, self.max_q_mvar]
            else:
                low, high = [self.min_p_mw], [self.max_p_mw]
            self.action.range = np.array([low, high], dtype=np.float32)
            self.action.c = rnd.uniform(low, high)
        else: # Renewable
            if not np.isnan(self.sn_mva) or not np.isnan(self.max_q_mvar):
                low, high = [self.min_q_mvar], [self.max_q_mvar]
                self.action.range = np.array([low, high], dtype=np.float32)
                self.action.c = rnd.uniform(low, high)

        if hasattr(self, 'startup_time'):
            self.action.ncats = 2
            self.action.d = np.ones(1)

    def update_state(self, scaling=1.0):
        # update uc status
        if self.action.d != None:
            self.update_uc_state()
        
        # update P and Q
        if self.action.c.size == 2:
            self.state.P, self.state.Q = self.action.c
        if self.action.c.size == 1:
            if self.type == 'fossil':
                self.state.P = self.action.c[0]
            else:
                self.state.P = self.max_p_mw * scaling
                self.state.Q = self.action.c[0]
        if self.action.c.size == 0 and self.type != 'fossil':
            self.state.P = self.max_p_mw * scaling

    def update_uc_state(self):
        # cannot start up and shut down at the same time
        assert not (self.state.shutting and self.state.starting)

        if not (self.state.shutting or self.state.starting):
            self.uc_cost = 0

        # shutting down
        if self.state.on and not self.action.d:
            self.state.shutting += 1
            if self.state.shutting > self.shutdown_time:
                self.state.on = 0
                self.state.shutting = 0
                self.uc_cost = self.shutdown_cost

        # starting up
        if not self.state.on and self.action.d:
            self.state.starting += 1
            if self.state.starting > self.startup_time:
                self.state.uc = 1
                self.state.starting = 0
                self.uc_cost = self.startup_cost
    
    def update_cost_safety(self):
        if len(self.cost_curve_coefs) == 3:  # quadratic cost
            a, b, c = self.cost_curve_coefs
            cost = a * (self.state.P**2) + b * self.state.P + c
        else:                                # piecewise linear
            assert len(self.cost_curve_coefs) % 2 == 0
            p0, f0 = self.max_p_mw, 0
            for i in range(0, len(self.cost_curve_coefs), 2):
                p1, f1 = self.cost_curve_coefs[i:i+2]
                if self.state.P <= p1:
                    cost = f0 + (self.state.P-p0) * (f1-f0)/(p1-p0)
                    break
                else:
                    p0, f0 = p1, f1
        
        self.cost = self.state.on * cost * self.dt
        if hasattr(self, 'uc_cost'):
            self.cost += self.uc_cost * self.dt

        # step safety
        self.safety = 0
        if self.action.dim_c > 1:
            S = np.sqrt(self.state.P**2 + self.state.Q**2)
            if self.sn_mva is not None:
                self.safety += max(0, S - self.sn_mva) / self.sn_mva
            # see https://www.ny-engineers.com/blog/diesel-genset-specifications-kw-kva-and-power-factor
            if hasattr(self, 'min_pf'):
                self.safety += max(0, self.min_pf - abs(self.state.P / S))

    def reset(self, rnd=None, scaling=1.0):
        self.set_action_space(rnd)
        self.update_state(scaling)


# properties of energy storage devices
class ESS(Device):
    def __init__(self, name, bus, min_p_mw, max_p_mw, max_e_mwh, 
                 min_e_mwh=0.0, 
                 init_soc=0.5, 
                 min_q_mvar=np.nan, 
                 max_q_mvar=np.nan, 
                 sn_mva=np.nan, 
                 ch_eff=0.98, 
                 dsc_eff=0.98, 
                 cost_curve_coefs=[0,0,0], 
                 dt=1):
        super(ESS, self).__init__()
        # properties
        self.type = 'ESS'
        self.name = name
        self.bus = bus
        self.min_p_mw = min_p_mw
        self.max_p_mw = max_p_mw
        self.min_e_mwh = min_e_mwh
        self.max_e_mwh = max_e_mwh
        self.min_q_mvar = min_q_mvar
        self.max_q_mvar = max_q_mvar
        self.sn_mva = sn_mva
        self.dt = dt
        self.ch_eff = ch_eff
        self.dsc_eff = dsc_eff
        self.cost_curve_coefs = cost_curve_coefs
        self.init_soc = init_soc
        self.min_soc = min_e_mwh / max_e_mwh
        self.max_soc = 1
        self.state.Pmax = max_p_mw
        self.state.Pmin = min_p_mw
        if not np.isnan(sn_mva):
            self.min_q_mvar = - np.sqrt(sn_mva**2 - max_p_mw**2)
            self.max_q_mvar = np.sqrt(sn_mva**2 - max_p_mw**2)
            self.state.Qmax = self.max_q_mvar
            self.state.Qmin = self.min_q_mvar

        self.reset()
    
    def set_action_space(self):
        # action spaces
        if not np.isnan(self.sn_mva) or not np.isnan(self.max_q_mvar):
            low = [self.min_p_mw, self.min_q_mvar]
            high = [self.max_p_mw, self.max_q_mvar]
        else:
            low, high = [self.min_p_mw], [self.max_p_mw]
        self.action.range = np.array([low, high], dtype=np.float32)
        self.action.c = np.random.uniform(low, high)

    def update_state(self):
        # self.feasible_action()
        # update P and Q
        if self.action.c.size > 1:
            self.state.P, self.state.Q = self.action.c
        else:
            self.state.P = self.action.c[0]
            
        if self.state.P >= 0: # charging
            self.state.soc += self.state.P * self.ch_eff * self.dt / self.max_e_mwh
        else: # discharging
            self.state.soc += self.state.P / self.dsc_eff * self.dt / self.max_e_mwh

    def update_cost_safety(self):
        # quadratic cost
        a, b, c, *_ = self.cost_curve_coefs
        self.cost = a * abs(self.state.P)
        # step safety
        safety = 0
        if not np.isnan(self.sn_mva):
            S = np.sqrt(self.state.P**2 + self.state.Q**2)
            safety += max(0, S - self.sn_mva) / self.sn_mva
        # restrict soc
        if self.state.soc > self.max_soc:
            safety += self.state.soc - self.max_soc
        if self.state.soc < self.min_soc:
            safety += self.min_soc - self.state.soc

        self.safety = safety

    def reset(self, rnd=None, init_soc=None):
        rnd = np.random if rnd is None else rnd
        if init_soc is None:
            self.state.soc = rnd.uniform(self.min_soc, self.max_soc)
        else:
            self.state.soc = init_soc
        self.set_action_space()
        self.update_state()
        self.cost, self.safety = 0, 0

    def feasible_action(self):
        max_dsc_power = (self.state.soc - self.min_soc) * self.max_e_mwh * \
                        self.dsc_eff / self.dt
        max_dsc_power = min(max_dsc_power, - self.min_p_mw)

        max_ch_power = (self.max_soc - self.state.soc) * self.max_e_mwh / \
                       self.ch_eff / self.dt
        max_ch_power = min(max_ch_power, self.max_p_mw)

        low, high = -max_dsc_power, max_ch_power
        if len(self.action.c) > 1:
            low = np.array([low, self.min_q_mvar])
            high = np.array([high, self.max_q_mvar])
        self.action.c = np.clip(self.action.c, low, high)

# properties of renewable energy resource device
class RES(Device):
    def __init__(self, name, bus, sn_mva, source, 
                 max_q_mvar=np.nan, 
                 min_q_mvar=np.nan, 
                 cost_curve_coefs=[0,0,0], 
                 dt=1):
        super(RES, self).__init__()
        assert source in ['SOLAR', 'WIND']
        self.type = source
        self.name = name
        self.bus = bus
        self.sn_mva = sn_mva
        self.max_q_mvar = max_q_mvar
        self.min_q_mvar = min_q_mvar
        self.cost_curve_coefs = cost_curve_coefs
        self.dt = dt
        # cost and safety
        self.cost = 0
        self.safety = 0

        self.set_action_space()
        self.update_state(0.0)

    def set_action_space(self):
        # action spaces
        if not np.isnan(self.max_q_mvar):
            low, high = self.min_q_mvar, self.max_q_mvar
            self.action.range = np.array([low, high], dtype=np.float32)
            # self.action.dim_c = 1
            self.action.c = np.random.uniform(low, high)
        else:
            self.action_callback = True

    def update_state(self, scaling):
        assert 0 <= scaling <= 1
        # update state
        self.state.P = self.sn_mva * scaling
        if self.action.c.size > 0:
            self.state.Q = self.action.c

    def update_cost_safety(self):
        # update safety
        if self.action.c.size > 0:
            S = np.sqrt(self.state.P**2 + self.state.Q**2)
            if S > self.sn_mva:
                self.safety = S - self.sn_mva
            else:
                self.safety = 0

# properties of capacitor device
class Shunt(Device):
    def __init__(self, name, bus, q_mvar, max_step=1):
        super(Shunt, self).__init__()
        # properties
        self.type = 'SCB'
        self.name = name
        self.bus = bus
        self.q_mvar = q_mvar
        self.max_step = max_step
        # cost and safety
        self.cost = 0
        self.safety = 0
        # action spaces
        self.action.ncats = max_step + 1
        self.action.dim_d = 1
        self.state.Qmax = q_mvar * max_step

    def update_state(self):
        self.state.step = self.action.d[0]

    def update_cost_safety(self):
        pass

    def reset(self, rnd):
        self.state.Q = 0

# properties of switch device
class Switch(Device):
    def __init__(self, *, name, fbus, tbus):
        super(Switch, self).__init__()
        # properties
        self.type = 'SW'
        self.name = name
        self.fbus = fbus
        self.tbus = tbus
        # status
        self.state.closed = True
        # action spaces
        self.action_callback = True

    def update_state(self):
        self.state.closed = self.state.closed

    def update_cost_safety(self):
        pass

    def reset(self, rnd):
        self.state.closed = True

# properties of substation device
class Transformer(Device):
    def __init__(self, name, type, fbus, tbus, 
                 sn_mva=None, 
                 tap_max=None, 
                 tap_min=None, 
                 dt=1):
        super(Transformer, self).__init__()
        # properties
        self.type = type # 'TAP' or 'Trafo'
        self.name = name
        self.fbus = fbus
        self.tbus = tbus
        self.sn_mva = sn_mva
        self.tap_max = tap_max
        self.tap_min = tap_min
        self.dt = dt
        # state variables
        self.state.loading = 0
        self.state.tap_position = 0
        # cost and safety
        self.cost = 0
        self.safety = 0
        # action spaces
        if tap_max is not None:
            self.action.ncats = tap_max - tap_min + 1
            self.action.dim_d = 1
        else:
            self.action_callback = True

    def update_state(self):
        if self.tap_max is not None:
            self.state.tap_position = self.action.d[0] + self.tap_min

    def update_cost_safety(self, loading):
        # update state
        self.state.loading = loading
        # update safety
        if loading > 100:
            self.safety = (loading - 100) / 100
        else:
            self.safety = 0

    def reset(self, rnd):
        self.state.loading = 0
        self.state.tap_position = 0

# properties of main grid Device
class Grid(Device):
    def __init__(self, name, bus, sn_mva, sell_discount=1, dt=1):
        super(Grid, self).__init__()
        # properties
        self.type = 'GRID'
        self.name = name
        self.bus = bus
        self.sn_mva = sn_mva
        self.sell_discount = sell_discount
        self.dt = dt
        # state variables
        self.state.P = 0
        self.state.Q = 0
        self.state.price = 0
        # cost and safety
        self.cost = 0
        self.safety = 0
        # not need action
        self.action_callback = True

    def update_state(self, price, P, Q=0):
        # update state
        self.state.P = P
        self.state.Q = Q
        self.state.price = price

    def update_cost_safety(self):
        # update cost
        if self.state.P > 0:
            self.cost = self.state.P * self.state.price * self.dt
        else:
            self.cost = self.state.P * self.state.price * self.sell_discount * self.dt
        # update safety
        S = np.sqrt(self.state.P**2 + self.state.Q**2)
        if S > self.sn_mva:
            self.safety = (S - self.sn_mva) / self.sn_mva
        else:
            self.safety = 0

    def reset(self, rnd):
        self.state.P = 0
        self.state.Q = 0
        self.state.price = 0
