import os, pickle
import numpy as np

class SimplePickleAdapter:
    """
    Exposes get_day(d, T) -> dict with arrays of length T:
      load_at(bus_i), qload_at(bus_i) are derived from net scaling in solver.
      Here we only return area-level profiles: load, solar, wind, price.
      Scaling to buses is handled inside the solver via ppc['bus'] PD/QD.
    """
    def __init__(self, base_dir, file, split, load_key, solar_key, wind_key, price_key, load_scale=1.0):
        path = os.path.join(base_dir, file)
        with open(path, "rb") as f:
            ds = pickle.load(f)
        self.load  = ds[split]["load"][load_key].reshape(366, 24) * float(load_scale)
        self.solar = ds[split]["solar"][solar_key].reshape(366, 24)
        self.wind  = ds[split]["wind"][wind_key].reshape(366, 24)
        self.price = ds[split]["price"][price_key].reshape(366, 24)

    def get_day(self, d: int, T: int):
        return {
            "load":  self.load[d, :T],
            "solar": self.solar[d, :T],
            "wind":  self.wind[d, :T],
            "price": self.price[d, :T],
        }
