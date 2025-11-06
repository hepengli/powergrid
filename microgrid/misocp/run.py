import argparse, yaml, pickle
import pandapower as pp
from importlib import import_module
from net_loader import load_network
from data_io import SimplePickleAdapter
from solver import solve_one_day

def _import_obj(spec: str):
    mod, name = spec.split(":")
    return getattr(import_module(mod), name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="Path to scenario YAML")
    ap.add_argument("--save", default="./misocp.pkl")
    args = ap.parse_args()

    with open(args.scenario, "r") as f:
        sc = yaml.safe_load(f)

    # Load network
    net = load_network(sc["network"])
    pp.runpp(net)        # warm-up
    net.sgen.scaling = 0 # disable internal sgen

    # Load data adapter
    adapter = SimplePickleAdapter(**sc["data"]["args"])

    T    = int(sc["time"]["horizon"])
    days = int(sc["time"]["days"])

    all_days = []
    for d in range(days):
        day_data = adapter.get_day(d, T)
        res = solve_one_day(net, sc, day_data, T)
        all_days.append(res)
        print(f"Day {d:03d}  obj={res['obj']:.3f}")

    with open(args.save, "wb") as f:
        pickle.dump({"scenario": sc, "results": all_days}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {args.save}")

if __name__ == "__main__":
    main()
