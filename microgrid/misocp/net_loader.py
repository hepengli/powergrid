from importlib import import_module

def load_network(target: str):
    """
    target: 'module.sub:callable' that returns a pandapower net.
    """
    mod_name, func_name = target.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, func_name)
    return fn()
