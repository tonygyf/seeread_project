import os
import yaml
from types import SimpleNamespace


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def _merge(a, b):
    for k, v in b.items():
        a[k] = _merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def load_config(path, overrides=None):
    d = yaml.safe_load(open(path))
    if "_base_" in d:
        b = d.pop("_base_")
        if not os.path.isabs(b):                     # relative _base_ resolves against the config file's own directory
            b = os.path.join(os.path.dirname(os.path.abspath(path)), b)
        d = _merge(yaml.safe_load(open(b)), d)
    return _ns(_merge(d, overrides or {}))
