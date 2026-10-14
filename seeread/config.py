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
        base = yaml.safe_load(open(d.pop("_base_") if path.count("/") == 0 else __import__("os").path.join(__import__("os").path.dirname(path), d.pop("_base_"))))
        d = _merge(base, d)
    return _ns(_merge(d, overrides or {}))
