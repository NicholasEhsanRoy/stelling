# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: record every assume-classification call's atom size."""
import atexit, json, math, os

_REC = {"calls": 0, "size0": 0, "sizes": {}, "shapes": {}}
_OUT = os.environ.get("SZ0_OUT", "/home/nick/MSF/.wt-sz0/lab/sz0_probe.json")


def pytest_configure(config):
    from stelling.propagate import _Propagator
    orig = _Propagator._apply_assumed_pred

    def wrapped(self, atom, *a, **k):
        shape = tuple(getattr(getattr(atom, "aval", None), "shape", ()) or ())
        n = math.prod(shape) if shape else 1
        _REC["calls"] += 1
        _REC["sizes"][str(n)] = _REC["sizes"].get(str(n), 0) + 1
        _REC["shapes"][str(shape)] = _REC["shapes"].get(str(shape), 0) + 1
        if n == 0:
            _REC["size0"] += 1
        return orig(self, atom, *a, **k)

    _Propagator._apply_assumed_pred = wrapped


@atexit.register
def _dump():
    with open(_OUT, "w") as f:
        json.dump(_REC, f, indent=1)
