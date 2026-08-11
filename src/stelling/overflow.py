# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Switch the overflow tripwire on. **One line, and this is the line.**

.. code-block:: python

    # conftest.py
    pytest_plugins = ["stelling.overflow"]

or, without editing anything:

.. code-block:: console

    $ pytest -p stelling.overflow

Either one turns ``--stelling-overflow`` from ``off`` to ``auto``: the
tripwire arms, your existing suite runs unchanged, and out-of-range integer
narrowings in the traced code you ran are reported in the terminal summary
with the writing line quoted, the arithmetic, and a reproducer. Add
``--stelling-overflow=require`` if you want a session that cannot arm it to
fail.

WHY OPT-IN. Arming reaches into a private jax registry and imports jax into
the test process. Someone who installed stelling for the verifier must not
find their suite instrumented because of it, so the always-registered plugin
adds one command-line flag and does nothing else — no jax import, no hook —
until this module is loaded.

**This module deliberately imports no jax.** Loading it is a declaration of
intent; ``stelling._tripwire.plugin`` does the work at ``pytest_configure``,
and everything jax-shaped stays behind ``stelling._tripwire._adapter_jax``.
The flag, the arming and the report all live in that plugin, and this module
exists so that ``pluginmanager.hasplugin("stelling.overflow")`` has something
true to say.

"ALWAYS REGISTERED" IS AN ASSUMPTION ABOUT THE READER'S CI, AND IT IS OFTEN
FALSE. The plugin reaches a session through the ``pytest11`` entry point in
``pyproject.toml``, and ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` — a common CI
hygiene setting — turns that off. Measured, in an installed environment: both
documented spellings above then did **nothing at all**, with no error and no
warning. ``pytest_plugins = ["stelling.overflow"]`` was silently green;
``-p stelling.overflow`` was silently green; only the undocumented
``-p stelling._tripwire.plugin`` worked. A silent no-op is the worst available
behaviour for a tool whose entire subject is instruments that are not running,
so :func:`pytest_addoption` below makes the documented spelling work on its
own: naming THIS module is enough, autoload or no autoload.
"""

from __future__ import annotations

from stelling._tripwire.plugin import ENTRY_POINT_NAME, OPT_IN_PLUGIN

__all__ = ["ENTRY_POINT_NAME", "OPT_IN_PLUGIN"]


def pytest_addoption(parser, pluginmanager):
    """Register the real plugin if nothing else has, and stay idempotent.

    ``pytest_addoption`` is a *historic* hook, so a plugin registered from
    inside it still has its own ``pytest_addoption`` replayed and still adds
    ``--stelling-overflow``. That is what lets this module be the entry point
    a user names.

    THE GUARD AND THE NAME ARE THE WHOLE DESIGN, and both are measured rather
    than reasoned about. Registering the same module object twice raises
    ``ValueError: Plugin already registered under a different name`` — an
    INTERNALERROR before a single test collects, which is far worse than the
    no-op it replaces — and there are two ways to hit it:

    * the entry point got there first, with autoload on and this module named
      in a ``conftest.py``. :meth:`is_registered` is the guard.
    * WE get there first, with autoload on and ``-p stelling.overflow``,
      because ``-p`` is consumed in ``Config._preparse`` *before*
      ``load_setuptools_entrypoints``. Registering under
      :data:`~stelling._tripwire.plugin.ENTRY_POINT_NAME` is the guard: the
      loader's own ``get_plugin(ep.name)`` check then sees it and skips.

    Both directions are driven in ``tests/test_tripwire_plugin.py``.

    Nothing here imports jax, and nothing here arms: this only makes the
    plugin present. It still does nothing until ``--stelling-overflow`` or the
    presence of this module resolves the mode to something other than ``off``.
    """
    from stelling._tripwire import plugin as _plugin

    if pluginmanager.is_registered(_plugin):
        return
    pluginmanager.register(_plugin, ENTRY_POINT_NAME)
