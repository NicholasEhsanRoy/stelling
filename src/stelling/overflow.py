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
narrowings in your own traced code are reported in the terminal summary with
the line you wrote, the arithmetic, and a reproducer. Add
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
It carries no hooks of its own: the flag, the arming and the report all live
in the always-registered plugin, and this module exists so that
``pluginmanager.hasplugin("stelling.overflow")`` has something true to say.
"""

from __future__ import annotations

from stelling._tripwire.plugin import OPT_IN_PLUGIN

__all__ = ["OPT_IN_PLUGIN"]
