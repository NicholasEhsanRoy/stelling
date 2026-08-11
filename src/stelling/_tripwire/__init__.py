# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The overflow tripwire. **Skeleton only — the instrument is not built yet.**

What is here is the package boundary and nothing else, so that the one
exemption this feature needs is load-bearing and tested from the day it lands
rather than the day it is used. The façade (``arm``/``disarm``/``Status``),
the finding record, the report and the pytest plugin are all still to come.

The rule this package exists inside: **only** ``_adapter_jax.py`` may name a
private jax module, and it is the only file in the repository that may.
``design/private-jax-boundary.md`` is why, and
``tests/test_import_hygiene.py::test_private_jax_modules_banned_everywhere``
is what holds it shut.

This module imports nothing. It must stay that way: ``import stelling`` pulls
in no jax today, and a package that armed a jax hook at import would be the
obvious way to lose that.
"""

from __future__ import annotations
