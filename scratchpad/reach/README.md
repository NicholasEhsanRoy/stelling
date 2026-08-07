<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# reach — the corpus behind `scratchpad/PREREG_REACH.md`

`cases.py` is GENERATED and deliberately not committed (7 935 lines, one
`S(...)` obligation per source line). Rebuild it, then run:

```
python scratchpad/reach/gen_cases.py              # writes cases.py, 736 cases
JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src \
  python scratchpad/reach/run.py OUT.json 300     # stelling x 3 legs + oracle
python scratchpad/reach/analyze.py OUT.json       # per-obligation classes
python scratchpad/reach/ledger.py BEFORE.json AFTER.json
python scratchpad/reach/deep_verified.py AFTER.json 20000
```

`oracle.py` never imports stelling: it executes the same case functions in
plain numpy with real Python control flow, and reports, per obligation,
how many sampled points EVALUATED it (`n_exec`) and how many evaluated it
false (`n_false`). `n_exec == 0` is what "unreachable" means throughout.

Obligation identity is the cases.py source LINE: stelling's
`source_info` carries the user frame, and the oracle reads the same line
from the caller's frame — so the join needs no naming convention that
either side could get wrong.
