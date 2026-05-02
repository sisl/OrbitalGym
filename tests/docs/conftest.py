"""Shared pytest fixtures for the docs test suite.

Files in tests/docs/ each pair with a markdown page in docs/. The
markdown pulls labelled snippet regions from these tests via
pymdownx.snippets:

    --8<-- "tests/docs/test_tut_t2_rtn_scenario.py:build-config"

Inside the .py file, mark each region with paired comments:

    # --8<-- [start:build-config]
    cfg = make_pursuit_evasion(...)
    # --8<-- [end:build-config]

The body of each region must read top-to-bottom as a tutorial script —
no test-isolation idioms (no fixtures, no monkeypatching) inside region
markers. Test setup that has to live in the .py file but not in the
docs goes outside the markers.
"""

from __future__ import annotations
