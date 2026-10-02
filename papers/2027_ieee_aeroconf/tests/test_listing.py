"""The paper's code listing runs end to end and reports a capture rate."""

import runpy
from pathlib import Path

LISTING = Path(__file__).resolve().parents[1] / "listing_1_pursuit_evasion.py"


def test_listing_runs(capsys):
    namespace = runpy.run_path(str(LISTING), run_name="__main__")
    assert capsys.readouterr().out.startswith("capture rate ")
    assert namespace["caught"].shape == (1024,)
    assert namespace["t_end"].shape == (1024,)
