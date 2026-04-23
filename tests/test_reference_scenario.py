"""End-to-end acceptance: construct -> rollout -> save -> load -> plot."""

import matplotlib

matplotlib.use("Agg")


def test_reference_scenario_end_to_end(tmp_path):
    from examples.reference_scenario import run

    run_path = tmp_path / "run.h5"
    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    run(hdf5_path=run_path, plots_dir=out_dir)

    assert run_path.exists()
    pngs = list(out_dir.glob("*.png"))
    # Three plots: defender_rtn_3d, reward, defender_mass
    assert len(pngs) >= 3
