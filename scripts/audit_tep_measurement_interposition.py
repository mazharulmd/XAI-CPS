"""Reproduce a plumbing audit for the public TEP simulator's measurement path.

This is not an IDS benchmark or a physically calibrated attack. It checks that
an identity observation wrapper exactly preserves a native run and that a
demonstration-only measurement edit can be delivered to a custom controller
and observer without editing the wrapper's copy of the process measurement.

Install the simulator at the source revision recorded in
docs/research/TEP_SIMULATOR_FEASIBILITY_AUDIT.md, then run:

    python scripts/audit_tep_measurement_interposition.py
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
import sys

import numpy as np


SIMULATOR_REVISION = "9a6c8e5fcef4a2850778704e7793c87b0a187005"
SEED = 42
DURATION_HOURS = 0.2
RECORD_INTERVAL = 1
PRESSURE_TEMPERATURE_INDEX = 8  # Zero-based XMEAS(9), reactor temperature.
DEMO_OFFSET = 20.0
DEMO_ONSET_STEP = 5


def import_tep():
    """Import TEP after giving a clear message for the optional dependency."""
    try:
        from tep import TEPSimulator
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Install the optional TEP simulator from the pinned revision in "
            "docs/research/TEP_SIMULATOR_FEASIBILITY_AUDIT.md."
        ) from exc
    return TEPSimulator


def run_native(tep_simulator):
    sim = tep_simulator(random_seed=SEED, backend="python")
    sim.initialize()
    return sim.simulate(
        duration_hours=DURATION_HOURS,
        record_interval=RECORD_INTERVAL,
    )


def run_wrapped(tep_simulator, *, spoof: bool):
    sim = tep_simulator(random_seed=SEED, backend="python")
    sim.initialize()
    native_controller = sim.controller
    controller_observations: list[np.ndarray] = []
    observer_observations: list[np.ndarray] = []
    raw_measurements: list[np.ndarray] = []

    def observation_controller(xmeas, xmv, step):
        # The simulator owns xmeas. Make an independent observation copy before
        # any attacker transformation so a channel edit cannot write to it.
        raw = np.array(xmeas, dtype=float, copy=True)
        observed = raw.copy()
        if spoof and step >= DEMO_ONSET_STEP:
            observed[PRESSURE_TEMPERATURE_INDEX] += DEMO_OFFSET
        raw_measurements.append(raw)
        controller_observations.append(observed.copy())
        # Capture the signal a downstream IDS would receive on the same
        # compromised channel; this smoke check does not execute an IDS model.
        observer_observations.append(observed.copy())
        assert raw.shape == (41,)
        assert np.isfinite(raw).all() and np.isfinite(observed).all()
        return native_controller.calculate(observed, xmv, step)

    result = sim.simulate_with_controller(
        duration_hours=DURATION_HOURS,
        controller=observation_controller,
        record_interval=RECORD_INTERVAL,
    )
    return result, raw_measurements, controller_observations, observer_observations


def main() -> None:
    tep_simulator = import_tep()
    native = run_native(tep_simulator)
    identity, raw, clean_controller_view, clean_observer_view = run_wrapped(
        tep_simulator, spoof=False
    )
    attacked, attacked_raw, attacked_controller_view, attacked_observer_view = (
        run_wrapped(tep_simulator, spoof=True)
    )

    # The identity wrapper must be behaviorally identical to the simulator's
    # own controller path for this pinned seed and configuration.
    np.testing.assert_array_equal(native.states, identity.states)
    np.testing.assert_array_equal(native.measurements, identity.measurements)
    np.testing.assert_array_equal(native.manipulated_vars, identity.manipulated_vars)
    np.testing.assert_array_equal(clean_controller_view, clean_observer_view)
    np.testing.assert_array_equal(attacked_controller_view, attacked_observer_view)

    # Confirm the configured change reaches the two observation consumers,
    # while the raw pre-transform measurement and state array remain separate.
    probe = DEMO_ONSET_STEP
    assert np.array_equal(raw[probe], clean_controller_view[probe])
    assert np.array_equal(attacked_raw[probe], clean_controller_view[probe])
    assert attacked_controller_view[probe][PRESSURE_TEMPERATURE_INDEX] == (
        clean_controller_view[probe][PRESSURE_TEMPERATURE_INDEX] + DEMO_OFFSET
    )
    assert not np.array_equal(native.states, attacked.states)
    assert not np.array_equal(native.manipulated_vars, attacked.manipulated_vars)
    assert native.states.shape[1] == 50
    assert native.measurements.shape[1] == 41

    report = {
        "audit_type": "simulator plumbing smoke check; not an IDS result",
        "simulator_version": "2.5.0",
        "simulator_commit": SIMULATOR_REVISION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "backend": "python",
        "seed": SEED,
        "duration_hours": DURATION_HOURS,
        "record_interval_steps": RECORD_INTERVAL,
        "recorded_samples": int(len(native.time)),
        "state_shape": list(native.states.shape),
        "measurement_shape": list(native.measurements.shape),
        "manipulated_variable_shape": list(native.manipulated_vars.shape),
        "identity_wrapper_exact_match": True,
        "controller_and_observer_input_traces_equal": True,
        "ids_model_evaluated": False,
        "raw_measurement_unchanged_at_attack_onset": True,
        "observed_channel_delta_at_attack_onset": float(
            attacked_controller_view[probe][PRESSURE_TEMPERATURE_INDEX]
            - clean_controller_view[probe][PRESSURE_TEMPERATURE_INDEX]
        ),
        "demonstration_edit": {
            "channel": "XMEAS(9), reactor temperature",
            "zero_based_index": PRESSURE_TEMPERATURE_INDEX,
            "offset": DEMO_OFFSET,
            "onset_step": DEMO_ONSET_STEP,
            "unit": "degrees Celsius",
            "physically_calibrated": False,
        },
        "max_abs_plant_state_difference_vs_native": float(
            np.max(np.abs(native.states - attacked.states))
        ),
        "max_abs_manipulated_variable_difference_vs_native": float(
            np.max(np.abs(native.manipulated_vars - attacked.manipulated_vars))
        ),
        "interpretation": (
            "The wrapper plumbing is reproducible for this configuration. "
            "The demonstration offset is arbitrary and is not evidence of a "
            "realistic, stealthy, or physically reachable attack. The returned "
            "plant measurements are generated from the resulting plant state; "
            "controller/observer inputs are separately captured."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output_path = Path("results/summaries/tep_measurement_interposition_smoke.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"Saved: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
