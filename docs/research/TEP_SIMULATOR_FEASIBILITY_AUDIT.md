# TEP simulator feasibility audit

**Audit date:** 2 November 2026  
**Purpose:** Establish whether a public Tennessee Eastman Process (TEP) simulator can support a controlled study that separates plant state from the measurements delivered to an IDS/controller. This is a source-level/API feasibility check, not an experiment or novelty claim.

## Decision

The `jkitchin/tennessee-eastman-profbraatz` Python implementation exposes all three relevant record types: 50 simulator states, 41 process measurements, and 12 manipulated variables. Source inspection confirms the update order: at each step the controller receives `process.get_xmeas()`, its commanded MVs are applied, and then the process is integrated; after integration, built-in detectors and the result recorder independently call `process.get_xmeas()` again. A custom controller therefore provides a documented interception point on the controller path, but the built-in IDS path does not automatically consume the same altered observation. This makes it a **plausible simulator for a controlled proof of concept with an explicit observation-mediation wrapper**, subject to local executable tests and exact source pinning.

The available implementation does **not** expose one shared observation-channel hook for both the controller and IDS while retaining an independent immutable record of uncorrupted measurements. A custom controller can alter only the controller's view; built-in detectors and saved measurement traces use direct process measurements. Merely perturbing saved measurements after a run would be an offline feature-space attack; it would not demonstrate a closed-loop measurement-integrity attack. Therefore the provisional research direction remains conditional. No alarm-preserving explanation-displacement result or physical-feasibility claim is supported by this audit.

The newer `jkitchin/tep-rust` project is not yet a drop-in substitute based on its README: its Python example returns an array containing 41 measurements and 12 manipulated variables, while its listed Python interface does not promise access to the 50 internal states or a measurement-interposition hook. Its core model has 50 states, but internal model state is not the same as state exposed to the experimenter through the selected API.

## Evidence reviewed

| Source | Evidence in the public documentation | Implication and limitation |
|---|---|---|
| [TEP Python API: simulation controls](https://github.com/jkitchin/tennessee-eastman-profbraatz/blob/9a6c8e5fcef4a2850778704e7793c87b0a187005/docs/api.md) | `simulate()` accepts duration, integration step, scheduled disturbances, and record interval; `step()` advances the simulator; `set_mv()` supports manual operation; measurements and manipulated variables can be read. | Provides a documented control surface for reproducible scenarios. The page does not document injecting altered measurements into the feedback path. |
| [TEP Python API: result fields](https://github.com/jkitchin/tennessee-eastman-profbraatz/blob/9a6c8e5fcef4a2850778704e7793c87b0a187005/docs/api.md) | `SimulationResult` lists `states` with shape `(n_steps, 50)`, `measurements` with shape `(n_steps, 41)`, and manipulated variables. The simulator API also documents `get_states()`. | Supports separate recording of simulated internal state and output measurements. The documentation does not define which states are directly observable in a field deployment or provide a sensor/measurement trust model. |
| [TEP Python API: process interfaces and controllers](https://github.com/jkitchin/tennessee-eastman-profbraatz/blob/9a6c8e5fcef4a2850778704e7793c87b0a187005/docs/api.md) | Low-level process interfaces document measurement/MV access, MV and disturbance setters; custom controllers receive measurements and MVs. | Confirms accessible control surfaces; source tracing and execution are still needed to validate any wrapper. |
| [TEP Python simulator source](https://github.com/jkitchin/tennessee-eastman-profbraatz/blob/9a6c8e5fcef4a2850778704e7793c87b0a187005/src/tep/simulator.py) | `step()` reads measurements for the controller before integration, applies commanded MVs, integrates the state, then gives direct process measurements to registered detectors. `simulate()` records the state vector and direct process measurements. | Confirms an observation mediation point is needed for an IDS that shares the attacked sensor feed with the controller. A custom controller alone does not mediate the built-in detector or result recorder. |
| [TEP-Rust README](https://github.com/jkitchin/tep-rust) | States the process has 50 states, 41 measurements, and 12 manipulated variables; its Python example says `run.to_numpy()` returns a `(960, 53)` matrix containing measurements then manipulated variables. | Strong fidelity claims and deterministic seeds are attractive, but the example output does not include internal states. Do not infer a latent-state API from the model dimensions. |

The TEP Python source and API links above are pinned to the locally inspected commit. The TEP-Rust README remains a moving branch URL and was only used to inspect the documented Python output shape. Archive and pin a release if that simulator is reconsidered.

## Reproduction smoke check

The public source checkout used for the local smoke check was version 2.5.0 at commit `9a6c8e5fcef4a2850778704e7793c87b0a187005`; the run used Python 3.12.10 and NumPy 2.5.4. The `python` backend was selected explicitly. For a fresh environment, install the optional simulator dependency from that exact source revision:

```powershell
python -m pip install "git+https://github.com/jkitchin/tennessee-eastman-profbraatz.git@9a6c8e5fcef4a2850778704e7793c87b0a187005"
python scripts/audit_tep_measurement_interposition.py
```

The committed audit script compares the simulator's native controller run to an identity observation wrapper at seed 42 for 0.2 simulated hours (1-second steps). It separately runs a **demonstration-only**, uncalibrated +20 °C offset on XMEAS(9) from step 5, passes the same copied observation to the custom controller and a trace collector representing a downstream IDS input, and retains simulator-generated state/measurement result arrays. No IDS model is executed. At attack onset, the pre-transform measurement matches the clean run and the observed channel differs by 20 °C. The identity wrapper matches native state, measurement, and manipulated-variable arrays exactly. In the demonstration run, maximum absolute differences from the native trajectory were 57.48819049040361 state units and 59.009226129052784 manipulated-variable percentage points. These large differences show that the intervention changes closed-loop behavior; they do not establish a plausible sensor compromise, an IDS detection change, or explanation displacement. The run emitted 721 samples with shapes `(721, 50)`, `(721, 41)`, and `(721, 12)` and saved its machine-readable output to `results/summaries/tep_measurement_interposition_smoke.json`.

This check validates only the wrapper's plumbing for one simulator version, seed, duration, and deliberately conspicuous signal edit. It does not validate the measurement model, calibration, attacker reachability, IDS performance, explanations, safety, or the candidate paper's hypotheses.

## Required implementation-level checks

Before using the simulator to support the candidate paper direction, a local audit must establish all of the following:

1. Identify the exact meanings, units, ordering, and update times of all 50 states and 41 measurements from the implementation and its source model.
2. Trace one integration step and verify when the simulator computes measurements, when the controller consumes them, and when manipulated variables are applied.
3. Add an explicit attacker-controlled observation channel between measurement generation and each intended consumer (controller and IDS). Keep the unmodified measurement and state separately recorded as ground truth. State explicitly whether the attacker intercepts the controller path, IDS path, or both.
4. Verify that a no-attack wrapper reproduces the upstream trajectory and controller behavior for identical seed, scenario, and settings.
5. Verify that the spoofed signal changes the consumer's observation but does not directly overwrite the plant state or the simulator's true measurement record. If it is intended to affect feedback control, verify and report the resulting closed-loop trajectory separately from open-loop IDS-only spoofing.
6. Define the attacker capability (channels, timing, amplitude/rate bounds, persistence, and knowledge) using documented process units and state whether safety shutdowns and instrument noise are included.
7. Evaluate benign measurement-noise and legitimate process-disturbance controls, alongside the measurement-integrity attack, to assess false attribution and identifiability.
8. Pin source commit, environment, all seeds, and wrapper code; publish a minimal clean-run test before generating the main experiment corpus.

## Research-design consequence

If the observation-channel wrapper passes the checks above, TEP can provide a simulator-grounded falsification test for the narrow question: *Can a bounded, explicitly placed measurement-channel intervention preserve an IDS alarm while shifting its local explanation away from the process variables implicated by the simulated event, and can a separately computed consistency signal flag that displacement?* The causal target must be defined from the simulator intervention/ground-truth channel, not from explanation rankings themselves.

This would still be a simulation result. It would not establish field reachability, attacker access to a particular sensor network, general CPS safety, or novelty over the unreviewed SSRN preprint and any source not yet obtained. If implementation inspection shows the controller cannot be fed altered measurements without invasive simulator changes, treat an IDS-only observation wrapper as a different, narrower experiment and reassess the direction before claiming closed-loop CPS evidence.

