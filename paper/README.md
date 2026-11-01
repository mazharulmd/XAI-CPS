# Manuscript status

`main.tex` is an exploratory, multi-dataset empirical audit draft covering HAI 21.03, Data4Cyber v3, and the partially labeled BATADAL dataset. HAI is the larger exact-tag anchor; Data4Cyber is a separate testbed with sparse, mixed-resolution target evidence; BATADAL is a five-interval component-level descriptive check. The manuscript covers target agreement, an event-matched random-ranking reference, representation-consistent local stability, deletion-based detector-score faithfulness, and recourse behavior. Data4Cyber now includes a normal-only Isolation Forest baseline whose explanations are explicitly black-box group ablations, not intrinsic forest attributions. This does not reproduce the released twelve-detector IPAL suite. It does not present the unvalidated process-constrained counterfactual as the primary contribution.

The latest source compiled with Tectonic 0.17.0 to a nine-page PDF, and every page was rendered and visually checked. The tables were resized to the IEEE two-column layout; the final build reports no overfull text lines. Minor underfull-box and fontconfig warnings remain. This is still not submission-ready. Remaining requirements include:

- an expanded, verified related-work search and stronger matched temporal explanation baselines;
- broader target-complete CPS datasets and independent replication beyond the current five HAI files and six Data4Cyber attack families;
- stronger detector families and comparable operating-point evaluation;
- external process/operator validation before any actionability or physical-feasibility claim.

The current results are descriptive and exploratory. Five BATADAL target intervals cannot support confirmatory cross-dataset inference. Neither target agreement, score deletion, empirical support, nor local stability establishes causality or plant safety. Keep raw data out of Git and reproduce commands from the repository README and `docs/research/EXPERIMENT_PROTOCOL.md`.

A separate TEP shared-view closed-loop spoof pilot is documented in `docs/research/TEP_CLOSED_LOOP_SPOOF_PILOT.md`. It is not yet integrated into `main.tex`: its single unbounded sensor-hold scenario and five simulator seeds do not establish realistic attack capability or novelty, and the paper's current empirical audit has a different scope.

The paired bounded-profile sensitivity follow-up is in `docs/research/TEP_BOUNDED_SPOOF_SENSITIVITY.md`. The tested 0.5- and 1.0-training-scale ramps did not displace the target ranking; these pilots remain separate from the manuscript pending a defensible threat range, a process-integrity method, and a resolved prior-art review.

