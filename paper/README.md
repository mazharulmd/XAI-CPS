# Manuscript status

`main.tex` is an exploratory HAI 21.03 / BATADAL empirical audit draft aligned to the currently completed pilot; it is not yet the manuscript for the recommended Data4Cyber target-validity study. The Data4Cyber work is tracked separately in `docs/research/DATA4CYBER_STUDY_SPEC.md` and remains exploratory. `main.tex` currently covers: target agreement, representation-consistent local stability, deletion-based detector-score faithfulness, and recourse behavior. It no longer presents the unvalidated process-constrained counterfactual as the primary contribution.

It is not submission-ready. Remaining requirements include:

- independently compiled and rendered LaTeX with figure/table layout review;
- an expanded, verified related-work search and stronger matched temporal explanation baselines;
- broader target-complete CPS datasets and scenario/file-aware uncertainty with independent replication;
- stronger detector families and comparable operating-point evaluation;
- external process/operator validation before any actionability or physical-feasibility claim.

The current results are descriptive and exploratory. Five BATADAL target intervals cannot support confirmatory cross-dataset inference. Neither target agreement, score deletion, empirical support, nor local stability establishes causality or plant safety. Keep raw data out of Git and reproduce commands from the repository README and `docs/research/EXPERIMENT_PROTOCOL.md`.
