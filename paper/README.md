# Manuscript status

`main.tex` is an exploratory, multi-dataset empirical audit draft covering HAI 21.03, Data4Cyber v3, and the partially labeled BATADAL dataset. HAI is the larger exact-tag anchor; Data4Cyber is a separate testbed with sparse, mixed-resolution target evidence; BATADAL is a five-interval component-level descriptive check. The manuscript covers target agreement, an event-matched random-ranking reference, representation-consistent local stability, deletion-based detector-score faithfulness, and recourse behavior. It does not present the unvalidated process-constrained counterfactual as the primary contribution.

It is not submission-ready. Remaining requirements include:

- independently compiled and rendered LaTeX with figure/table layout review;
- an expanded, verified related-work search and stronger matched temporal explanation baselines;
- broader target-complete CPS datasets and independent replication beyond the current five HAI files and six Data4Cyber attack families;
- stronger detector families and comparable operating-point evaluation;
- external process/operator validation before any actionability or physical-feasibility claim.

The current results are descriptive and exploratory. Five BATADAL target intervals cannot support confirmatory cross-dataset inference. Neither target agreement, score deletion, empirical support, nor local stability establishes causality or plant safety. Keep raw data out of Git and reproduce commands from the repository README and `docs/research/EXPERIMENT_PROTOCOL.md`.
