# Validation

The local CPU regression suite passes 18 tests under Python 3.11.0 and
PyTorch 2.5.1+cpu. The B-spline check uses efficient-kan commit
7b6ce1c87f18c8bc90c208f6b494042344216b11, matching requirements.txt.

    python -m unittest discover -s tests -v

Coverage includes six-loss weighting and gradients, auxiliary sentiment labels,
IoU matching, unknown teacher masks, no-teacher/no-ACEQ behavior, stream ablations,
empty anchors, duplicate/singleton image batches, KG fallback counterfactuals,
frozen backbone behavior, checkpoint/config round trips, A100 enforcement,
B-spline construction/backpropagation, and a synthetic one-epoch training,
evaluation, and checkpoint cycle.

Tests use synthetic features or mocked pretrained backbones, so they require no
model downloads, datasets, or teacher inference. The synthetic F1 is not a
benchmark result. The RBF backend is explicitly selected for small unit tests;
a separate test exercises the actual pinned B-spline implementation.

Not validated in this environment: a clean installation of every pinned package,
pretrained model downloads, 8-bit teacher generation, full dataset preprocessing,
A100 execution, dataset training, numerical table reproduction, or human-evaluated
diagnostics. Those need the external artifacts described in reproducibility.md.
