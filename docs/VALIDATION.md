# Validation scope

Release preparation checks were performed on 2026-10-25 with Python 3.10.15, PyTorch 2.1.1, CUDA 11.8, NumPy 1.26.4 and an RTX 3090. These are code-release checks, not a new complete benchmark evaluation.

## Automated tests

All ten tests passed:

- Cached replay, permutation alignment and independent scoring.
- Detection of a modified prediction archive.
- Geometry refresh uses only preceding observations.
- Frequency correction uses the preceding frequency estimate.
- Geometry and frequency choices do not change prototype updates.
- A zero candidate anchor retains its previous value in guarded mode.
- Reset removes all adaptation history.
- Invalid configuration values are rejected.
- Frequency-disabled and frequency-only configurations use the intended scores.
- A synthetic default-parameter FP16 cancellation case remains finite in guarded mode and reproduces the failure in legacy mode.

Several checks are grouped within individual test functions. Run the suite with:

```bash
python -m unittest discover -s tests -v
```

The FP16 cancellation regression requires CUDA and is explicitly skipped on CPU.

## Archived-method comparisons

| Cached stream | Observations checked | Scope |
|---|---:|---|
| ViT-B/16, Flowers102 | 2,463 | Complete stream |
| ViT-B/16, Caltech101, h=50 | 260 | Prefix including two geometry refreshes |
| ViT-B/16, ImageNet-A, multiview | 260 | Prefix including two geometry refreshes |
| ResNet-50, ImageNet-A, multiview | 260 | Prefix including two geometry refreshes |

The legacy and guarded implementations matched the archived implementation bitwise for all four score branches, anchors and frequency state in these comparisons. A separate command-line replay and scoring of the complete Flowers102 stream matched the stored prediction results. Additional synthetic 512- and 1024-dimensional sequences covered the two transform refreshes and shared prototype state.

This evidence concerns the listed streams and checks. It does not establish bitwise equivalence on arbitrary future inputs or different hardware.

## Input pipeline

- Generic manifest shuffling, explicit path ordering, label alignment and invalid-order rejection were checked.
- External ImageNet-A and ImageNet-R loaders produced the archived image order and class mapping.
- Multi-template text encoding matched the historical class-vector implementation.
- AugMix tensors matched the evaluated augmentation at three different row seeds.
- For two real ImageNet-A images per backbone, the pinned official CLIP dependency produced exactly matching cached multiview features and logits. ResNet-50 selected-view indices also matched.
- Simulated interruption and resume reproduced the uninterrupted multiview output.
- Extraction succeeded when the evaluation-label file was deliberately unreadable, confirming that this path did not open it.

The dependency is pinned to official CLIP commit `c13005fd422b20dcd11774e9fff46370887218e4`. Its encoder and tokenizer logic matches the evaluated snapshot. Later revisions change ResNet-50 attention pooling arithmetic and should not be substituted without revalidation.

## What is not validated by this release

- A complete fresh rerun of every paper table.
- A complete raw-point-cloud pipeline.
- End-to-end FPS inferred from cached replay time.
- Exact accuracy reproduction from changed prompts, splits, class mappings, input orders or library versions.

The paper directory, its original code, saved benchmark outputs and PDF were not modified by this code-only packaging task.
