# Method, protocol and numerical behavior

## State and update timing

One instance stores text vectors, visual prototypes, normalized anchors, a feature count, first and second feature moments, a historical mean and transform, and soft prediction frequencies. Create a new instance or call `reset()` at every dataset boundary.

For each image:

1. Refresh geometry if the preceding history is eligible.
2. Update prototypes using the current uncorrected text probabilities.
3. Compute baseline and geometric scores.
4. Apply frequency correction using the previous frequency estimate.
5. Update frequency from uncorrected geometric scores and add the feature to history.

With default settings, the first transformed prediction is image 129. Neither corrected predictions nor ground-truth labels affect prototype updates. All output branches share one prototype trajectory.

## Numerical modes

CUDA float16 is the reference input/prototype path. Feature moments and covariance operations use float64. The scaled inverse is cast to float32 before its symmetric eigendecomposition, then the square-root transform is cast to float32. The release deliberately retains this operation order instead of substituting a mathematically equivalent direct inverse-square-root expression. Frequency history uses NumPy float64 storage with the original float32 probability calculations.

`normalization="guarded"` retains the previous anchor only if a candidate or its norm is nonfinite, or its norm is zero. Valid candidates use the same normalization arithmetic as the reference. `normalization="legacy"` performs the original division without protection and may fail on a cancellation case. Both the baseline and proposed branches use the same selected rule.

Warm-up preserves the reference mixed-precision expression. The geometric warm-up score is algebraically the baseline in exact arithmetic but need not be bitwise equal to it.

CPU float32 mode and different hardware/library versions may change predictions, especially near ties. `topk` is used for the prediction rule to preserve the reference tie handling on the validated environment.

For repeated stream orders, extract features once and permute cached rows. Do not redraw augmentations for each test order. Multiview feature means are not renormalized, and selected-view mean logits must be retained. A geometry weight of 0.5 denotes the coefficient in the formula, not equal score magnitudes under every input.

## Ablations

The default replay emits the unadapted text score, shared prototype baseline, geometry before frequency correction and complete method. `frequency_only.json` applies frequency correction to the original baseline. Config fields also support controlled changes to geometry weight, frequency strength, prototype update rate and refresh timing. These runs are new experiments unless all conditions match an archived experiment.

## Controlled class composition

```bash
python scripts/make_order.py --labels manifests/example/labels.npy --mode dirichlet --concentration 0.1 --blocks 100 --seed 11 --output outputs/example/dirichlet_order.npy
python scripts/replay.py --cache cache/example --config configs/non_iid.json --order-file outputs/example/dirichlet_order.npy --output outputs/example/dirichlet_predictions.npz
python scripts/score.py --predictions outputs/example/dirichlet_predictions.npz --labels manifests/example/labels.npy
```

The stream constructor allocates each class across chronological blocks using a Dirichlet draw and multinomial counts, then shuffles within blocks. `--mode separate` presents whole classes successively. Labels are used for constructing these evaluation conditions, then only row indices enter adaptation. This portable stream generator defines a new explicit realization. Exact historical realization requires its original saved permutation.

Frequency history estimates the model's prediction distribution, not the true label prior. The class-concentrated configuration disables frequency correction. This experiment therefore evaluates geometric adaptation under changes in class composition.

## Reference comparison

If the original source and feature cache are available externally:

```bash
python scripts/compare_reference.py --reference-code /external/archived_code --cache /external/cache/example --samples 260 --output outputs/reference_check.json
```

The external directory must contain `fgm.py` and `prototype_core.py`. For compatibility with archived implementations, `fggm.py` and its `FGGM` class name are also accepted. The comparison uses the current names when both are present. It compares all score branches, anchors and frequency histories for both release normalization modes. Omit `--samples` for a complete cached stream. It does not use labels or run the encoder.

## Efficiency

Cached replay excludes image loading and encoder computation. Its elapsed time is not end-to-end inference FPS and must not be substituted for the paper's efficiency table.
