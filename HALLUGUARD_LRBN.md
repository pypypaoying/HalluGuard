# HalluGuard-LRBN: Learnable Reversible Boundary Normalization

This is the server-run package for the four follow-up directions after
`HalluGuard-RDN-level_only` showed that boundary anchoring can help while slope
extrapolation fails.

## Variants

`fixed_level_only`

```text
z = context - last(context)
y_hat = model(z) + last(context)
```

`learnable_robust_anchor`

```text
anchor = alpha * last(context) + (1-alpha) * tail_median(context)
z = context - anchor
y_hat = model(z) + anchor
```

`learnable_residual_gate`

```text
y_raw = raw_model(context)
y_anchor = anchor_model(context - last) + last
gate = sigmoid(g(context_features))
y_hat = y_raw + gate * (y_anchor - y_raw)
```

`learnable_horizon_gate`

```text
y_hat[t] = y_raw[t] + gate[t] * (y_anchor[t] - y_raw[t])
```

`unified_revin_rdn_hybrid`

```text
center = beta * boundary_anchor + (1-beta) * instance_mean
scale = gamma * robust_tail_scale + (1-gamma) * instance_std
z = (context - center) / scale
y_hat = model(z) * scale + center
```

These are trainable modules, not test-time threshold tuning. Parameters are
learned on the training split. Validation/test targets are used only for
evaluation.

## One-command run

```bash
bash scripts/run_halluguard_lrbn_table.sh
```

Fast smoke:

```bash
DATASET_SET=ETTm1 MODELS=DLinear HORIZONS=96 \
LRBN_VARIANTS=fixed_level_only,learnable_robust_anchor \
EPOCHS=1 MAX_TRAIN_WINDOWS=128 MAX_EVAL_WINDOWS=16 \
  bash scripts/run_halluguard_lrbn_table.sh
```

Full server run:

```bash
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_halluguard_lrbn_table.sh
```

Recommended first full pass, if you want to validate one direction at a time:

```bash
LRBN_VARIANTS=fixed_level_only,learnable_horizon_gate,unified_revin_rdn_hybrid \
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_halluguard_lrbn_table.sh
```

Run one direction at a time:

```bash
LRBN_VARIANTS=learnable_residual_gate bash scripts/run_halluguard_lrbn_table.sh
LRBN_VARIANTS=learnable_horizon_gate bash scripts/run_halluguard_lrbn_table.sh
LRBN_VARIANTS=unified_revin_rdn_hybrid bash scripts/run_halluguard_lrbn_table.sh
```

## Combination Ablations

After the first full run, use the combination runner to test whether
`fixed_level_only`, `learnable_robust_anchor`, and
`unified_revin_rdn_hybrid` are complementary:

```bash
bash scripts/run_halluguard_lrbn_combo_table.sh
```

Default combo variants:

```text
fixed_level_only
learnable_robust_anchor
unified_revin_rdn_hybrid
robust_unified_hybrid
robust_unified_no_scale
fixed_anchor_unified_scale
fixed_hybrid_output_blend
```

The key candidate is `robust_unified_hybrid`:

```text
robust_anchor = alpha * last + (1-alpha) * tail_median
center = beta * robust_anchor + (1-beta) * instance_mean
scale = gamma * robust_tail_scale + (1-gamma) * instance_std
z = (context - center) / scale
y_hat = model(z) * scale + center
```

Diagnostic ablations:

- `robust_unified_no_scale`: tests whether center mixing alone is enough.
- `fixed_anchor_unified_scale`: tests whether robust anchor matters beyond
  last-value anchoring.
- `fixed_hybrid_output_blend`: diagnostic only; learns an output blend between
  hard level anchoring and the unified hybrid, so it is less claim-clean than
  the single-normalizer variants.

Focused combo run:

```bash
LRBN_VARIANTS=unified_revin_rdn_hybrid,robust_unified_hybrid,robust_unified_no_scale,fixed_anchor_unified_scale \
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_halluguard_lrbn_combo_table.sh
```

## Outputs

```text
baseline_predictions/halluguard_lrbn/*.jsonl
baseline_predictions/halluguard_lrbn_raw/*.jsonl
experiments/halluguard/results/halluguard_lrbn/lrbn_metrics.csv
experiments/halluguard/results/halluguard_lrbn/lrbn_summary.csv
experiments/halluguard/results/halluguard_lrbn/lrbn_metrics.json
experiments/halluguard/results/halluguard_lrbn/summary.md
```

Combo outputs default to:

```text
baseline_predictions/halluguard_lrbn_combo/*.jsonl
baseline_predictions/halluguard_lrbn_combo_raw/*.jsonl
experiments/halluguard/results/halluguard_lrbn_combo/lrbn_metrics.csv
experiments/halluguard/results/halluguard_lrbn_combo/lrbn_summary.csv
experiments/halluguard/results/halluguard_lrbn_combo/summary.md
```

Every row includes the external schema:

```text
sample_id, dataset, model, split, context, prediction, target
```

The metrics table includes `raw_no_correction` and
`mse_delta_pct_vs_raw` / `mae_delta_pct_vs_raw`. It also includes
`learned_params` for variants with interpretable scalar gates such as
`alpha`, `beta`, `gamma`, and `blend`.

## Claim Boundary

This script is designed to test whether the useful part of RDN-level-only can
be upgraded into a learnable, robust, reversible boundary-normalization layer.
It should not be claimed as a frequency repair method, and it should not be
claimed to beat RevIN/NST/smoothing unless the generated tables support that
under the same run settings.

Interpretation hints:

- If `fixed_level_only` helps DLinear but hurts PatchTST, boundary anchoring is
  useful but too blunt.
- If `learnable_horizon_gate` improves PatchTST relative to `fixed_level_only`,
  the residual/horizon gate is reducing over-correction.
- If `unified_revin_rdn_hybrid` is strong, the RevIN-style instance statistics
  and HalluGuard boundary anchor are complementary in a single normalization
  family.
- If all learnable variants lose to `raw_no_correction`, the input-layer route
  should be downgraded and HalluGuard-SP should remain the main parent line.
