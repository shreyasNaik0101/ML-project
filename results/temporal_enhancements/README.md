# Executed temporal enhancement results

Thirteen configurations were compared using 2024 validation data. Model and decision-weight selections were written before evaluating the previously inspected 2025 cohort. All comparisons below use 19,062 evaluation rows, including 204 Fatal crashes. This remains exploratory evidence, not an untouched external test.

## Measured results

| Operating point | Model | Accuracy | Macro F1 | Serious recall | Fatal recall |
|---|---|---:|---:|---:|---:|
| baseline_lr | baseline_lr | 70.60% | 0.4172 | 30.79% | 12.25% |
| baseline_rf | baseline_rf | 70.84% | 0.4269 | 31.26% | 19.61% |
| balanced | cat_5_False | 69.52% | 0.4247 | 37.35% | 12.25% |
| constrained | cat_5_False | 69.06% | 0.4204 | 36.29% | 16.67% |
| accuracy | baseline_rf | 76.90% | 0.3202 | 5.07% | 0.00% |

## Decision

Retain the current demonstration models. The validation-selected CatBoost alternative did not improve evaluation accuracy or macro F1 over the reproduced forest baseline. Its validation superiority did not transfer to 2025. The accuracy-only forest reaches 76.90%, but detects zero Fatal cases and only 214 of 4,219 Serious cases. Its macro F1 is 0.3202. This is unsuitable as a general severity-model upgrade even though it narrowly exceeds the 76.80% always-Slight accuracy baseline.

The constrained CatBoost meets the preset Fatal >=22% and Serious >=33% recall requirements on 2024, but Fatal recall falls to 16.67% on 2025. Validation constraints are not future-performance guarantees. The balanced CatBoost reaches 37.35% Serious recall but only 12.25% Fatal recall. Full confusion matrices and precision are retained in `inspected_test.json`.

## Reproducibility and comparison limits

The reproduced LR accuracy is 70.60%, matching the recorded notebook result. The reproduced RF accuracy is 70.84%, versus the historical 70.19%; the new run uses sklearn 1.8.0 rather than the notebook environment 1.9.1. Treat that difference as a rerun difference, not an algorithmic improvement. The experiment retains the notebook feature order, forest settings, sample weights and historical decision-weight grid. Compare new candidates against the reproduced baseline under the same environment.

`validation_trials.json` records every candidate and its selected operating points, including failures. `selection.json` records the frozen winners, exact dependency versions, source-data SHA256 and partition counts. `features.json` records predictor order. Fitted bundles are regenerated locally and excluded from git; their reload predictions were checked. The executed optional notebook defaults to the validation-selected balanced point for inspection, not a recommendation to replace the default models.

## Next research steps

A single validation year selected the model and decision weights, and refitting with 2024 can change the score distribution to which those weights are applied. A next study should predefine rolling-year model-selection folds and a separate calibration policy before accessing a fresh outcome year. It should also examine reporting changes, road-matching coverage and feature availability. These experiments do not justify a new accuracy claim or a guarantee of rare-class detection. AADF remains average daily volume, and the prepared major-road cohort is not every road.

Run instructions: [experiment guide](../../experiments/temporal_enhancements/README.md).
