# Verified experiment results

The standalone runner used 19,397 matched crashes and the fixed split: 12,371 training, 3,112 validation and 3,914 test rows. The test set contains 49 fatal cases. All six exported bundles were reloaded and checked against their in-memory test predictions.

| Model | AADF | Accuracy | Macro F1 | Fatal precision | Fatal recall |
|---|---|---:|---:|---:|---:|
| logistic_regression | without_aadf | 68.65% | 0.4414 | 22.58% | 14.29% |
| logistic_regression | with_aadf | 67.04% | 0.4350 | 22.58% | 14.29% |
| decision_tree | without_aadf | 67.88% | 0.3952 | 6.74% | 12.24% |
| decision_tree | with_aadf | 72.43% | 0.3833 | 2.94% | 2.04% |
| random_forest | without_aadf | 74.81% | 0.4008 | 10.00% | 8.16% |
| random_forest | with_aadf | 73.76% | 0.3846 | 3.51% | 4.08% |

## Interpretation

The majority-only baseline reaches 78.10% accuracy while detecting no fatal crashes. Logistic regression leads macro F1 in these configurations, but still misses most fatal crashes. Adding AADF lowers macro F1 for each model. The raw traffic proxy therefore does not demonstrate an improvement under this protocol.

The standalone logistic regression without AADF scores approximately 0.441, compared with the historical notebook value of 0.442. Historical outputs are retained as separate results. Use artifacts/metrics.json as the source for this rerun, including its selected settings and validation trials.

## Scope and limitations

This estimates recorded crash severity conditional on an injury crash having occurred. It does not predict whether a crash will occur. The training population is Great Britain; performance in Bangalore or elsewhere in India has not been established. The split is by crash, allowing road links to recur across partitions. Future-year and unseen-link validation remain unmeasured. The existing shared test has already been inspected during development; results must not be presented as fresh external validation.

AADF covers a selected major-road subset, with many estimated traffic values. Only 49 test fatalities make recall sensitive to individual predictions. Feature attribution is not causal explanation. The score weighting improves a validation objective and does not calibrate probabilities.

## Reproduction and verification

Run `python train_models.py`, then `python -m unittest discover -s tests -v`. Six tests cover partition integrity, leakage-column exclusion, bundle predictions, class-order weights, invalid inputs and readable scenario encoding. GitHub Actions repeats training and both prediction entry points.

## Academic handoff

The executable ML workflow is complete. Before formal submission, the team must supply the required author affiliations, assign at least two reviewed papers per member, confirm its institution’s template, and approve the final manuscript. Team member reviews and external validation must not be represented as completed by this repository.
