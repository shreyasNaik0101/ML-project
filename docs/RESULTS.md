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

This estimates recorded crash severity conditional on an injury crash having occurred. It does not predict whether a crash will occur. The training population is Great Britain; performance in Bangalore or elsewhere in India has not been established. The split is by crash, allowing road links to recur across partitions. Future-year validation remains unmeasured. A separate unseen-link experiment is reported below. The existing shared test has already been inspected during development; results must not be presented as fresh external validation.

AADF covers a selected major-road subset, with many estimated traffic values. Only 49 test fatalities make recall sensitive to individual predictions. Feature attribution is not causal explanation. The score weighting improves a validation objective and does not calibrate probabilities.

## Reproduction and verification

Run `python train_models.py`, then `python -m unittest discover -s tests -v`. Nine tests cover partition integrity, leakage-column exclusion, bundle predictions, class-order weights, invalid inputs readable scenario encoding, malformed JSON types, unexpected predictors and grouped-link separation. GitHub Actions repeats training and both prediction entry points.

## Unseen road links

Run `python evaluate_unseen_links.py` to rematch the source data, create disjoint count-point partitions and retune every model family within those partitions. Seed 42 is used without searching seeds. There are 4,977 train links, 1,245 validation links and 1,556 test links, with zero overlap. The test has 3,862 crashes and 38 fatalities. The forest is retuned here, so changes from the historical experiment are not attributable solely to the split.

| Model | AADF | Accuracy | Macro F1 | Fatal recall |
|---|---|---:|---:|---:|
| logistic_regression | without_aadf | 67.84% | 0.4400 | 31.58% |
| logistic_regression | with_aadf | 67.53% | 0.4337 | 28.95% |
| decision_tree | without_aadf | 67.76% | 0.4054 | 18.42% |
| decision_tree | with_aadf | 67.79% | 0.4010 | 15.79% |
| random_forest | without_aadf | 64.06% | 0.4082 | 34.21% |
| random_forest | with_aadf | 64.79% | 0.4122 | 28.95% |

Detailed counts and settings are in `validation/unseen_links/audit.json` and `metrics.json`. These are new partitions of previously inspected data; future-year external validation remains separate.

## Submission material

The manuscript and editable IEEEtran source are in `docs/paper/`. Eight source-backed literature notes are in `docs/LITERATURE.md`. `docs/DEMO.md` covers execution and viva questions. The author block uses the project-team name until the real author list and order are supplied. Member reading activity must be confirmed by the members rather than inferred from these notes.

## Post merge inference validation

The inference checks now include 14 passing tests. Empty inputs, duplicate headers in CSV and gzip CSV files, unsupported speed limits, and invalid sine/cosine clock pairs are rejected. A fully missing numeric clock pair and missing speed remain imputable, while partially missing clock pairs are rejected. JSON scenarios support an unknown surface and require an HH:MM clock. `python scenario.py --options` lists supported inputs without loading a model.

All six saved historical models were scored on every one of the 3,914 fixed test crashes after these changes. Their confusion matrices exactly match `artifacts/metrics.json`; no model was retrained and no reported score was changed. The paper documents the original experiment and its original nine-test snapshot; this section records the additional inference checks.
