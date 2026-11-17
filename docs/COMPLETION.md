# Project completion and handoff

The academic modeling workflow is implemented: prepare collision-level data, train and compare three model families, predict individual scenarios or CSV batches, evaluate held-out road links, and run grouped development searches. This is a research prototype with measured limitations, not a validated emergency-response or pre-crash risk system.

## Project team

| Name | USN |
|---|---|
| Paarth Jain | 1MS24IS079 |
| Shreyas Naik | 1MS24IS070 |
| Hrishikesh Shetty | 1MS24IS054 |

Under the guidance of Dr. Sumana Maradithaya, HOD, Dept. of ISE, Ramaiah Institute of Technology.

## Delivered work

| Area | Evidence |
|---|---|
| Collision preparation and data attribution | `data/README.md`, `train_random_forest.py`, `compare_aadf.py` |
| LR, Decision Tree and Random Forest with and without AADF | `train_models.py`, `artifacts/metrics.json` |
| Readable scenarios and CSV predictions | `scenario.py`, `predict.py`, `examples/` |
| Road-link holdout evaluation | `evaluate_unseen_links.py`, `validation/unseen_links/metrics.json` |
| Grouped cross-validation, sampling and decision-rule search | `grouped_experiments.py`, `validation/refined_search/` |
| Coverage audit and review sample | `audit_data.py`, `validation/data_quality/` |
| Literature, manuscript and demonstration guide | `docs/LITERATURE.md`, `docs/paper/`, `docs/DEMO.md` |
| Automated regression checks | `tests/`, `.github/workflows/tests.yml` |

## Results to present

The original grouped seven-input LR baseline scores 67.53% accuracy, 0.4337 macro F1 and 28.95% Fatal recall. The development-selected follow-up scores 71.28%, 0.4153 and 39.47% on the same previously inspected test set of 3,862 crashes. This improves accuracy and Fatal recall while reducing macro F1 and Fatal precision. It detects 15 of 38 Fatal crashes, with 159 false Fatal predictions. These are not fresh external validation results. See `validation/RESULTS.md` for the full comparison.

The readable demo still uses the documented historical default. The selected follow-up is a separately recorded operating point; its higher recall alone does not establish that it should replace the demo model.

## Release checklist

- Merge the engineered-category correction in PR #4 and rerun CI. Older engineered bundles require retraining; the selected plain-feature follow-up is unaffected.
- Run the documented training, tests and example predictions from a fresh clone in the intended environment. CI covers both the historical dependency pins and the recorded follow-up environment, refitting bundles in each.
- Use the report with the team names, USNs and guide details. Confirm the institutional course and academic-year fields before submission.
- Preserve the recorded evidence and distinguish historical scores from any later reruns.

## Remaining validation

| Task | Acceptance evidence |
|---|---|
| Independent road-match review | Complete the review sample with reviewer decisions and reasons; summarize disagreements and any corrected experiment |
| Fresh-year evaluation | Obtain compatible external data; freeze the model and protocol before inspecting outcomes; report class counts, macro F1, accuracy, per-class precision/recall and confusion matrix |
| Crash-time density study, if required | Obtain timestamped pre-crash flow and speed measurements, document linkage and units, and evaluate separately from daily AADF |
| Deployment suitability | Define intended users and acceptable error costs; assess calibration, input availability and local external performance |

AADF is average daily traffic volume, not instantaneous density. Current results do not establish performance in India or justify emergency dispatch. These remaining claims require data and human review rather than additional code alone.
