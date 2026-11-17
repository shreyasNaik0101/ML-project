# Grouped model improvement experiments

The selected follow-up improves accuracy and Fatal recall on the previously inspected grouped test, with lower macro F1. It is a trade-off, not an improvement on every metric. No deployment default was changed.

| Same 3,862 test crashes | Accuracy | Macro F1 | Fatal recall |
|---|---:|---:|---:|
| Earlier LR with AADF | 67.53% | 0.434 | 28.95% (11/38) |
| Selected follow-up LR | 71.28% | 0.415 | 39.47% (15/38) |
| Always Slight | 77.21% | 0.290 | 0% (0/38) |

Accuracy rises 3.75 percentage points and Fatal recall rises 10.53 points, while macro F1 falls about 0.018. The Serious-class F1 remains weak (0.275). Fatal precision is 8.62%: 15 correct Fatal predictions out of 174, with 159 false Fatal predictions. Eighteen Fatal crashes are predicted Slight. These results still do not justify a production safety claim.

## Selection evidence

All searches use the same 15,535 development crashes and three fixed, disjoint road-link folds, containing 58, 58 and 57 Fatal crashes. Each candidate fits its own preprocessing inside each fold. Test labels do not determine the search, resampling or score multipliers.

1. **Three-family pilot:** 54 configuration evaluations, original/engineered features, three sampled parameter settings per family, and no/over/undersampling. Its recall-only operating requirement sacrificed too much accuracy.
2. **Calendar pilot:** 18 configuration evaluations with optional month/weekend inputs. This used the initial decision grid and did not establish an overall benefit from engineered features.
3. **Focused LR search:** all 15 regularization/class-weight combinations, original/engineered features, no resampling, and the expanded multiplier grid: 30 configuration evaluations. Select by OOF macro F1 subject to development accuracy >=67.53% and Fatal recall >=28.95%. These development requirements were set before evaluating the final selected model.

The final choice is LR `C=1`, training weights Slight=1, Serious=2, Fatal=5, original features plus AADF, no sampling. Decision multipliers are Slight=1, Serious=1.25, Fatal=4. Development OOF accuracy is 70.74%, macro F1 is 0.400, and Fatal recall is 30.64% (53/173). The best qualifying candidate used the original feature set; added features did not win this search. OOF scores reuse predictions for selection and are not independent validation estimates.

The search selected the model before final scoring. The existing grouped test has already been inspected in earlier work and in the pilot; it is not fresh external validation. The stricter follow-up was prompted by development trade-offs, not pilot test scores. Earlier single-validation experiments and this grouped CV search use different selection protocols and package versions, which limits causal attribution to a particular change.

## Reproduce and inspect

```bash
python -m pip install -r validation/refined_search/requirements.txt
python audit_data.py
python grouped_experiments.py --families logistic_regression --trials 15 --sampling none --include-calendar --minimum-accuracy 0.6753 --output-dir validation/refined_search --evaluate-inspected-test
python -m unittest discover -s tests -v
```

Each search folder contains readable `summary.json`, `tradeoffs.csv`, environment versions and a compressed `search.json.gz` with every trial, per-class results and confusion matrices. Summaries record commands and source refs; the first two searches used the initial multiplier grid. Models and crash-to-fold assignment files are regenerated rather than committed. The selected bundle can be scored with `predict.py`; calendar-enabled runs require month/weekend numeric columns in input CSVs, even if the selected baseline transformer ignores them.

Validation: 17 tests pass, including group isolation, training-only sampling, Fatal-as-Slight counts, calendar feature selection, score agreement with scikit-learn, bundle reload, and independence of model selection from holdout labels. The final selected bundle was reloaded and scored on all 3,862 test rows.

## Data limitations and next work

Matching retains 19,397 of 101,525 crashes. Fatal retention is 14.52%, versus 19.91% for Slight. The selected road cohort therefore needs careful interpretation. See `data_quality/class_coverage.csv` and `quality.json`. The generated map-review CSV has blank manual review fields: junction and parallel-road matches have not yet been independently checked.

Remaining work requires data or human review: verify the map sample, obtain a compatible fresh year, and pilot timestamped pre-crash sensor measurements before claiming traffic-density or future-year performance. AADF remains average daily traffic volume. Confirm that all proposed inputs are available at the intended prediction time. The classifier currently describes recorded crashes; it is not a validated pre-crash risk model.
