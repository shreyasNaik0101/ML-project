# Does average daily traffic volume improve crash severity prediction?

On the matched 2025 crashes, adding annual average daily flow (AADF) reduced Random Forest macro F1 from **0.401 to 0.385**. The model picked up more Serious crashes and fewer Fatal crashes.

## Results on the same 3,914 test crashes

| Metric | Six inputs | Six inputs + AADF |
|---|---:|---:|
| Accuracy | 74.81% | 73.76% |
| Macro F1 | 0.401 | 0.385 |
| Serious recall | 20.30% (164/808) | 22.15% (179/808) |
| Fatal precision | 10.00% (4/40) | 3.51% (2/57) |
| Fatal recall | 8.16% (4/49) | 4.08% (2/49) |
| Fatal average precision | 0.052 | 0.045 |

Fatal average precision summarizes the precision–recall curve using each model’s raw Fatal scores. The test contained 49 Fatal crashes, so a change of one correctly detected crash moves fatal recall by about two percentage points.

## Matching and experimental setup

- Matched 19,397 of 101,525 STATS19 crashes (19.1%) to 7,778 DfT major-road links. Each match used the same road number and a mapped segment within 50 metres in British National Grid coordinates. The nearest eligible segment had to be more than 10 metres closer than any competing segment with a different count-point ID.
- Joined the 2025 AADF value by count-point ID. The feature is `aadf_motor_vehicles_per_day`. DfT flagged 13,525 matched crashes’ AADF values as Estimated and 5,872 as Counted. AADF measures typical daily traffic volume. The matching method does not supply a crash-time density measurement.
- Kept the original crash-level split and used the same 15,483 development crashes and 3,914 test crashes for both models. Each forest used the earlier tuned settings: 150 trees, depth 14, minimum 10 samples per leaf, moderate class weights, and the same class score multipliers. Match identifiers, coordinates, road numbers, and matching distances were used only for the join.
- This experiment estimates the effect of adding AADF for the matched major-road sample with these settings. Some road links can appear in both development and test because the split is by crash. Evaluation on unseen road links or future years would require a different split.

[DfT AADF, count points, and 2025 Major Roads Database](https://roadtraffic.dft.gov.uk/downloads)

[STATS19 collision data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)

## Three-model comparison

All six runs below used the same 3,914 test crashes. For logistic regression and the decision tree, model complexity, class weights, and decision score multipliers were selected on the matched validation rows with AADF included. The selected settings were then used for both feature sets. The Random Forest retains the settings from the earlier full-data validation experiment. The comparison is between these fitted configurations.

| Model | AADF | Accuracy | Macro F1 | Fatal precision | Fatal recall |
|---|---|---:|---:|---:|---:|
| Logistic regression | Included | 67.02% | **0.435** | **22.58%** (7/31) | **14.29%** (7/49) |
| Random Forest | Included | **73.76%** | 0.385 | 3.51% (2/57) | 4.08% (2/49) |
| Decision tree | Included | 72.43% | 0.383 | 2.94% (1/34) | 2.04% (1/49) |
| Logistic regression | Omitted | 68.75% | 0.442 | 21.88% (7/32) | 14.29% (7/49) |
| Random Forest | Omitted | 74.81% | 0.401 | 10.00% (4/40) | 8.16% (4/49) |
| Decision tree | Omitted | 67.88% | 0.395 | 6.74% (6/89) | 12.24% (6/49) |

With all seven input groups, logistic regression had the highest macro F1 and caught the most fatal crashes, while Random Forest had the highest accuracy among the three. Predicting Slight for every test crash would reach 78.10% accuracy with zero fatal recall. Adding AADF reduced macro F1 in all three comparisons. Logistic regression caught seven fatal crashes under each feature set and made one fewer false fatal prediction with AADF.

Open [logistic_regression.ipynb](logistic_regression.ipynb) or [decision_tree.ipynb](decision_tree.ipynb) for the validation trials and confusion matrices. Each notebook needs `matched_crashes.csv.gz` in its working directory. The pinned modeling environment is documented in the [original project README](../README.md).

## Reproduce and share

`matched_crashes.csv.gz` contains the six original input groups, AADF, the severity label, the crash ID, and the split. To repeat the comparison with another model, hold the rows and split fixed and toggle only the AADF column. Use `split` to select rows and `severity` as the target. Exclude `collision_index` and `split` from the predictors.

To rerun the Random Forest comparison, use the Python environment from the original experiment, plus `shapely==2.2.0` and `pyshp==3.1.8`. Download `mrdb-2025.zip` from the DfT page above. Run the command from the folder containing `compare_aadf.py`, `train_random_forest.py`, `team_crashes.csv.gz`, `metrics.json`, and `tuned_metrics.json`.

```powershell
python compare_aadf.py --data-dir "C:\Users\shrey\ML-Traffic-Project" --geometry "C:\path\to\mrdb-2025.zip"
```

The saved `.joblib` files contain the fitted pipeline, feature names, and decision weights. To score prepared rows, multiply each column returned by `pipeline.predict_proba` by its decision weight in `pipeline.classes_` order, then select the class with the largest adjusted score.
