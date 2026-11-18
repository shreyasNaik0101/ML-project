# Additional years and gradient boosting

The project adds 2022-2024 STATS19 crash and vehicle records and compares logistic regression with histogram gradient boosting. All models use the same seven input groups and the same 19,062 matched 2025 crashes.

| Model | Years used in final fit | Accuracy | Macro F1 | Fatal precision | Fatal recall | Fatal detected |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | 2023-2024 | 69.42% | 0.410 | 7.97% | 10.78% | 22/204 |
| Logistic regression | 2022-2024 | 70.00% | 0.413 | 9.85% | 9.80% | 20/204 |
| Gradient boosting | 2023-2024 | 67.85% | 0.408 | 6.35% | 28.43% | 58/204 |
| Gradient boosting | 2022-2024 | 69.38% | 0.423 | 10.06% | 15.69% | 32/204 |

Adding 2022 to the final training set improved accuracy and macro F1 for both algorithms in this comparison. Fatal recall decreased. The three-year gradient-boosting run had the highest macro F1, while the two-year run caught the most fatal crashes at a higher false-alarm rate.

## Data and method

- Source is the [Department for Transport's STATS19 open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data) and [road traffic statistics](https://roadtraffic.dft.gov.uk/downloads).
- Matched counts were 20,506 crashes in 2022, 20,356 in 2023, 19,176 in 2024 and 19,062 in 2025. Fatal counts were 240, 234, 272 and 204 respectively.
- Within each year, use that year's road geometry and previous year's AADF traffic volume. Match the recorded road number to a segment within 50 m, with a greater-than-10 m gap to the next distinct count-point link. Same-year mapped road geometry and retrospectively recorded fields make this a retrospective study. AADF is a typical daily traffic-volume proxy, not observed crash-time traffic density.
- The smaller training run selects model settings on 2024 after fitting on 2023. The larger run selects them on the same 2024 data after fitting on 2022-2023. Each final model is refitted on its training period plus 2024, then evaluated on 2025. Therefore final training periods are 2023-2024 and 2022-2024.
- The search compared four logistic-regression configurations and eight gradient-boosting configurations per training period. It selected model settings and Serious/Fatal decision multipliers on 2024 macro F1. Full search settings and confusion matrices are recorded in metrics.json. The encoded vehicle-flag column schema was assembled from the available years, including 2025, without using test labels.
- The test set is a full matched 2025 cohort rather than the previously reported 2025 grouped holdout. It has already been inspected in earlier project work. The 2025 test labels were not used to select settings for these runs. Road links can occur in both training and test years. Fresh-year and geographically disjoint performance have not been established.
- Predicting Slight for all test rows yields 76.80% accuracy and zero Fatal recall. On the same test, three-year gradient boosting predicted Fatal 318 times and was correct 32 times. It predicted Slight for 86 Fatal crashes.

## Files

matched_2022_2025.csv.gz contains crash IDs, severity, year, matched count-point IDs, and model inputs. The IDs and year were excluded from the predictors. metrics.json contains validation trials, selected settings, full test reports, and confusion matrices. The four joblib bundles contain the trained pipeline, feature list, and decision weights. Each was reloaded to reproduce its test predictions.

experiment_more_years.py is the experiment script. To rerun it, place it at the root of the paarth/grouped-model-improvements checkout, put the 2022-2024 crash and vehicle CSVs and mrdb-2022.zip through mrdb-2024.zip in a sibling yearly-data folder, and keep the existing data folder. The script expects a writable workspace outputs/additional_years folder two levels above the checkout. It reuses prepare_data and match_crashes from the project. A previously prepared matched_2022_2025.csv.gz at the output path skips the matching step.

The bundles need the scikit-learn version recorded in the project's original requirements (1.9.1). For predictions, pass the bundle's features to the saved pipeline and multiply score columns by decision_weights in the order of pipeline.classes_, then select the highest adjusted score. These weighted scores are not calibrated probabilities.
