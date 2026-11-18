# Three-model comparison

All three models use the same processed crash rows and seven input groups, including previous-year AADF as a traffic-volume proxy, vehicle-type flags and counts, and cyclical time features. Training used 40,862 matched crashes from 2022-2023. Model settings and Serious/Fatal score multipliers were selected by macro F1 on 19,176 crashes from 2024. Selected models were refitted on 2022-2024 and evaluated on the same 19,062 crashes from 2025.

| 2025 test result | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 69.54% | 68.89% | 66.95% |
| Macro F1 | 0.4165 | 0.4313 | 0.4303 |
| Slight recall | 80.79% | 78.94% | 73.83% |
| Serious recall | 33.23% | 36.26% | 45.60% |
| Fatal recall | 12.75% | 22.06% | 14.71% |
| Fatal detected / actual | 26/204 | 45/204 | 30/204 |
| Fatal precision | 9.06% | 10.07% | 10.64% |

Random Forest and XGBoost have very similar macro F1. Random Forest identifies more Fatal crashes, while XGBoost identifies more Serious crashes. Logistic Regression has the highest overall accuracy among these configurations. Predicting Slight for every 2025 test row would yield 76.80% accuracy and zero Serious and Fatal recall.

Each model family used four fitting configurations: two values of its main complexity parameter and two sample-weighting choices. All used the same grid of decision-score multipliers. The chosen configurations are a limited search, not exhaustive. The models were reloaded and verified against their test predictions. Full validation trials and confusion matrices are in three_model_comparison.json.

The 2025 data has been inspected in previous experiments, and road links may occur in multiple years. It is an existing comparison set rather than a fresh independent test. The seven input groups include all vehicle types involved, with composition counts. Their availability at the intended prediction time should be considered. AADF measures typical road volume, not crash-time density. The crash-to-road match retains a selected major-road population.

Run compare_three_models.py from this directory with processed_crashes.csv.gz, vehicle_composition_optional.joblib, and train_xgboost.py alongside it. It uses scikit-learn 1.9.1 and XGBoost 3.4.1. Each comparison_*.joblib bundle contains the fitted pipeline, features, class order and decision weights. To predict, multiply probability columns by the class-order decision weights before selecting the highest score; adjusted scores are not calibrated probabilities.
