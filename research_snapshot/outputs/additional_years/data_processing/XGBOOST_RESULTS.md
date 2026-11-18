# XGBoost experiment

On the same 19,062 matched 2025 crashes and vehicle-composition features, XGBoost reached 66.95% accuracy and 0.4303 macro F1. Histogram gradient boosting reached 67.00% and 0.4284. XGBoost recall was 73.83% Slight (10,808/14,639), 45.60% Serious (1,924/4,219), and 14.71% Fatal (30/204). Histogram gradient boosting recall was 74.36%, 43.94%, and 15.69% respectively.

The model used 2022-2023 for training and 2024 for selecting one of eight configurations and decision weights by macro F1. It was refitted on 2022-2024 and evaluated on 2025. Its seven input groups use the existing vehicle-type counts and time features. Traffic volume is previous-year AADF, a proxy for density. Both models use the same rows and representation, but their tuning budgets differ.

The 2025 data were inspected in previous project work and road links can recur across years. These are development results, not fresh external validation. Counts of vehicles of each type are inferred from all vehicle records involved in a crash and require that information at prediction time. Raw model scores and weighted decision scores should not be interpreted as calibrated probabilities.

Files: train_xgboost.py is the reproducible script, xgboost_results.json records the settings, validation trials, test metrics and confusion matrices, and xgboost.joblib includes the trained pipeline, feature list, class order and score weights. The model was reloaded and its predictions matched. It needs xgboost 3.4.1 and scikit-learn 1.9.1. Run the script from the folder containing processed_crashes.csv.gz and vehicle_composition_optional.joblib. To predict, call pipeline.predict_proba on the listed features, multiply score columns by decision_weights in class_order, and select the highest weighted score.
