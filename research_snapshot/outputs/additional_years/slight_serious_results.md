# Slight and Serious follow-up

Three histogram-gradient-boosting configurations and a shared decision-weight grid were tested using 2022-2023 training and 2024 validation data. Selected pipelines were refitted on 2022-2024 and evaluated on the previously inspected 2025 cohort. All runs retained the same seven input groups. These follow-ups do not provide independent external validation.

| Test result | Previous model | Focus on Slight and Serious | Preserve baseline Fatal recall on validation |
|---|---:|---:|---:|
| Slight recall | 80.30% | 80.59% | 82.12% |
| Serious recall | 34.08% | 35.29% | 31.88% |
| Fatal recall | 15.69% | 0% | 15.69% |
| Accuracy | 69.38% | 69.70% | 70.29% |

The unconstrained search selected the decision rule maximizing the smaller validation recall gain for Slight and Serious, with their mean recall as the tie-breaker. It suppressed Fatal predictions. The second search added a requirement to retain the previous model's validation Fatal recall. Its 2025 Slight recall improved, but Serious recall fell. Therefore this search did not demonstrate improvement of both Slight and Serious while retaining Fatal detection.

The saved slight_serious_model.joblib is the second, constrained experiment. It is an alternative operating point rather than an overall replacement for the previous model. Full reports are in slight_serious_metrics.json and unconstrained_metrics.json. The previous gradient_boosting_2022_2023.joblib remains available for comparison. Saved-model reload predictions were verified.
