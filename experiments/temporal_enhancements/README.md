# Temporal model enhancement experiment

This optional experiment keeps the classroom notebooks unchanged. It compares the existing LR and RF configurations with deeper forests and XGBoost/CatBoost alternatives on the same prepared features. Numeric category codes are one-hot encoded; CatBoost is evaluated on that common representation. IDs, road-link IDs, year and severity are excluded from predictors.

## Reproduce

Create a separate Python 3.12 environment and install this directory's `requirements.txt`. From the repository root:

```sh
python -m pip install -r experiments/temporal_enhancements/requirements.txt
python experiments/temporal_enhancements/run.py
python -m unittest discover -s tests -v
```

The runner fits on 2021–2023 and chooses the model and decision weights using 2024 only. It records all candidates, the data SHA256, environment versions and three operating points:

- Accuracy: maximize validation accuracy.
- Balanced: maximize validation macro F1.
- Constrained: maximize validation accuracy while requiring validation Fatal recall >=22% and Serious recall >=33%. These requirements do not guarantee the same recall on later data. Infeasible candidates are explicitly excluded.

Selection reuses the validation year across candidates and decision weights; its scores are selection estimates. A fixed seed is used, not a search for favorable seeds. No new independent external test is claimed. The matching, coverage and input-availability caveats in `data/README.md` still apply.

Only after freezing the validation selection, optionally refit on 2021–2024 and score the previously inspected 2025 cohort:

```sh
python experiments/temporal_enhancements/run.py --evaluate-inspected-test
```

This also reproduces LR/RF baseline decision-weight selection under the same runtime. It writes separate evaluation results; it never overwrites `results/final_model_comparison.csv`. Readable JSON evidence is committed; large fitted joblib bundles are ignored and can be regenerated. Bundle reload is checked against the original predictions. Only load trusted bundles using the recorded package versions. The `weights` array follows `labels` order and adjusts raw scores; it does not calibrate probabilities.

The highest-accuracy model may sacrifice rare-class detection. Compare confusion matrices and precision/recall alongside accuracy. A majority classifier already scores 76.80% on 2025 while finding no Serious or Fatal crashes. A genuine research improvement should make its error trade-offs explicit.
