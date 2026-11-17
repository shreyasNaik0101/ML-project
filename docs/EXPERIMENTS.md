# Grouped development experiments

The seven-input baseline is a useful starting point, not a deployment claim. Accuracy alone rewards ignoring Fatal crashes. The new workflow keeps road links separate, compares every candidate on the same three folds, and selects using pooled out-of-fold (OOF) macro F1 subject to explicit requirements.

## Run

```bash
python audit_data.py
python grouped_experiments.py --trials 8 --sampling none over under
python -m unittest discover -s tests -v
```

`--trials` is the number of reproducible parameter draws **per family**, each tested with original/engineered features and each requested sampling method. Eight trials with three methods means 144 configurations and 432 fold fits. Start with `--trials 3` for a manageable pilot. All models use the same folds. The sampled grid covers LR regularization, tree depth/leaf size, and forest size/depth/leaf size/max_features, plus unweighted/moderate/balanced class weights. A finite randomized sample does not guarantee every setting is visited.

Use `--minimum-fatal-recall 0.2895 --minimum-accuracy 0.6753` to require both proposed starting targets on development predictions. The default accuracy floor is zero; the Fatal recall floor is 28.95%. These are **development selection requirements**, not guarantees on a holdout. A failed requirement saves the trade-offs and exits without exporting a model. Set thresholds before running and do not tune them against test scores.

The runner exports `search.json`, reproducible crash-to-fold assignments, and `selected.joblib`. The JSON includes class precision/recall/F1/support, fixed-order confusion matrices (Slight, Serious, Fatal), Fatal-as-Slight counts, unadjusted per-fold results, and three OOF operating points for every candidate: accuracy, macro F1, and macro F1 subject to requirements. OOF multiplier selection and hyperparameter selection reuse development predictions, so these scores are optimistic selection estimates. Use fresh external data for independent claims.

To evaluate the selected approach on the **previously inspected** grouped holdout:

```bash
python grouped_experiments.py --trials 3 --sampling none over under --evaluate-inspected-test
```

This opt-in command fixes the model and decision rule before scoring test rows. It reports the always-Slight comparator on those same rows. Do not repeatedly use these holdout results to choose changes. The holdout is from the same source year and is not an independent future-year validation.

## Implemented comparisons

- Original features versus log1p AADF, time-of-day bands, speed bands, weather/surface combinations, road/speed interactions, and grouped cycle/motorcycle/goods presence flags (the goods group includes HGV codes 20/21 and unknown-weight code 98). Original individual vehicle flags and cyclical time remain available.
- Randomized tuning for all three existing model families.
- No resampling, random oversampling, and random undersampling. Resampling occurs only on fold-training rows; validation rows retain their real prevalence. Imputers, encoders and scalers are fitted inside each training pipeline.
- Serious/Fatal score multipliers, including constrained selection. Scores are not calibrated probabilities.
- Matching retention by severity, raw field distributions, missingness, invalid traffic checks and a reproducible sample for manual map review. No match radius was widened.

## Data and interpretation limits

The source contains 101,525 crashes; the matched cohort contains 19,397. Review `validation/data_quality/class_coverage.csv` before interpreting results as covering all roads. `matching_review.csv` has coordinates and blank review fields; nobody has manually confirmed these matches yet. Automated plausibility flags are not a substitute for reading the DfT codebook. Vehicle flags were validated by the existing preparation pipeline; the audit does not reconstruct every vehicle category from raw records.

The prepared table has cyclical time but no date. Add `--include-calendar` to join source dates by unique crash ID with explicit day/month/year parsing and test month/weekend in the engineered variant. This is an optional separate ablation. Calendar bundles require numeric `month` and `weekend` in prediction CSVs; the existing readable scenario interface does not supply these fields. Do not infer dates from row order or IDs. Confirm whether vehicle involvement and weather are known at the intended prediction time: this is currently a crash-severity classifier, not a proven pre-crash risk predictor.

Fresh-year evaluation needs compatible crash, vehicle, road geometry and traffic files. Keep the final year's labels out of all feature, sampling, threshold and model decisions. Report row counts, Fatal counts, road-link overlap, source versions and coverage shifts. Current experiments do not establish performance on future years.

For a crash-time density pilot, collect timestamped sensor flow and measured speed, sensor coordinates, direction and road IDs. Match only intervals ending before the crash; store time gap and spatial distance. Estimate density only with compatible flow/speed units and positive speed. Audit matched coverage and Fatal counts before training. Sensor access and representative matching are still required; AADF is average daily traffic volume and remains a proxy.

## Inference

The selected bundle works with `predict.py` on the original prepared numeric schema. The feature transformer is stored inside the pipeline and applies consistently at inference. Keep `grouped_experiments.py` beside the inference scripts when loading the bundle, and load only trusted joblib files. Existing demo defaults and historical results remain separate until there is evidence to choose a replacement.

## Pilot environment

### Engineered bundle compatibility

Engineered interaction categories now use a consistent numeric representation when fitting and scoring. Earlier code could encode integer training codes as `1:1` but float-normalized batch inputs as `1.0:1.0`, causing learned interactions to be ignored. Retrain any engineered-feature bundles produced before this correction; loading an old bundle with the corrected transformer does not migrate its fitted encoder. Original-feature bundles, including the recorded selected LR follow-up, are unaffected. Historical experiment outputs remain records of their original code and environment; rerun engineered comparisons before using them for a new selection. Regression checks cover all three model families through CSV input and model reload.

The committed pilot uses Python 3.12 and scikit-learn 1.8.0; its exact package versions are recorded in `validation/grouped_search/environment.json` and `requirements.txt`. These differ from the historical project pins. Reproduce the pilot in a separate environment using `python -m pip install -r validation/grouped_search/requirements.txt`. Do not compare differences between package versions as model improvements.

For a focused follow-up, use `--families logistic_regression --trials 15` to cover every combination in the current LR grid. The expanded decision grid tests Fatal multipliers up to 16; inspect precision alongside recall. Compressed `search.json.gz` archives retain the full candidate evidence without requiring model binaries in git.
