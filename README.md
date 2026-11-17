# Road Accident Severity Prediction

[Progress report (PDF)](Road%20Accident%20Severity%20-%20Progress%20Report.pdf) · [Three-model comparison](aadf_experiment/README.md) · [Source data and attribution](data/README.md)

## Complete workflow

Python 3.12 is recommended. Clone the project, create an isolated environment, and install the recorded dependencies:

```powershell
git clone https://github.com/shreyasNaik0101/ML-project.git
cd ML-project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS activate with `source .venv/bin/activate` instead.

### Train and compare all three models

```powershell
python train_models.py
python -m unittest discover -s tests -v
```

This command uses the included prepared matched data. It trains logistic regression, decision tree and Random Forest, with and without AADF, and saves six bundles, detailed metrics and a comparison CSV in `artifacts/`. Selection uses validation rows; final fitting uses train plus validation. Test rows are used only for evaluation. Logistic regression and decision tree share their notebook selection protocol; Random Forest keeps its earlier locked settings. These are comparisons of those configurations, rather than equally tuned model families.

### Predict a readable scenario

Edit [examples/scenario.json](examples/scenario.json), then run:

```powershell
python scenario.py --input examples/scenario.json
python scenario.py --input examples/scenario.json --model artifacts/logistic_regression_with_aadf.joblib
```

The default is logistic regression without AADF. It is a convenient demo default; no new independent test was used to select a deployment model. Road, weather, surface and vehicle names are listed in [scenario.py](scenario.py). Speed is a **limit in mph**, not vehicle speed. AADF is required by the seven-input bundle and is a typical daily traffic volume, not live traffic density. The example's AADF is an illustrative scenario value. For real cases use a measured or correctly matched value.

For batch scoring of prepared CSV rows:

```powershell
python predict.py --input examples/prepared_crashes.csv --output predictions.csv
```

Input CSVs require the bundle's feature columns. IDs are optional and preserved. Severity and split columns are ignored. Outputs contain the predicted class and raw model scores; the chosen class uses the saved class multipliers. Raw scores are **not calibrated fatality probabilities**, and can rank classes differently from the adjusted decision. Load only trusted joblib files.

[Research paper](docs/paper/paper.pdf) · [Verified results](docs/RESULTS.md) · [Eight literature reviews](docs/LITERATURE.md) · [Demo and viva guide](docs/DEMO.md)

### Evaluate unseen road links

```powershell
python evaluate_unseen_links.py
```

This uses the included raw source files to rematch road links and keep each count-point link in one partition. All three model families are selected within the new development partitions. Audits and results go to `validation/unseen_links/`; regenerated datasets and bundles are excluded from version control. The report contains the executed metrics. This evaluates new links within the same already inspected source year, not a future population.

### Improve models with grouped cross-validation

See [measured follow-up results](validation/RESULTS.md) and [the experiment guide](docs/EXPERIMENTS.md) for the development-only search, data audit, sampling comparisons, selection requirements and remaining external-data work.

```powershell
python audit_data.py
python grouped_experiments.py --trials 3 --sampling none over under
```

The search uses fixed three-fold road-link groups and fits preprocessing inside each fold. It saves per-class trade-offs and a selected bundle separately from the historical models. Final scoring is opt-in with `--evaluate-inspected-test`; that test has already been inspected and is not fresh external validation.

### Original preparation and notebooks

```powershell
python train_random_forest.py --data-dir data
python tune_random_forest.py
python compare_aadf.py --data-dir data --geometry data/mrdb-2025.zip
python -m jupyterlab aadf_experiment
```

The original scripts overwrite their historical outputs when run without a separate `--output-dir`. The new `train_models.py` writes to `artifacts/` and preserves the historical experiments. The notebook results below describe the original run; small numerical differences may occur across solver/library environments.

Random Forest experiment using Great Britain’s 2025 STATS19 police-reported injury crashes. The unit of prediction is one crash, with the recorded label Slight, Serious, or Fatal. Data and field definitions are available from the [Department for Transport](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data).

## Data preparation

- Checked the crash ID and the crash/vehicle ID pair for duplicates. The two source files contained 101,525 crashes and 183,948 vehicle records. No duplicate keys or unmatched records were found.
- Converted vehicle rows to binary presence flags by `collision_index`. For example, `vehicle_type_9` flags a car. A crash can have several flags. Nine vehicle rows had a missing or out-of-range type and were captured by `vehicle_type_missing`. Code 99, the explicit self-reported unknown type, has its own flag.
- Used six input groups: road type, weather, road surface, speed limit, time, and vehicle type. The source files have no direct traffic-density measurement, so the model omits that feature. The feature list also excludes crash IDs, injury counts, and severity-derived columns.
- Preserved DfT’s explicit unknown categories. Recoded 634 road-surface values marked missing or out of range. The pipeline assigns a separate missing category and one-hot encodes road type, weather, and road surface. It uses speed limit in mph and two cyclical time features to keep times around midnight close together. Numeric median imputation is fitted within the pipeline. Speed limit and time were complete after validation.

## Model selection and test results

The stratified split used seed 42 and assigned 64,976 crashes to train, 16,244 to validation, and 20,305 to test. Training compared depths 8 and 14 under three weighting schemes: unweighted, moderate custom weights, and inverse-frequency class weights. Each candidate used 150 trees and at least 10 training samples per leaf. The depth-14 forest with balanced weights had the highest validation macro F1 (0.353). It was refitted on train plus validation before testing.

| Model | Test macro F1 | Test accuracy |
|---|---:|---:|
| Majority baseline (always Slight) | 0.283 | 73.76% |
| Random Forest | 0.349 | 53.59% |

| Actual severity | Precision | Recall | F1 | Test crashes |
|---|---:|---:|---:|---:|
| Slight | 79.66% | 63.76% | 0.708 | 14,976 |
| Serious | 33.27% | 23.18% | 0.273 | 5,038 |
| Fatal | 3.45% | 57.04% | 0.065 | 291 |

The forest detected 166 of 291 fatal crashes. It predicted Fatal for 4,808 crashes, including 4,642 that were Slight or Serious. It also missed 125 fatal crashes, of which 66 were predicted Slight. The macro F1 gain over the majority baseline came with lower accuracy and many false fatal predictions. The test split estimates performance on held-out 2025 crashes from the same dataset. Performance on future years remains unmeasured.

See [severity distribution](severity_distribution.png), [test confusion matrix](confusion_matrix.png), and [full metrics and validation trials](metrics.json).

## Follow-up tuning

The follow-up used the same six input groups and split. It selected the forest settings and the Serious and Fatal score multipliers by validation macro F1. The selected forest used moderate class weights, with a maximum depth of 14. At prediction time it multiplied the forest scores by 1.63 for Serious and 3.95 for Fatal, with Slight fixed at 1.00, then picked the highest adjusted score.

| Model | Test accuracy | Macro F1 | Fatal precision | Fatal recall |
|---|---:|---:|---:|---:|
| Original Random Forest | 53.59% | 0.349 | 3.45% | 57.04% |
| Tuned Random Forest | 69.84% | 0.409 | 6.95% | 10.65% |

The tuned model made 446 fatal predictions, 31 of which were correct. Its overall scores improved while the number of correctly identified fatal crashes fell from 166 to 31. The shared 2025 test split had already been used to evaluate the original model. A future year would provide a fresh comparison.

Run `python tune_random_forest.py` from this directory after the original training run. The saved `tuned_random_forest.joblib` contains the pipeline, the ordered feature list, and a class-to-multiplier mapping. For predictions, call the pipeline’s `predict_proba`, multiply each column by the corresponding weight in `pipeline.classes_` order, and select the class with the highest adjusted score. See the [tuned confusion matrix](tuned_confusion_matrix.png) and [tuning results](tuned_metrics.json).

## Share the split with your teammates

Give each teammate `team_crashes.csv.gz`. Pandas reads it directly. The `split` column assigns each crash to the same partition used in this experiment.

```python
import pandas as pd

data = pd.read_csv("team_crashes.csv.gz", dtype={"collision_index": str})
features = data.columns.drop(["collision_index", "split", "severity"]).tolist()
train = data.loc[data["split"] == "train"]
validation = data.loc[data["split"] == "validation"]
test = data.loc[data["split"] == "test"]
X_train, y_train = train[features], train["severity"]
```

Treat road type, weather, and road surface codes as categories. Fit encoders, imputers, and any scaling on the training rows. Select model settings on the validation rows, refit the selected pipeline on train plus validation, and evaluate on the test rows. Use macro F1, per-class precision and recall, and confusion matrices with the class order Slight, Serious, Fatal.

## Reproduce the run

With Python 3.12, install the package versions recorded in `metrics.json`. From the folder containing the training script, run:

```powershell
python -m pip install numpy==2.5.4 pandas==3.0.1 scikit-learn==1.9.1 joblib==1.6.0 matplotlib==3.11.2
python train_random_forest.py --data-dir "C:\Users\shrey\ML-Traffic-Project"
```

Use `--output-dir` to choose a separate destination for a rerun. The saved `random_forest.joblib` contains both the fitted preprocessing and the classifier. It expects the prepared feature columns listed in `metrics.json`. The run verified that reloading the saved pipeline reproduced the test predictions.
