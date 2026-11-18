# Adding 2021 crash data

DfT 2021 collision and vehicle records were matched to 2021 road geometry and 2020 annual average traffic, using the existing road-number, 50-metre distance and ambiguity rules. The match yielded 20,083 crashes: 16,271 Slight, 3,596 Serious and 216 Fatal. Other years and the test set were unchanged. The broader matched dataset is saved in matched_2021_2025.csv.gz.

Each experiment trained on 2022-2023, adding either all of 2021 or only Serious and Fatal crashes from 2021. It selected decision multipliers on the unchanged 2024 validation set. The final fit included those training records plus 2024, and evaluation used the same 19,062 previously inspected 2025 crashes. The comparison uses the seven input groups, vehicle-type counts, and time features. AADF is a prior-year average daily traffic-volume proxy.

| Model and 2021 strategy | 2025 accuracy | Macro F1 | Slight recall | Serious recall | Fatal recall |
|---|---:|---:|---:|---:|---:|
| Random Forest, no 2021 | 68.89% | 0.431 | 78.94% | 36.26% | 22.06% (45/204) |
| Random Forest, all 2021 | 70.19% | 0.431 | 81.58% | 33.02% | 22.06% (45/204) |
| Random Forest, Serious and Fatal 2021 only | 73.81% | 0.417 | 89.16% | 23.56% | 11.27% (23/204) |
| XGBoost, no 2021 | 66.95% | 0.430 | 73.83% | 45.60% | 14.71% (30/204) |
| XGBoost, all 2021 | 68.51% | 0.428 | 76.98% | 41.83% | 12.75% (26/204) |
| XGBoost, Serious and Fatal 2021 only | 70.61% | 0.422 | 82.77% | 30.81% | 21.08% (43/204) |

Each scenario chooses its decision weights on validation, so differences combine the effect of training records and the selected decision rule. The models have fixed previously selected hyperparameters and class-weight settings. The Serious/Fatal-only strategy changes the training class distribution. Its higher accuracy does not show improved results for every class.

2025 was inspected in earlier work. Road links may appear in multiple years. These are retrospective comparisons and not fresh external validation. Severity reporting methods and vehicle codes changed across years. A future-year test and systematic review of older label consistency would strengthen the conclusions.

The full 2021 audit, class counts, selected weights, validation scores and test confusion matrices are in additional_2021_results.json. The six model bundles are named by model and strategy. Each was reloaded and its test predictions checked. data_processing/add_2021.py contains the preparation and experiment code, which depends on the current local project modules and DfT source files.

DfT has removed the old individual 2018-2020 links used in earlier releases. Those years were not added here. They remain in DfT's larger historical dataset and would require a separate extraction and compatibility check.

Sources: [DfT road safety data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data) and [DfT traffic data](https://roadtraffic.dft.gov.uk/downloads).
