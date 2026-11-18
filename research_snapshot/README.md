# Multi-year research snapshot

This folder records the team's subsequent data-processing, model-comparison, and WebTRIS traffic-sensor experiments. Start with [the multi-year sensor results](outputs/traffic_pilot/MULTIYEAR_RESULTS.md), [data processing results](outputs/additional_years/data_processing/README.md), and [three-model comparison](outputs/additional_years/data_processing/THREE_MODEL_COMPARISON.md). These are development experiments on previously inspected years, not a fresh independent test. Read the reports for cohort selection, class imbalance and matching limitations before comparing results.

The folder structure is deliberately preserved: the scripts resolve `research_snapshot` as their workspace root. It includes the processed dataset and a gradient boosting model bundle needed by the latest sensor comparison. Other large fitted models and downloaded raw data are not included. Earlier experiments may therefore need their own input files, fitted models and packages (including XGBoost), as indicated in their scripts and reports.

## Reproduce the latest sensor experiment

Install the repository's `requirements.txt` and `scipy==1.18.1` in an isolated Python environment. Supply these external inputs:

- Place the 2022, 2023 and 2024 DfT collision and vehicle CSV files in `research_snapshot/work/yearly-data/`, with the filenames `dft-road-casualty-statistics-collision-YYYY.csv` and `dft-road-casualty-statistics-vehicle-YYYY.csv`.
- Place the corresponding DfT major road database geometry archives in that folder as `mrdb-2022.zip`, `mrdb-2023.zip`, and `mrdb-2024.zip`. Obtain source data and geometry from [DfT road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data) and [DfT road traffic downloads](https://roadtraffic.dft.gov.uk/downloads). The experiment used the geometry versions already present in the original workspace; source updates may change reproduced matches.
- Save the WebTRIS site catalogue response from `https://apptris-a-uks.azurewebsites.net/api/v1.0/sites` as `research_snapshot/work/traffic_pilot/sites.json`. The original catalogue is not bundled. Current site status and network metadata can change over time, so rerunning with a newer catalogue can produce different results.

Then from the repository root:

```powershell
python research_snapshot/outputs/traffic_pilot/multiyear_candidates.py
python research_snapshot/outputs/traffic_pilot/collect_multiyear.py
python research_snapshot/outputs/traffic_pilot/compare_multiyear.py
```

The collection step requests historical sensor records from the public [WebTRIS API](https://webtris.nationalhighways.co.uk/api/swagger/ui/index) and caches responses under `research_snapshot/work/traffic_pilot/readings/`. Availability and results can change; the 2026 service update noted that new-data updates were temporarily paused. The checked-in result JSON and CSV files preserve the original observed outputs; rerunning scripts overwrites those output files. The raw traffic-reading cache is not included.

The experiment uses collision-recorded vehicle direction to select matches and assumes time conventions described in the report; it is a retrospective research comparison rather than a validated live prediction system. The bundled model is a trusted local project artifact; do not load model files from untrusted sources.
