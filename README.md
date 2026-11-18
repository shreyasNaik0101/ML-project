# Road Accident Severity Prediction - Final Notebook Demonstration

This branch is the clean demonstration of our selected Logistic Regression, Random Forest and XGBoost models. It uses the full 2021-2025 project dataset and the seven input groups specified in the problem statement.

## Run the notebooks

Use Python 3.12 and open a terminal in this repository folder. On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyter lab
```

If PowerShell blocks activation, use Command Prompt with `.venv\Scripts\activate.bat`. In Jupyter, select the **Python 3** kernel from this project's environment. Start Jupyter in the project folder, and run each notebook's cells from the top in order:

1. `01_Data_Preparation.ipynb` - source records, seven inputs, class imbalance and the time split.
2. `02_Logistic_Regression.ipynb` - the linear baseline.
3. `03_Random_Forest.ipynb` - the tree ensemble.
4. `04_XGBoost.ipynb` - the boosted-tree model.

After completing setup, Windows users can also double-click `Start_Demo.bat` in this project folder to start JupyterLab.

Each model notebook is self-contained. It trains on 2021-2023, chooses decision multipliers using 2024, refits on 2021-2024, and evaluates on 2025. Each model is fitted twice; the Random Forest notebook can take several minutes. Practice a full run on the demonstration computer and keep saved outputs visible in case of delays. Saved outputs show earlier execution; they are not evidence of a fresh live run.

## Main files

- `data/final_matched_2021_2025.csv.gz`: prepared collision-level modelling data used by all notebooks. It is already linked and feature-engineered; the data notebook does not rerun the historical geospatial matching process.
- `data/raw/`: public collision, vehicle, AADF and road-geometry source files used in this project. See [data/README.md](data/README.md) for provenance and the preparation performed previously.
- `results/final_model_comparison.csv`: the previously recorded results for the same 2025 collision rows. Running a notebook calculates and prints its own current metrics separately.
- `requirements.txt`: package versions from the project environment.

## Recorded results with the full 2021 data

| Model | Accuracy | Slight recall | Serious recall | Fatal recall |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 70.60% | 82.89% | 30.79% | 12.25% |
| Random Forest | 70.19% | 81.58% | 33.02% | 22.06% |
| XGBoost | 68.51% | 76.98% | 41.83% | 12.75% |

Class recall measures the proportion of actual cases of that class correctly identified. The 2025 cohort was inspected during development and is not an untouched final test. Predicting every collision as Slight would achieve approximately 76.80% accuracy on this cohort, but detect no Serious or Fatal collisions.

The original experiments are preserved on the `experiment/traffic-sensor-multiyear` branch. This branch is a presentation-friendly selection, not a replacement for that history.

## Key points for the viva

- **Target:** Slight, Serious or Fatal. Slight corresponds to minor in the project statement.
- **Seven input groups:** road type, weather, time, traffic density, vehicle type, speed limit and road conditions. AADF is annual average daily flow and is a traffic-volume proxy, not direct instantaneous density.
- **Imbalance:** serious and fatal crashes are much rarer, so show class-specific precision and recall, not accuracy alone.
- **No preprocessing leakage:** the imputer and encoder are inside each model pipeline and are fitted only on the rows in the current training fit.
- **Decision multipliers:** selected using validation-year macro F1 to trade off classes; adjusted scores are not calibrated probabilities.
- **Scope:** vehicle composition may not be known before a collision; matched major roads do not represent all roads. The 2025 cohort is previously inspected, and a fresh final test is still needed.
