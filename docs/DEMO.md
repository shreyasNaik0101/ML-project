# Demonstration and viva guide

## Start from a fresh clone

```powershell
git clone https://github.com/shreyasNaik0101/ML-project.git
cd ML-project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scenario.py --input examples/scenario.json
```

The repository contains six model bundles, so the first scenario does not require retraining. Edit the time, weather and vehicle involvement in `examples/scenario.json` and run again. Use only supported categories documented in `scenario.py`.

## Three minute walkthrough

**0:00–0:30 Problem.** “We classify the recorded severity of an injury crash as Slight, Serious or Fatal. We use road conditions, time, vehicle involvement and the posted speed limit, and test a daily traffic-volume proxy.”

**0:30–1:00 Data.** Open the README's preparation section. Explain one row per crash, multiple vehicle flags, cyclical clock features and excluded outcome-derived fields. Mention that only 19.1% of source crashes match the major-road traffic sample.

**1:00–1:40 Prediction.** Open `examples/scenario.json` and run the scenario command. Explain the predicted class and raw scores. The class uses saved decision weights; the raw scores are not calibrated fatality risks. Run `python predict.py --input examples/prepared_crashes.csv --output predictions.csv` to show batch scoring.

**1:40–2:20 Results.** Open `docs/paper/paper.pdf`. Show that accuracy alone favors the majority baseline. Explain logistic regression's macro F1 and the seven detected fatalities among 49 test fatalities. State that adding AADF did not improve its macro F1.

**2:20–3:00 Verification.** Show the second table, which keeps entire road links outside training and retunes within the new partitions. Run `python -m unittest discover -s tests -v`. Close with the remaining empirical limitation: a future-year external dataset has not been tested.

## Commands for a complete rerun

```powershell
python train_models.py
python evaluate_unseen_links.py
python -m unittest discover -s tests -v
```

The second command rematches road geometries and retrains the grouped experiment. It writes regenerated data and bundles to ignored paths and records partition and class counts.

## Questions to be ready for

**Why not accuracy alone?** Slight dominates the test population. An always-Slight rule is 78.10% accurate on the fixed split and catches no fatalities. Macro F1 and per-class counts expose that failure.

**Is AADF traffic density?** No. It is annual average daily motor-vehicle flow. It is a tested proxy for typical volume and does not measure crash-time congestion or instantaneous density.

**Why three models?** LR provides a linear baseline, the tree captures simple nonlinear rules, and RF aggregates trees. Comparisons use shared rows within each experiment, although the historical forest and LR/tree have different selection budgets.

**Why use multipliers?** They select a validation trade-off between classes. They are not probability calibration, and the final class can differ from the largest raw score.

**What prevents leakage?** IDs, target, split assignments and outcome-derived fields are excluded. The runner rejects unexpected predictor columns. Preprocessors fit on the development partition appropriate to each stage. The grouped experiment additionally keeps count-point links disjoint.

**Does the link test prove deployment readiness?** No. Adjacent roads can remain correlated and the source year has already been inspected. The result is an additional sensitivity analysis, not prospective validation.

**Can it be used in Bangalore?** Indian performance is not established. A local external dataset and a use-case assessment are required before that claim.

**What is the strongest result?** The reproducible comparison and explicit limitations. The model still misses most fatal cases and should not be presented as an emergency-response system.
