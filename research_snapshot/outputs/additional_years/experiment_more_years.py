"""Evaluate adding earlier years, followed by histogram gradient boosting."""
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from compare_aadf import match_crashes, TRAFFIC, POINT_ID
from train_random_forest import prepare_data, preprocessing, NUMERIC, ID, TARGET, LABELS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
YEARS = ROOT.parent / "yearly-data"
OUTPUT = ROOT.parents[1] / "outputs" / "additional_years"
OUTPUT.mkdir(parents=True, exist_ok=True)


def prepare_year(year):
    source = DATA if year == 2025 else YEARS
    geometry = DATA / "mrdb-2025.zip" if year == 2025 else YEARS / f"mrdb-{year}.zip"
    # Reuse the established preparation and link matching with a small adapter.
    # Prior-year annual traffic volume is available before the crash year ends.
    with tempfile.TemporaryDirectory(dir=YEARS) as temp:
        folder = Path(temp)
        for kind in ("collision", "vehicle"):
            src = source / f"dft-road-casualty-statistics-{kind}-{year}.csv"
            dst = folder / f"dft-road-casualty-statistics-{kind}-2025.csv"
            shutil.copyfile(src, dst)
        traffic = pd.read_csv(DATA / "dft_traffic_counts_aadf.zip", low_memory=False)
        traffic = traffic.loc[traffic.year == year - 1].copy()
        if traffic.empty:
            raise ValueError(f"No prior-year traffic data for {year}")
        traffic[["count_point_id"]].to_csv(
            folder / "count_points.zip", index=False,
            compression={"method": "zip", "archive_name": "count_points.csv"})
        traffic["year"] = 2025
        traffic.to_csv(folder / "dft_traffic_counts_aadf.zip", index=False,
                       compression={"method": "zip", "archive_name": "traffic.csv"})
        prepared, feature_columns, audit = prepare_data(folder)
        matches = match_crashes(folder, geometry)
    result = prepared.drop(columns="split").merge(
        matches[[ID, POINT_ID, TRAFFIC]], on=ID, validate="one_to_one"
    )
    result["year"] = year
    print(f"{year}: {len(result):,} matched, Fatal={(result[TARGET] == 'Fatal').sum()}", flush=True)
    return result, audit


def choose_weights(probabilities, labels, classes):
    actual = pd.Categorical(labels, categories=classes).codes
    counts = np.bincount(actual, minlength=3)
    best = None
    # The same selection objective and search grid is used for each experiment.
    for serious in [0.75, 1, 1.25, 1.5, 2, 3]:
        for fatal in [1, 2, 4, 6, 8, 12]:
            weights = np.array([{"Slight": 1, "Serious": serious, "Fatal": fatal}[c]
                                for c in classes])
            prediction = (probabilities * weights).argmax(axis=1)
            predicted = np.bincount(prediction, minlength=3)
            correct = np.bincount(actual[actual == prediction], minlength=3)
            f1 = float(np.mean(2 * correct / np.maximum(counts + predicted, 1)))
            accuracy = float(correct.sum() / len(actual))
            fatal_idx = list(classes).index("Fatal")
            recall = float(correct[fatal_idx] / counts[fatal_idx])
            # Select macro F1 without a constraint that depends on prior test scores.
            candidate = (f1, accuracy, recall, weights)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return best


def make_model(features, family, settings):
    prep = preprocessing(features, [*NUMERIC, TRAFFIC])
    if family == "logistic_regression":
        from sklearn.preprocessing import StandardScaler
        model = Pipeline([
            ("prepare", prep), ("scale", StandardScaler()),
            ("model", LogisticRegression(C=settings["C"], class_weight=settings["weight"],
                                         max_iter=1500)),
        ])
    else:
        model = Pipeline([
            ("prepare", prep),
            ("model", HistGradientBoostingClassifier(
                max_iter=180, max_leaf_nodes=settings["leaves"],
                learning_rate=settings["learning_rate"], min_samples_leaf=30,
                l2_regularization=1,
                early_stopping=False, random_state=42)),
        ])
    return model


def fit_model(model, rows, features, family, settings):
    if family == "gradient_boosting" and settings["weight"] is not None:
        weights = rows[TARGET].map(settings["weight"]).to_numpy()
        model.fit(rows[features], rows[TARGET], model__sample_weight=weights)
    else:
        model.fit(rows[features], rows[TARGET])


def score(actual, predictions):
    report = classification_report(actual, predictions, labels=LABELS,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1": float(f1_score(actual, predictions, labels=LABELS, average="macro")),
        "fatal_precision": report["Fatal"]["precision"],
        "fatal_recall": report["Fatal"]["recall"],
        "per_class": {name: report[name] for name in LABELS},
        "confusion_matrix": confusion_matrix(actual, predictions, labels=LABELS).tolist(),
    }


def experiment(data, features, train_years, family):
    train = data[data.year.isin(train_years)]
    validation = data[data.year == 2024]
    test = data[data.year == 2025]
    moderate = {"Slight": 1, "Serious": 2, "Fatal": 5}
    if family == "logistic_regression":
        configurations = [{"C": c, "weight": weight}
                          for c in [0.1, 1] for weight in [None, moderate]]
    else:
        configurations = [{"leaves": leaves, "learning_rate": rate, "weight": weight}
                          for leaves in [7, 15] for rate in [0.05, 0.1]
                          for weight in [None, moderate]]
    trials = []
    for settings in configurations:
        model = make_model(features, family, settings)
        fit_model(model, train, features, family, settings)
        probs = model.predict_proba(validation[features])
        f1, accuracy, recall, weights = choose_weights(probs, validation[TARGET], model.classes_)
        trials.append({"settings": settings, "validation_macro_f1": f1,
                       "validation_accuracy": accuracy, "validation_fatal_recall": recall,
                       "decision_weights": dict(zip(model.classes_, weights.tolist()))})
    selected = max(trials, key=lambda x: (x["validation_macro_f1"], x["validation_accuracy"]))
    model = make_model(features, family, selected["settings"])
    fit_model(model, pd.concat([train, validation]), features, family, selected["settings"])
    weights = np.array([selected["decision_weights"][c] for c in model.classes_])
    pred = model.classes_[(model.predict_proba(test[features]) * weights).argmax(axis=1)]
    result = score(test[TARGET], pred)
    name = f"{family}_{'_'.join(map(str, train_years))}"
    bundle = {"pipeline": model, "features": features, "decision_weights": selected["decision_weights"]}
    joblib.dump(bundle, OUTPUT / f"{name}.joblib", compress=3)
    loaded = joblib.load(OUTPUT / f"{name}.joblib")
    restored = loaded["pipeline"]
    assert np.array_equal(pred, restored.classes_[
        (restored.predict_proba(test[features]) * weights).argmax(axis=1)])
    print(name, result, flush=True)
    return {"train_years": train_years, "validation_year": 2024, "test_year": 2025,
            "selected": selected, "validation_trials": trials, "test": result}


def main():
    prepared, audits = [], {}
    cache = OUTPUT / "matched_2022_2025.csv.gz"
    if cache.exists():
        dataset = pd.read_csv(cache, dtype={ID: str}, float_precision="round_trip")
    else:
        for year in (2022, 2023, 2024, 2025):
            data, audit = prepare_year(year)
            prepared.append(data)
            audits[str(year)] = audit
        dataset = pd.concat(prepared, ignore_index=True)
    flags = [column for column in dataset if column.startswith("vehicle_type_")]
    dataset[flags] = dataset[flags].fillna(0)
    features = ["road_type", "weather_conditions", "road_surface_conditions",
                *NUMERIC, *sorted(flags), TRAFFIC]
    dataset = dataset[[ID, TARGET, "year", POINT_ID, *features]]
    assert dataset[ID].is_unique and dataset[TRAFFIC].notna().all()
    dataset.to_csv(OUTPUT / "matched_2022_2025.csv.gz", index=False)
    counts = pd.crosstab(dataset.year, dataset[TARGET]).to_dict(orient="index")
    print("Matched counts:", counts, flush=True)
    results = {"counts": counts, "audits": audits, "features": features, "experiments": {}}
    with threadpool_limits(limits=2):
        for family in ["logistic_regression", "gradient_boosting"]:
            for years in [[2023], [2022, 2023]]:
                results["experiments"][f"{family}_{'_'.join(map(str, years))}"] = experiment(
                    dataset, features, years, family
                )
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
