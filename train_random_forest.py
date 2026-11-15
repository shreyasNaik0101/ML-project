"""Prepare STATS19 2025 crash data and train and evaluate a Random Forest."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEED = 42
ID = "collision_index"
TARGET = "severity"
LABELS = ["Slight", "Serious", "Fatal"]
SEVERITY_CODES = {1: "Fatal", 2: "Serious", 3: "Slight"}
MODERATE_WEIGHTS = {"Slight": 1.0, "Serious": 1.5, "Fatal": 4.0}
CATEGORICAL = ["road_type", "weather_conditions", "road_surface_conditions"]
NUMERIC = ["speed_limit", "time_sin", "time_cos"]
PREFIX = "dft-road-casualty-statistics"


def prepare_data(source):
    collisions = pd.read_csv(
        source / f"{PREFIX}-collision-2025.csv",
        usecols=[ID, "collision_severity", "time", "speed_limit", *CATEGORICAL],
        dtype={ID: str},
    )
    vehicles = pd.read_csv(
        source / f"{PREFIX}-vehicle-2025.csv",
        usecols=[ID, "vehicle_reference", "vehicle_type"],
        dtype={ID: str},
    )
    collision_rows, vehicle_rows = len(collisions), len(vehicles)
    collisions = collisions.drop_duplicates().copy()
    vehicles = vehicles.drop_duplicates().copy()
    if collisions[ID].isna().any() or not collisions[ID].is_unique:
        raise ValueError("Missing or conflicting collision identifiers")
    if vehicles[[ID, "vehicle_reference"]].isna().any().any() or vehicles.duplicated(
        [ID, "vehicle_reference"]
    ).any():
        raise ValueError("Missing or conflicting vehicle identifiers")

    collisions[TARGET] = collisions.pop("collision_severity").map(SEVERITY_CODES)
    if collisions[TARGET].isna().any():
        raise ValueError("Unexpected collision severity code")
    for column in CATEGORICAL:
        codes = pd.to_numeric(collisions[column], errors="coerce")
        collisions[column] = codes.where(codes.ge(0))

    speed = pd.to_numeric(collisions["speed_limit"], errors="coerce")
    collisions["speed_limit"] = speed.where(speed.isin([20, 30, 40, 50, 60, 70]))
    clock = pd.to_datetime(collisions.pop("time"), format="%H:%M", errors="coerce")
    angle = 2 * np.pi * (clock.dt.hour * 60 + clock.dt.minute) / (24 * 60)
    collisions["time_sin"], collisions["time_cos"] = np.sin(angle), np.cos(angle)

    # A multi-vehicle crash can have several vehicle-type flags set to 1.
    type_codes = pd.to_numeric(vehicles["vehicle_type"], errors="coerce")
    vehicle_categories = type_codes.where(type_codes.ge(0)).astype("Int64").astype("string")
    vehicles["type"] = vehicle_categories.fillna("missing")
    flags = pd.crosstab(vehicles[ID], vehicles["type"]).gt(0).astype("uint8")
    flags.columns = [f"vehicle_type_{code}" for code in flags.columns]
    unmatched = int((~collisions[ID].isin(flags.index)).sum())
    orphan_vehicles = int((~vehicles[ID].isin(collisions[ID])).sum())
    data = collisions.merge(flags, how="left", on=ID, validate="one_to_one")
    data[list(flags.columns)] = data[list(flags.columns)].fillna(0).astype("uint8")
    data["vehicle_record_missing"] = (~data[ID].isin(flags.index)).astype("uint8")
    vehicle_features = list(flags.columns)
    if unmatched:
        vehicle_features.append("vehicle_record_missing")
    features = [*CATEGORICAL, *NUMERIC, *vehicle_features]

    train_val, test = train_test_split(
        data, test_size=0.20, stratify=data[TARGET], random_state=SEED
    )
    train, val = train_test_split(
        train_val, test_size=0.20, stratify=train_val[TARGET], random_state=SEED
    )
    split = pd.concat([
        train.assign(split="train"),
        val.assign(split="validation"),
        test.assign(split="test"),
    ])[[ID, "split", TARGET, *features]].sort_values(ID).reset_index(drop=True)
    audit = {
        "collision_source_rows": collision_rows,
        "vehicle_source_rows": vehicle_rows,
        "collision_duplicates_removed": collision_rows - len(collisions),
        "vehicle_duplicates_removed": vehicle_rows - len(vehicles),
        "collisions_without_vehicle_records": unmatched,
        "vehicle_rows_without_collision": orphan_vehicles,
        "vehicle_rows_with_missing_type": int(vehicle_categories.isna().sum()),
        "missing_values_after_recoding": split[features].isna().sum().to_dict(),
    }
    return split, features, audit


def preprocessing(features, numeric=NUMERIC):
    vehicle_features = [f for f in features if f not in CATEGORICAL and f not in numeric]
    categories = Pipeline([
        ("missing", SimpleImputer(strategy="constant", fill_value=-1)),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("categories", categories, CATEGORICAL),
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("vehicles", "passthrough", vehicle_features),
    ])


def build_forest(features, depth, weight, jobs, numeric=NUMERIC):
    return Pipeline([
        ("preprocess", preprocessing(features, numeric)),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=depth,
            min_samples_leaf=10,
            class_weight=weight,
            random_state=SEED,
            n_jobs=jobs,
        )),
    ])


def evaluate(actual, predicted):
    report = classification_report(
        actual, predicted, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def make_figures(data, results, destination):
    colors = ["#177e89", "#e3a13c", "#b74743"]
    counts = data[TARGET].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    bars = ax.barh(LABELS, counts, color=colors)
    ax.bar_label(bars, [f"{n:,}  ({n / counts.sum():.1%})" for n in counts], padding=6)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.34)
    ax.set_xlabel("Recorded crashes")
    ax.set_title("Crash severity distribution | STATS19 2025", loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(destination / "severity_distribution.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray(results["confusion_matrix"])
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.0), layout="constrained")
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set(xlabel="Predicted severity", ylabel="Actual severity")
    ax.set_title("Random Forest | Held-out crashes", loc="left", weight="bold", pad=14)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{matrix[row, col]:,}\n{shares[row, col]:.1%}",
                    ha="center", va="center", fontsize=12,
                    color="white" if shares[row, col] > 0.55 else "#172a3a")
    fig.savefig(destination / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, features, audit = prepare_data(args.data_dir)
    train = data.loc[data["split"] == "train"]
    val = data.loc[data["split"] == "validation"]
    test = data.loc[data["split"] == "test"]
    development = data.loc[data["split"] != "test"]
    data.to_csv(args.output_dir / "team_crashes.csv.gz", index=False)
    print(f"Split sizes: {len(train):,} train, {len(val):,} validation, {len(test):,} test",
          flush=True)

    candidates = []
    for depth in (8, 14):
        for weight in (None, MODERATE_WEIGHTS, "balanced"):
            model = build_forest(features, depth, weight, args.jobs)
            model.fit(train[features], train[TARGET])
            result = evaluate(val[TARGET], model.predict(val[features]))
            candidate = {
                "depth": depth,
                "class_weight": weight,
                "macro_f1": result["macro_f1"],
                "fatal_recall": result["per_class"]["Fatal"]["recall"],
                "fatal_precision": result["per_class"]["Fatal"]["precision"],
            }
            candidates.append(candidate)
            print(candidate, flush=True)

    selected = max(candidates, key=lambda candidate: candidate["macro_f1"])
    model = build_forest(features, selected["depth"], selected["class_weight"], args.jobs)
    model.fit(development[features], development[TARGET])
    predicted = model.predict(test[features])
    result = evaluate(test[TARGET], predicted)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(development[[ID]], development[TARGET])
    baseline_result = evaluate(test[TARGET], baseline.predict(test[[ID]]))

    model.set_params(model__n_jobs=1)
    model_path = args.output_dir / "random_forest.joblib"
    joblib.dump(model, model_path, compress=3)
    np.testing.assert_array_equal(joblib.load(model_path).predict(test[features]), predicted)

    metrics = {
        "label_order": LABELS,
        "features": features,
        "seed": SEED,
        "split_counts": pd.crosstab(data["split"], data[TARGET]).to_dict(orient="index"),
        "audit": audit,
        "validation_candidates": candidates,
        "selected_settings": model.named_steps["model"].get_params(),
        "random_forest_test": result,
        "majority_baseline_test": baseline_result,
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    make_figures(data, result, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
