"""Select Random Forest class decision weights using the shared validation split."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copyfile

import joblib
import numpy as np
import pandas as pd

from train_random_forest import (
    ID, LABELS, MODERATE_WEIGHTS, TARGET, build_forest, evaluate, make_figures,
)


def tune_decision_weights(probabilities, actual, classes):
    """Keep Slight fixed and search multipliers for Serious and Fatal."""
    actual_codes = pd.Categorical(actual, categories=classes).codes
    class_counts = np.bincount(actual_codes, minlength=len(classes))
    serious, fatal = [classes.index(label) for label in ("Serious", "Fatal")]
    multipliers = sorted(set(np.geomspace(0.04, 8, 31).tolist() + [1.0]))
    best = None

    for serious_weight in multipliers:
        for fatal_weight in multipliers:
            weights = np.ones(len(classes))
            weights[serious], weights[fatal] = serious_weight, fatal_weight
            predicted = (probabilities * weights).argmax(axis=1)
            prediction_counts = np.bincount(predicted, minlength=len(classes))
            correct_counts = np.bincount(
                actual_codes[actual_codes == predicted], minlength=len(classes)
            )
            macro_f1 = (2 * correct_counts / (class_counts + prediction_counts)).mean()
            accuracy = correct_counts.sum() / len(actual)
            score = (float(macro_f1), float(accuracy))
            if best is None or score > best[0]:
                best = score, weights
    return best


def main():
    directory = Path(__file__).resolve().parent
    original = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    data = pd.read_csv(directory / "team_crashes.csv.gz", dtype={ID: str},
                       float_precision="round_trip")
    features = original["features"]
    train = data.loc[data["split"] == "train"]
    validation = data.loc[data["split"] == "validation"]
    test = data.loc[data["split"] == "test"]
    development = data.loc[data["split"] != "test"]
    candidates = []

    for depth in (8, 14):
        for training_weight in (None, MODERATE_WEIGHTS, "balanced"):
            model = build_forest(features, depth, training_weight, jobs=4)
            model.fit(train[features], train[TARGET])
            classes = model.classes_.tolist()
            probabilities = model.predict_proba(validation[features])
            score, weights = tune_decision_weights(probabilities, validation[TARGET], classes)
            candidate = {
                "depth": depth,
                "training_class_weight": training_weight,
                "decision_weights": dict(zip(classes, weights.tolist())),
                "validation_macro_f1": score[0],
                "validation_accuracy": score[1],
            }
            candidates.append(candidate)
            print(candidate, flush=True)

    selected = max(candidates, key=lambda item: (
        item["validation_macro_f1"], item["validation_accuracy"]
    ))
    model = build_forest(features, selected["depth"], selected["training_class_weight"], jobs=4)
    model.fit(development[features], development[TARGET])
    model.set_params(model__n_jobs=1)
    weights = np.asarray([selected["decision_weights"][label] for label in model.classes_])
    predictions = model.classes_[(model.predict_proba(test[features]) * weights).argmax(axis=1)]
    test_result = evaluate(test[TARGET], predictions)

    artifact = {
        "pipeline": model,
        "decision_weights": selected["decision_weights"],
        "features": features,
    }
    path = directory / "tuned_random_forest.joblib"
    joblib.dump(artifact, path, compress=3)
    restored = joblib.load(path)
    pipeline = restored["pipeline"]
    restored_weights = [restored["decision_weights"][label] for label in pipeline.classes_]
    restored_predictions = pipeline.classes_[
        (pipeline.predict_proba(test[restored["features"]]) * restored_weights).argmax(axis=1)
    ]
    np.testing.assert_array_equal(predictions, restored_predictions)

    result = {
        "label_order": LABELS,
        "selected": selected,
        "validation_candidates": candidates,
        "tuned_random_forest_test": test_result,
        "original_random_forest_test": original["random_forest_test"],
        "majority_baseline_test": original["majority_baseline_test"],
        "test_note": "Same holdout as the original experiment. The tuning procedure used only "
                     "train and validation rows. A future-year test would provide a fresh benchmark.",
    }
    (directory / "tuned_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with TemporaryDirectory(dir=directory) as temp:
        make_figures(data, test_result, Path(temp))
        copyfile(Path(temp) / "confusion_matrix.png", directory / "tuned_confusion_matrix.png")
    print(json.dumps(test_result, indent=2))


if __name__ == "__main__":
    main()
