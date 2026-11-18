"""Select a model and decision rule for Slight and Serious recall on 2024."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
LABELS = ['Slight', 'Serious', 'Fatal']


def metrics(y, prediction):
    return dict(accuracy=float(accuracy_score(y, prediction)),
                per_class=classification_report(y, prediction, labels=LABELS,
                    output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(y, prediction, labels=LABELS).tolist())


def main():
    data = pd.read_csv(ROOT / 'matched_2022_2025.csv.gz', float_precision='round_trip')
    train, val, test = [data[data.year.isin(years)] for years in
                        [[2022, 2023], [2024], [2025]]]
    original = joblib.load(ROOT / 'gradient_boosting_2022_2023.joblib')
    features = original['features']
    baseline = clone(original['pipeline'])
    baseline.fit(train[features], train.severity)
    classes = baseline.classes_.tolist()
    probabilities = baseline.predict_proba(val[features])
    old_weights = np.array([original['decision_weights'][c] for c in classes])
    reference = metrics(val.severity, np.array(classes)[(probabilities * old_weights).argmax(1)])
    reference_recalls = np.array([reference['per_class'][c]['recall'] for c in LABELS[:2]])
    truth = pd.Categorical(val.severity, categories=classes).codes
    counts = np.bincount(truth, minlength=3)
    candidates = [baseline]
    for leaves in [15, 31]:
        model = clone(baseline).set_params(model__max_leaf_nodes=leaves,
                                            model__learning_rate=0.05, model__max_iter=250)
        model.fit(train[features], train.severity)
        candidates.append(model)
    trials = []
    best = None
    for index, model in enumerate(candidates):
        probs = model.predict_proba(val[features])
        for serious in np.geomspace(0.5, 6, 35):
            for fatal in [0.1, 0.5, 1, 2, 4, 8, 12]:
                weights = np.array([{'Slight': 1, 'Serious': serious, 'Fatal': fatal}[c]
                                    for c in classes])
                pred = (probs * weights).argmax(1)
                correct = np.bincount(truth[pred == truth], minlength=3)
                recall = correct / counts
                if recall[classes.index('Fatal')] < reference['per_class']['Fatal']['recall']:
                    continue
                pair = np.array([recall[classes.index(c)] for c in LABELS[:2]])
                gains = pair - reference_recalls
                rank = (float(gains.min()), float(pair.mean()))
                if best is None or rank > best[0]:
                    best = rank, index, weights
        print('Validated tree size', model.named_steps['model'].max_leaf_nodes, flush=True)
    rank, index, weights = best
    selected = candidates[index]
    val_predictions = np.array(classes)[(selected.predict_proba(val[features]) * weights).argmax(1)]
    development = pd.concat([train, val])
    selected.fit(development[features], development.severity)
    predictions = np.array(classes)[(selected.predict_proba(test[features]) * weights).argmax(1)]
    baseline_test = np.array(classes)[(original['pipeline'].predict_proba(test[features]) * old_weights).argmax(1)]
    bundle = dict(pipeline=selected, features=features,
                  decision_weights=dict(zip(classes, weights.tolist())))
    joblib.dump(bundle, ROOT / 'slight_serious_model.joblib', compress=3)
    restored = joblib.load(ROOT / 'slight_serious_model.joblib')
    np.testing.assert_array_equal(predictions, np.array(classes)[
        (restored['pipeline'].predict_proba(test[features]) * weights).argmax(1)])
    results = dict(selection_objective='Maximize the smaller validation recall gain for Slight and Serious relative to the previous GB model; then their mean recall. Retain at least baseline Fatal recall on validation.',
        baseline_validation=reference, selected_validation=metrics(val.severity, val_predictions),
        baseline_test=metrics(test.severity, baseline_test), selected_test=metrics(test.severity, predictions),
        decision_weights=bundle['decision_weights'],
        max_leaf_nodes=selected.named_steps['model'].max_leaf_nodes,
        learning_rate=selected.named_steps['model'].learning_rate,
        max_iter=selected.named_steps['model'].max_iter)
    (ROOT / 'slight_serious_metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
