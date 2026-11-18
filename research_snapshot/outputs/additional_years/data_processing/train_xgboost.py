"""Train XGBoost with temporal validation using the existing processed crash data."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
LABELS = np.array(['Fatal', 'Serious', 'Slight'])


def evaluate(y, predicted):
    return {'accuracy': float(accuracy_score(y, predicted)),
            'report': classification_report(y, predicted, labels=LABELS.tolist(),
                                             output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, predicted, labels=LABELS.tolist()).tolist()}


def best_weights(probabilities, actual):
    actual_code = pd.Categorical(actual, categories=LABELS).codes
    actual_counts = np.bincount(actual_code, minlength=3)
    best = None
    for serious in [0.75, 1, 1.25, 1.5, 2, 3]:
        for fatal in [1, 2, 4, 6, 8, 12]:
            weights = np.array([fatal, serious, 1.0])
            predicted = (probabilities * weights).argmax(axis=1)
            pred_counts = np.bincount(predicted, minlength=3)
            correct = np.bincount(actual_code[predicted == actual_code], minlength=3)
            score = float(np.mean(2 * correct / np.maximum(actual_counts + pred_counts, 1)))
            if best is None or score > best[0]:
                best = score, weights
    return best


def build(preprocessor, depth, rate):
    return Pipeline([
        ('prepare', clone(preprocessor)),
        ('model', XGBClassifier(n_estimators=250, max_depth=depth, learning_rate=rate,
            min_child_weight=10, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, objective='multi:softprob', num_class=3,
            tree_method='hist', random_state=42, n_jobs=2)),
    ])


def main():
    data = pd.read_csv(ROOT / 'processed_crashes.csv.gz', dtype={'collision_index': str},
                       float_precision='round_trip')
    gb = joblib.load(ROOT / 'vehicle_composition_optional.joblib')
    features = gb['features']
    train = data[data.year.isin([2022, 2023])]
    validation = data[data.year == 2024]
    test = data[data.year == 2025]
    target_train = pd.Categorical(train.severity, categories=LABELS).codes
    assert set(target_train) == {0, 1, 2}
    preprocessor = gb['pipeline'].named_steps['prepare']
    trials = []
    for depth in (3, 5):
        for rate in (0.05, 0.1):
            for weighting in ('none', 'moderate'):
                model = build(preprocessor, depth, rate)
                weights = train.severity.map({'Slight': 1, 'Serious': 2, 'Fatal': 5}).to_numpy()
                fit_args = {'model__sample_weight': weights} if weighting == 'moderate' else {}
                model.fit(train[features], target_train, **fit_args)
                probabilities = model.predict_proba(validation[features])
                score, decision = best_weights(probabilities, validation.severity)
                trials.append({'max_depth': depth, 'learning_rate': rate,
                               'sample_weighting': weighting, 'validation_macro_f1': score,
                               'decision_weights': dict(zip(LABELS.tolist(), decision.tolist()))})
                print(trials[-1], flush=True)
    selected = max(trials, key=lambda trial: trial['validation_macro_f1'])
    development = data[data.year < 2025]
    y_dev = pd.Categorical(development.severity, categories=LABELS).codes
    model = build(preprocessor, selected['max_depth'], selected['learning_rate'])
    weights = development.severity.map({'Slight': 1, 'Serious': 2, 'Fatal': 5}).to_numpy()
    fit_args = {'model__sample_weight': weights} if selected['sample_weighting'] == 'moderate' else {}
    model.fit(development[features], y_dev, **fit_args)
    decision = np.array([selected['decision_weights'][c] for c in LABELS])
    predictions = LABELS[(model.predict_proba(test[features]) * decision).argmax(axis=1)]
    bundle = {'pipeline': model, 'features': features, 'class_order': LABELS.tolist(),
              'decision_weights': selected['decision_weights'],
              'versions': {'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__}}
    joblib.dump(bundle, ROOT / 'xgboost.joblib', compress=3)
    loaded = joblib.load(ROOT / 'xgboost.joblib')
    reloaded = LABELS[(loaded['pipeline'].predict_proba(test[features]) * decision).argmax(axis=1)]
    np.testing.assert_array_equal(predictions, reloaded)
    old_predictions = gb['pipeline'].classes_[
        (gb['pipeline'].predict_proba(test[features]) *
         np.array([gb['decision_weights'][c] for c in gb['pipeline'].classes_])).argmax(axis=1)]
    results = {'selected': selected, 'validation_trials': trials,
               'xgboost_test': evaluate(test.severity, predictions),
               'gradient_boosting_test': evaluate(test.severity, old_predictions),
               'class_order': LABELS.tolist(), 'test_rows': len(test),
               'note': '2025 has been inspected in earlier project experiments; roads may recur across years.',
               'versions': bundle['versions']}
    (ROOT / 'xgboost_results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
