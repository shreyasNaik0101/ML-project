"""Compare three classifiers with shared data, features and selection procedure."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_xgboost import LABELS, best_weights, build, evaluate

ROOT = Path(__file__).resolve().parent


def make_model(family, parameter, preprocessor):
    if family == 'XGBoost':
        return build(preprocessor, parameter, 0.05)
    if family == 'Random Forest':
        estimator = RandomForestClassifier(n_estimators=150, max_depth=parameter,
            min_samples_leaf=10, random_state=42, n_jobs=2)
        return Pipeline([('prepare', clone(preprocessor)), ('model', estimator)])
    return Pipeline([('prepare', clone(preprocessor)), ('scale', StandardScaler()),
                     ('model', LogisticRegression(C=parameter, max_iter=1500))])


def fit(model, data, features, weighted):
    target = pd.Categorical(data.severity, categories=LABELS).codes
    weights = data.severity.map({'Slight': 1, 'Serious': 2, 'Fatal': 5}).to_numpy()
    kwargs = {'model__sample_weight': weights} if weighted else {}
    model.fit(data[features], target, **kwargs)


def main():
    data = pd.read_csv(ROOT/'processed_crashes.csv.gz', dtype={'collision_index': str},
                       float_precision='round_trip')
    reference = joblib.load(ROOT/'vehicle_composition_optional.joblib')
    features = reference['features']
    prep = reference['pipeline'].named_steps['prepare']
    train = data[data.year.isin([2022, 2023])]
    validation = data[data.year == 2024]
    development, test = data[data.year < 2025], data[data.year == 2025]
    assert len(train) == 40862 and len(validation) == 19176 and len(test) == 19062
    families = {'Logistic Regression': [0.1, 1], 'Random Forest': [8, 14], 'XGBoost': [3, 5]}
    results = {'protocol': 'Train 2022-2023, choose settings and score multipliers on 2024 macro F1, refit 2022-2024, evaluate 2025.',
               'features': features, 'class_order': LABELS.tolist(), 'models': {}}
    for family, parameters in families.items():
        trials = []
        for parameter in parameters:
            for weighted in (False, True):
                model = make_model(family, parameter, prep)
                fit(model, train, features, weighted)
                score, weights = best_weights(model.predict_proba(validation[features]), validation.severity)
                trial = {'parameter': parameter, 'weighted': weighted,
                         'validation_macro_f1': score,
                         'decision_weights': dict(zip(LABELS.tolist(), weights.tolist()))}
                trials.append(trial)
                print(family, trial, flush=True)
        selected = max(trials, key=lambda item: item['validation_macro_f1'])
        weights = np.array([selected['decision_weights'][label] for label in LABELS])
        model = make_model(family, selected['parameter'], prep)
        fit(model, development, features, selected['weighted'])
        pred = LABELS[(model.predict_proba(test[features]) * weights).argmax(axis=1)]
        name = family.lower().replace(' ', '_')
        bundle = {'pipeline': model, 'features': features, 'class_order': LABELS.tolist(),
                  'decision_weights': selected['decision_weights']}
        joblib.dump(bundle, ROOT/f'comparison_{name}.joblib', compress=3)
        saved = joblib.load(ROOT/f'comparison_{name}.joblib')
        np.testing.assert_array_equal(pred, LABELS[(saved['pipeline'].predict_proba(test[features])*weights).argmax(axis=1)])
        results['models'][family] = {'selected': selected, 'validation_trials': trials,
                                     'test': evaluate(test.severity, pred)}
        print(family, 'test', results['models'][family]['test'], flush=True)
    (ROOT/'three_model_comparison.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
