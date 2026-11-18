"""Validation-only model/decision search; inspected 2025 evaluation is opt-in."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

LABELS = np.array(['Fatal', 'Serious', 'Slight'])
CATEGORICAL = ['road_type', 'weather_conditions', 'road_surface_conditions']
NUMERIC = ['speed_limit', 'time_sin', 'time_cos', 'aadf_motor_vehicles_per_day',
           'time_sin_2', 'time_cos_2', 'time_sin_3', 'time_cos_3',
           'vehicle_composition_cycle', 'vehicle_composition_motorcycle',
           'vehicle_composition_car_taxi', 'vehicle_composition_bus',
           'vehicle_composition_goods', 'vehicle_composition_other']
METADATA = {'collision_index', 'year', 'count_point_id', 'severity'}
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def split_data(data):
    if not data.collision_index.is_unique or data.year.isna().any():
        raise ValueError('Unique collision IDs and complete years are required')
    if set(data.severity) != set(LABELS):
        raise ValueError('Expected three severity labels')
    parts = [data[data.year.isin(years)].copy() for years in ([2021, 2022, 2023], [2024], [2025])]
    if any(part.empty for part in parts):
        raise ValueError('All temporal partitions must be present')
    return parts


def metrics(y, prediction):
    actual = pd.Categorical(y, categories=LABELS).codes
    if (actual < 0).any():
        raise ValueError('Unexpected severity')
    matrix = np.bincount(actual*3 + prediction, minlength=9).reshape(3, 3)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    correct = matrix.diagonal()
    return {'accuracy': float(correct.sum()/matrix.sum()),
            'macro_f1': float(np.mean(2*correct/np.maximum(support+predicted, 1))),
            'recall': dict(zip(LABELS, (correct/np.maximum(support, 1)).tolist())),
            'precision': dict(zip(LABELS, (correct/np.maximum(predicted, 1)).tolist())),
            'confusion_matrix': matrix.tolist(), 'label_order': LABELS.tolist()}


def operating_points(y, probabilities):
    points = []
    for serious in [.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4.]:
        for fatal in [.5, 1., 1.5, 2., 3., 4., 6., 8., 12., 16.]:
            weights = [fatal, serious, 1.]
            score = metrics(y, (probabilities*np.array(weights)).argmax(axis=1))
            points.append(dict(score, weights=weights))
    # Fix these requirements before scoring the previously inspected 2025 cohort.
    eligible = [p for p in points if p['recall']['Fatal'] >= .22 and p['recall']['Serious'] >= .33]
    return {'accuracy': max(points, key=lambda p: (p['accuracy'], p['macro_f1'])),
            'balanced': max(points, key=lambda p: (p['macro_f1'], p['accuracy'])),
            'constrained': max(eligible, key=lambda p: (p['accuracy'], p['macro_f1'])) if eligible else None}


def candidates():
    yield {'name': 'baseline_lr', 'family': 'lr', 'params': {}, 'weighted': False}
    yield {'name': 'baseline_rf', 'family': 'rf', 'params': {}, 'weighted': True}
    for depth, leaf in [(16, 5), (22, 5), (None, 10)]:
        yield {'name': f'rf_{depth}_{leaf}', 'family': 'rf', 'params': {'max_depth': depth, 'min_samples_leaf': leaf, 'n_estimators': 250}, 'weighted': True}
    for depth, weight in [(4, False), (6, False), (6, True), (8, True)]:
        yield {'name': f'xgb_{depth}_{weight}', 'family': 'xgb', 'params': {'max_depth': depth}, 'weighted': weight}
    for depth, weight in [(5, False), (7, False), (7, True), (9, True)]:
        yield {'name': f'cat_{depth}_{weight}', 'family': 'cat', 'params': {'depth': depth}, 'weighted': weight}


def make_model(config, features):
    family, params = config['family'], config['params']
    numbers = [c for c in NUMERIC if c in features]
    flags = sorted(c for c in features if c.startswith('vehicle_type_'))
    if family == 'cat':
        from experiments.temporal_enhancements.estimators import CatBoostEstimator
        # Numeric published category codes remain categorical via one-hot encoding.
        estimator = CatBoostEstimator(**params)
    elif family == 'xgb':
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_estimators=500, learning_rate=.04, min_child_weight=15,
                                  subsample=.85, colsample_bytree=.85, reg_lambda=5.,
                                  objective='multi:softprob', num_class=3, tree_method='hist',
                                  random_state=42, n_jobs=2, **params)
    elif family == 'rf':
        estimator = RandomForestClassifier(random_state=42, n_jobs=2,
                    **dict({'n_estimators': 150, 'max_depth': 14, 'min_samples_leaf': 10}, **params))
    else:
        estimator = LogisticRegression(C=1, max_iter=1500)
    numeric = [('missing', SimpleImputer(strategy='median', add_indicator=True))]
    prep = ColumnTransformer([
        ('categories', Pipeline([('missing', SimpleImputer(strategy='constant', fill_value=-1)),
                                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL),
        ('numeric', Pipeline(numeric), numbers), ('vehicles', 'passthrough', flags)])
    steps = [('prepare', prep)]
    if family == 'lr':
        steps.append(('scale', StandardScaler()))
    return Pipeline(steps + [('model', estimator)])


def fit(model, config, frame, features):
    y = pd.Categorical(frame.severity, categories=LABELS).codes
    options = {'model__sample_weight': frame.severity.map({'Fatal': 5., 'Serious': 2., 'Slight': 1.}).to_numpy()} if config['weighted'] else {}
    with threadpool_limits(limits=2):
        model.fit(frame[features], y, **options)
    return model


def run(data_path, output, evaluate=False, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path, dtype={'collision_index': str}, float_precision='round_trip')
    train, validation, test = split_data(data)
    features = CATEGORICAL + NUMERIC + sorted(c for c in data if c.startswith('vehicle_type_'))
    if set(features) != set(data) - METADATA:
        raise ValueError('Unexpected or missing predictor columns')
    (output/'features.json').write_text(json.dumps(features, indent=2))
    trials, configurations = [], list(candidates())
    if limit:
        configurations = configurations[:limit]
    for config in configurations:
        start = time.monotonic()
        model = fit(make_model(config, features), config, train, features)
        with threadpool_limits(limits=2):
            probabilities = model.predict_proba(validation[features])
        result = {'config': config, 'points': operating_points(validation.severity, probabilities),
                  'seconds': round(time.monotonic()-start, 2)}
        trials.append(result)
        print(config['name'], json.dumps(result['points']['constrained']), flush=True)
        (output/'validation_trials.json').write_text(json.dumps(trials, indent=2))
    selected = {}
    for point in ['accuracy', 'balanced', 'constrained']:
        available = [t for t in trials if t['points'][point] is not None]
        if available:
            key = 'macro_f1' if point == 'balanced' else 'accuracy'
            winner = max(available, key=lambda t: (t['points'][point][key], t['points'][point]['macro_f1']))
            selected[point] = {'config': winner['config'], 'validation': winner['points'][point]}
    environment = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','catboost','joblib','threadpoolctl']}
    summary = {'protocol': '2021-23 fit, 2024 selection, refit 2021-24; 2025 previously inspected, evaluation opt-in',
               'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
               'rows': {'train': len(train), 'validation': len(validation), 'inspected_test': len(test)},
               'environment': dict(environment, python=platform.python_version()), 'selected': selected}
    (output/'selection.json').write_text(json.dumps(summary, indent=2))
    if evaluate:
        requests = dict(selected)
        # Reproduce historical baseline decision selection in the same runtime.
        historical_grid = [(f, s, 1.) for s in [.75,1.,1.25,1.5,2.,3.] for f in [1.,2.,4.,6.,8.,12.]]
        for config in configurations[:2]:
            model = fit(make_model(config, features), config, train, features)
            probabilities = model.predict_proba(validation[features])
            weights = max(historical_grid, key=lambda w: metrics(validation.severity, (probabilities*np.array(w)).argmax(axis=1))['macro_f1'])
            requests[config['name']] = {'config': config, 'validation': {'weights': list(weights)}}
        evaluations = {}
        development = pd.concat([train, validation], ignore_index=True)
        for name, chosen in requests.items():
            model = fit(make_model(chosen['config'], features), chosen['config'], development, features)
            weights = np.array(chosen['validation']['weights'])
            with threadpool_limits(limits=2):
                raw = model.predict_proba(test[features])
            evaluations[name] = dict(metrics(test.severity, (raw*weights).argmax(axis=1)), config=chosen['config'], weights=weights.tolist())
            bundle = {'pipeline': model, 'features': features, 'labels': LABELS.tolist(), 'weights': weights.tolist()}
            path = output/f'{name}.joblib'
            joblib.dump(bundle, path)
            reloaded = joblib.load(path)
            np.testing.assert_allclose(reloaded['pipeline'].predict_proba(test[features].head(20)), raw[:20])
            print('EVALUATED', name, json.dumps(evaluations[name]), flush=True)
        (output/'inspected_test.json').write_text(json.dumps(evaluations, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/final_matched_2021_2025.csv.gz')
    parser.add_argument('--output', type=Path, default=ROOT/'results/temporal_enhancements')
    parser.add_argument('--evaluate-inspected-test', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.data, args.output, args.evaluate_inspected_test, args.limit)
