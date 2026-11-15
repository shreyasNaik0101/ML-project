"""Reproduce all three matched-data classifiers and export inference bundles."""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from modeling import AADF, BASE_NUMERIC, CATEGORICAL, LABELS, make_pipeline, choose_score_weights
from train_random_forest import ID, TARGET, MODERATE_WEIGHTS, build_forest, evaluate

ROOT = Path(__file__).resolve().parent


def validate_data(data):
    required = {ID, TARGET, 'split', AADF, *BASE_NUMERIC, *CATEGORICAL}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing required data columns: {sorted(missing)}')
    unexpected = [c for c in data if c not in required and
                  not re.fullmatch(r'vehicle_type_(?:[0-9]+|missing)|vehicle_record_missing', c)]
    if unexpected:
        raise ValueError(f'Unexpected predictors could leak labels: {unexpected}')
    if data[ID].isna().any() or not data[ID].is_unique:
        raise ValueError('Crash IDs must be present and unique')
    if set(data['split']) != {'train', 'validation', 'test'}:
        raise ValueError('Expected fixed train, validation and test partitions')
    for split, rows in data.groupby('split'):
        if set(rows[TARGET]) != set(LABELS):
            raise ValueError(f'{split} must contain all three severity classes')
    if not np.isfinite(data[AADF]).all() or data[AADF].lt(0).any():
        raise ValueError('AADF must be finite and nonnegative')


def run(data_path, destination, tune_forest=False):
    data = pd.read_csv(data_path, dtype={ID: str}, float_precision='round_trip')
    validate_data(data)
    destination.mkdir(parents=True, exist_ok=True)
    base = data.columns.drop([ID, 'split', TARGET, AADF]).tolist()
    train, val, test = [data.loc[data['split'].eq(s)] for s in ('train', 'validation', 'test')]
    development = data.loc[~data['split'].eq('test')]
    results = {'sklearn_version': sklearn.__version__, 'label_order': LABELS,
               'split_counts': data['split'].value_counts().to_dict(), 'models': {}}
    for family in ('logistic_regression', 'decision_tree', 'random_forest'):
        trials = []
        if family == 'random_forest':
            if tune_forest:
                from tune_random_forest import tune_decision_weights
                for depth in (8, 14):
                    for weight in (None, MODERATE_WEIGHTS, 'balanced'):
                        candidate = build_forest(base + [AADF], depth, weight, jobs=2,
                                                 numeric=BASE_NUMERIC + [AADF])
                        candidate.fit(train[base + [AADF]], train[TARGET])
                        score, weights = tune_decision_weights(
                            candidate.predict_proba(val[base + [AADF]]),
                            val[TARGET], candidate.classes_.tolist())
                        trials.append(dict(depth=depth, training_class_weight=weight,
                            decision_weights=dict(zip(candidate.classes_, weights.tolist())),
                            validation_macro_f1=score[0], validation_accuracy=score[1]))
                selected = max(trials, key=lambda r: (r['validation_macro_f1'], r['validation_accuracy']))
            else:
                selected = json.loads((ROOT / 'tuned_metrics.json').read_text())['selected']
        else:
            values = (0.1, 1., 10.) if family == 'logistic_regression' else (4, 8, 14)
            for complexity in values:
                for weight in (None, MODERATE_WEIGHTS, 'balanced'):
                    model = make_pipeline(base + [AADF], complexity, weight, family)
                    model.fit(train[base + [AADF]], train[TARGET])
                    score, weights = choose_score_weights(model.predict_proba(val[base + [AADF]]),
                                                          val[TARGET], model.classes_.tolist())
                    trials.append(dict(complexity=complexity, class_weight=weight,
                                       decision_weights=weights, validation_macro_f1=score[0],
                                       validation_accuracy=score[1]))
            selected = max(trials, key=lambda r: (r['validation_macro_f1'], r['validation_accuracy']))
        result = {'selected': selected, 'validation_trials': trials}
        for with_aadf in (False, True):
            key = 'with_aadf' if with_aadf else 'without_aadf'
            features = base + ([AADF] if with_aadf else [])
            if family == 'random_forest':
                model = build_forest(features, selected['depth'], selected['training_class_weight'],
                                     jobs=2, numeric=BASE_NUMERIC + ([AADF] if with_aadf else []))
            else:
                model = make_pipeline(features, selected['complexity'], selected['class_weight'], family)
            model.fit(development[features], development[TARGET])
            raw = model.predict_proba(test[features])
            weights = np.array([selected['decision_weights'][c] for c in model.classes_])
            predicted = model.classes_[(raw * weights).argmax(axis=1)]
            scores = evaluate(test[TARGET], predicted)
            scores['fatal_average_precision'] = float(average_precision_score(
                test[TARGET].eq('Fatal'), raw[:, model.classes_.tolist().index('Fatal')]))
            bundle = dict(pipeline=model, features=features, decision_weights=selected['decision_weights'],
                          sklearn_version=sklearn.__version__, model_family=family)
            path = destination / f'{family}_{key}.joblib'
            joblib.dump(bundle, path, compress=3)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(predicted, loaded['pipeline'].classes_[
                (loaded['pipeline'].predict_proba(test[features]) * weights).argmax(axis=1)])
            result[key] = scores
            print(f'{family} {key}: macro F1={scores["macro_f1"]:.4f}', flush=True)
        results['models'][family] = result
    results['majority_baseline'] = evaluate(test[TARGET], ['Slight'] * len(test))
    (destination / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    rows = [dict(model=name, features=variant, accuracy=r[variant]['accuracy'],
                 macro_f1=r[variant]['macro_f1'], fatal_recall=r[variant]['per_class']['Fatal']['recall'])
            for name, r in results['models'].items() for variant in ('without_aadf', 'with_aadf')]
    pd.DataFrame(rows).to_csv(destination / 'comparison.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'aadf_experiment/matched_crashes.csv.gz')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data, args.output_dir)
