"""Development-only grouped model search; final evaluation is a separate opt-in step."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from modeling import AADF, CATEGORICAL, LABELS

ROOT = Path(__file__).resolve().parent
ID, GROUP, TARGET = 'collision_index', 'count_point_id', 'severity'


class Features(BaseEstimator, TransformerMixin):
    """Deterministic transformations only; learned preprocessing remains in the pipeline."""
    def __init__(self, engineered=False):
        self.engineered = engineered

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        if not self.engineered:
            return result
        result[AADF] = np.log1p(result[AADF])
        hour = np.mod(np.arctan2(result.time_sin, result.time_cos), 2*np.pi)*24/(2*np.pi)
        result['time_band'] = np.floor(hour/6)
        result['speed_band'] = pd.cut(result.speed_limit, [0, 30, 50, 70, np.inf], labels=False)
        result['weather_surface'] = result.weather_conditions.astype(str) + ':' + result.road_surface_conditions.astype(str)
        result['road_speed'] = result.road_type.astype(str) + ':' + result.speed_band.astype(str)
        for name, codes in {'motorcycles': [2, 3, 4, 5, 23, 97], 'hgv': [20, 21, 98], 'cycles': [1]}.items():
            columns = [f'vehicle_type_{code}' for code in codes if f'vehicle_type_{code}' in result]
            result[name] = result[columns].max(axis=1) if columns else 0
        return result


def pipeline(columns, family, params, engineered):
    categorical = CATEGORICAL + (['time_band', 'speed_band', 'weather_surface', 'road_speed'] if engineered else [])
    numeric = [c for c in columns if c not in CATEGORICAL and (engineered or c not in ['month', 'weekend'])] + (['motorcycles', 'hgv', 'cycles'] if engineered else [])
    prep = ColumnTransformer([
        ('categories', Pipeline([('missing', SimpleImputer(strategy='most_frequent')),
                                 ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical),
        ('numbers', Pipeline([('missing', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler())]), numeric)])
    constructors = {'logistic_regression': lambda: LogisticRegression(max_iter=2000, **params),
                    'random_forest': lambda: RandomForestClassifier(random_state=42, n_jobs=2, **params),
                    'decision_tree': lambda: DecisionTreeClassifier(random_state=42, **params)}
    return Pipeline([('features', Features(engineered)), ('preprocess', prep), ('model', constructors[family]())])


def metrics(actual, predicted):
    report = classification_report(actual, predicted, labels=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    return dict(accuracy=float(accuracy_score(actual, predicted)),
                macro_f1=float(f1_score(actual, predicted, labels=LABELS, average='macro', zero_division=0)),
                per_class={c: report[c] for c in LABELS}, confusion_matrix=matrix.tolist(),
                fatal_as_slight=int(matrix[2, 0]))


def folds_for(data):
    if data[GROUP].isna().any() or data[ID].isna().any() or not data[ID].is_unique:
        raise ValueError('Unique crash IDs and nonmissing road links are required')
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(data, data[TARGET], data[GROUP]))
    for train, valid in folds:
        if set(data.iloc[train][GROUP]) & set(data.iloc[valid][GROUP]):
            raise ValueError('Road-link leakage')
        if any(set(data.iloc[index][TARGET]) != set(LABELS) for index in (train, valid)):
            raise ValueError('Every training and validation fold must contain all three classes')
    return folds


def resample_indices(y, method, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.asarray(y) == label) for label in LABELS]
    if method == 'none':
        return np.arange(len(y))
    size = max(map(len, groups)) if method == 'over' else min(map(len, groups))
    return rng.permutation(np.concatenate([rng.choice(g, size, replace=len(g) < size) for g in groups]))


def decision_rules(y, probabilities, minimum_recall, minimum_accuracy):
    """Report three OOF operating points; never consult holdout rows."""
    actual = pd.Categorical(y, categories=LABELS).codes
    if (actual < 0).any():
        raise ValueError('Unknown severity label')
    candidates = []
    for serious in (.5, .75, 1., 1.25, 1.5, 2., 3.):
        for fatal in (.5, .75, 1., 1.5, 2., 3., 4., 6., 8., 12., 16.):
            weights = np.array([1., serious, fatal])
            prediction = (probabilities*weights).argmax(axis=1)
            matrix = np.bincount(actual*3+prediction, minlength=9).reshape(3, 3)
            support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
            denominator = support + predicted
            f1 = np.divide(2*np.diag(matrix), denominator, out=np.zeros(3), where=denominator > 0)
            candidates.append(dict(weights=weights, accuracy=float(np.trace(matrix)/len(actual)),
                macro_f1=float(f1.mean()), fatal_recall=float(matrix[2, 2]/support[2]) if support[2] else 0.))
    feasible = [r for r in candidates if r['accuracy'] >= minimum_accuracy and r['fatal_recall'] >= minimum_recall]
    def describe(point):
        if point is None:
            return None
        prediction = np.asarray(LABELS)[(probabilities*point['weights']).argmax(axis=1)]
        return dict(weights=dict(zip(LABELS, point['weights'].tolist())), **metrics(y, prediction))
    return dict(accuracy=describe(max(candidates, key=lambda r: (r['accuracy'], r['macro_f1']))),
                macro_f1=describe(max(candidates, key=lambda r: (r['macro_f1'], r['accuracy']))),
                constrained=describe(max(feasible, key=lambda r: (r['macro_f1'], r['accuracy'])) if feasible else None))



def run(data, output, trials=8, minimum_recall=.2895, minimum_accuracy=0., sampling=('none',), families=None):
    output.mkdir(parents=True, exist_ok=True)
    for stale in ('selected.joblib', 'inspected_test.json'):
        (output/stale).unlink(missing_ok=True)
    if set(data['split']) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation and test partitions')
    dev = data.loc[data['split'] != 'test'].reset_index(drop=True)
    holdout_links = set(data.loc[data['split'] == 'test', GROUP])
    if set(dev[GROUP]) & holdout_links:
        raise ValueError('Development and holdout links overlap')
    columns = [c for c in data if c in CATEGORICAL + ['speed_limit', 'time_sin', 'time_cos', AADF, 'month', 'weekend'] or c.startswith('vehicle_type_') or c == 'vehicle_record_missing']
    folds = folds_for(dev)
    assignments = dev[[ID, GROUP, TARGET]].copy()
    assignments['fold'] = -1
    audit = []
    for fold, (train, valid) in enumerate(folds):
        assignments.loc[valid, 'fold'] = fold
        audit.append(dict(fold=fold, train_rows=len(train), validation_rows=len(valid),
                          validation_class_counts=dev.iloc[valid][TARGET].value_counts().to_dict(),
                          validation_links=int(dev.iloc[valid][GROUP].nunique())))
    assignments.to_csv(output/'development_folds.csv.gz', index=False)
    weights = [None, {'Slight': 1, 'Serious': 2, 'Fatal': 5}, 'balanced']
    spaces = {
        'logistic_regression': dict(C=[.01, .1, 1., 10., 100.], class_weight=weights),
        'decision_tree': dict(max_depth=[4, 8, 14, None], min_samples_leaf=[5, 10, 20, 40], class_weight=weights),
        'random_forest': dict(n_estimators=[150, 300, 500], max_depth=[8, 14, None], min_samples_leaf=[5, 10, 20, 40], max_features=['sqrt', .5, 1.], class_weight=weights)}
    results, selected = [], None
    for family, space in spaces.items():
        if families is not None and family not in families:
            continue
        for params in ParameterSampler(space, n_iter=trials, random_state=42):
            for engineered in (False, True):
                for sampling_method in sampling:
                    probabilities = np.zeros((len(dev), 3))
                    fold_scores = []
                    for fold, (train, valid) in enumerate(folds):
                        model = pipeline(columns, family, params, engineered)
                        local = resample_indices(dev.iloc[train][TARGET], sampling_method, 42+fold)
                        sampled = dev.iloc[train[local]]
                        model.fit(sampled[columns], sampled[TARGET])
                        raw = model.predict_proba(dev.iloc[valid][columns])
                        probabilities[valid] = raw[:, [list(model.classes_).index(c) for c in LABELS]]
                        fold_scores.append(metrics(dev.iloc[valid][TARGET], model.predict(dev.iloc[valid][columns])))
                    rules = decision_rules(dev[TARGET], probabilities, minimum_recall, minimum_accuracy)
                    record = dict(family=family, params=params, engineered=engineered, sampling=sampling_method, rules=rules, unadjusted_folds=fold_scores)
                    results.append(record)
                    point = rules['constrained']
                    if point is not None and (selected is None or (point['macro_f1'], point['accuracy']) > (selected['rules']['constrained']['macro_f1'], selected['rules']['constrained']['accuracy'])):
                        selected = record
                    print(f'{family} engineered={engineered} sampling={sampling_method}: OOF macro F1={rules["macro_f1"]["macro_f1"]:.4f}', flush=True)
    report = dict(sklearn_version=sklearn.__version__, seed=42, protocol='Development OOF selection; OOF scores are selection estimates, not independent performance. Existing test previously inspected; no test metrics used for selection.',
                  label_order=LABELS, features=columns, development_rows=len(dev), folds=audit,
                  minimum_fatal_recall=minimum_recall, minimum_accuracy=minimum_accuracy,
                  majority_baseline=metrics(dev[TARGET], ['Slight']*len(dev)), trials=results, selected=selected,
                  data_sha256=hashlib.sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes()).hexdigest())
    (output/'search.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'search.json.gz').write_bytes(gzip.compress((output/'search.json').read_bytes(), mtime=0))
    table = []
    for candidate_id, candidate in enumerate(results):
        for rule, point in candidate['rules'].items():
            if point is not None:
                table.append(dict(candidate=candidate_id, family=candidate['family'], engineered=candidate['engineered'], sampling=candidate['sampling'], rule=rule,
                    accuracy=point['accuracy'], macro_f1=point['macro_f1'], fatal_precision=point['per_class']['Fatal']['precision'], fatal_recall=point['per_class']['Fatal']['recall'], fatal_as_slight=point['fatal_as_slight']))
    pd.DataFrame(table).to_csv(output/'tradeoffs.csv', index=False)

    if selected is None:
        raise ValueError('No candidate meets the requirements. Trade-offs saved; no model exported.')
    model = pipeline(columns, selected['family'], selected['params'], selected['engineered'])
    indices = resample_indices(dev[TARGET], selected['sampling'], 42)
    model.fit(dev.iloc[indices][columns], dev.iloc[indices][TARGET])
    bundle = dict(pipeline=model, features=columns, decision_weights=selected['rules']['constrained']['weights'], model_family=selected['family'])
    joblib.dump(bundle, output/'selected.joblib', compress=3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'validation/grouped_search')
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--families', nargs='+', choices=['logistic_regression', 'decision_tree', 'random_forest'])
    parser.add_argument('--minimum-fatal-recall', type=float, default=.2895)
    parser.add_argument('--minimum-accuracy', type=float, default=0.)
    parser.add_argument('--sampling', nargs='+', choices=['none', 'over', 'under'], default=['none'])
    parser.add_argument('--evaluate-inspected-test', action='store_true')
    parser.add_argument('--include-calendar', action='store_true', help='Join source dates; engineered models then require month and weekend at inference')
    args = parser.parse_args()
    if args.trials < 1 or not all(0 <= x <= 1 for x in (args.minimum_accuracy, args.minimum_fatal_recall)):
        parser.error('Trials must be positive and metric requirements must be between 0 and 1')
    from compare_aadf import match_crashes
    from evaluate_unseen_links import partition_links
    matches = match_crashes(ROOT/'data', ROOT/'data/mrdb-2025.zip')
    data = pd.read_csv(ROOT/'aadf_experiment/matched_crashes.csv.gz', dtype={ID: str})
    joined = data.merge(matches[[ID, GROUP]], on=ID, validate='one_to_one')
    if len(joined) != len(data):
        raise ValueError('Every prepared crash must have a road-link match')
    data = partition_links(joined)
    if args.include_calendar:
        dates = pd.read_csv(ROOT/'data/dft-road-casualty-statistics-collision-2025.csv', usecols=[ID, 'date'], dtype={ID: str})
        parsed = pd.to_datetime(dates['date'], format='%d/%m/%Y', errors='raise')
        dates['month'] = parsed.dt.month
        dates['weekend'] = (parsed.dt.dayofweek >= 5).astype(int)
        data = data.merge(dates.drop(columns='date'), on=ID, validate='one_to_one')
        if len(data) != len(joined):
            raise ValueError('Missing dates for prepared crashes')
    with threadpool_limits(limits=2):
        run(data, args.output_dir, args.trials, args.minimum_fatal_recall, args.minimum_accuracy, args.sampling, args.families)
    if args.evaluate_inspected_test:
        from predict import predict
        test = data.loc[data['split'] == 'test']
        result = metrics(test[TARGET], predict(joblib.load(args.output_dir/'selected.joblib'), test).predicted_severity)
        (args.output_dir/'inspected_test.json').write_text(json.dumps(dict(status='Previously inspected 2025 holdout; not fresh external validation', selected=result, majority=metrics(test[TARGET], ['Slight']*len(test))), indent=2)+'\n')


if __name__ == '__main__':
    from grouped_experiments import main as entrypoint
    entrypoint()
