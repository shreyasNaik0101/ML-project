"""Compare single feature-group removals on the shared temporal split."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from train_xgboost import build, LABELS

ROOT = Path(__file__).resolve().parent


def evaluate(y, probabilities, weights):
    prediction = LABELS[(probabilities * weights).argmax(axis=1)]
    return {'accuracy': float(accuracy_score(y, prediction)),
            'report': classification_report(y, prediction, labels=LABELS.tolist(),
                                             output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, prediction, labels=LABELS.tolist()).tolist()}


def make_preprocessor(original, features):
    transforms = []
    for name, transformer, columns in original.transformers:
        remaining = [column for column in columns if column in features]
        if remaining:
            transforms.append((name, clone(transformer) if transformer != 'passthrough' else transformer, remaining))
    return ColumnTransformer(transforms)


def main():
    data = pd.read_csv(ROOT/'processed_crashes.csv.gz', dtype={'collision_index':str}, float_precision='round_trip')
    bundle = joblib.load(ROOT/'xgboost.joblib')
    reference = json.loads((ROOT/'xgboost_results.json').read_text())
    features = bundle['features']
    weights = np.array([bundle['decision_weights'][c] for c in LABELS])
    groups = {
        'all_features': [],
        'without_aadf': ['aadf_motor_vehicles_per_day'],
        'without_weather': ['weather_conditions'],
        'without_surface': ['road_surface_conditions'],
        'without_road_type': ['road_type'],
        'without_speed_limit': ['speed_limit'],
        'without_extra_time_harmonics': ['time_sin_2','time_cos_2','time_sin_3','time_cos_3'],
        'without_vehicle_counts': [c for c in features if c.startswith('vehicle_composition_')],
    }
    train = data[data.year.isin([2022,2023])]
    validation = data[data.year == 2024]
    codes = pd.Categorical(train.severity, categories=LABELS).codes
    original_preprocessor = bundle['pipeline'].named_steps['prepare']
    params = reference['selected']
    assert params['sample_weighting'] == 'none'
    trials = {}
    for name, removed in groups.items():
        columns = [c for c in features if c not in removed]
        prep = make_preprocessor(original_preprocessor, columns)
        model = build(prep, params['max_depth'], params['learning_rate'])
        model.fit(train[columns], codes)
        trials[name] = {'removed': removed, 'features': columns,
                        'validation': evaluate(validation.severity, model.predict_proba(validation[columns]), weights)}
        print(name, trials[name]['validation']['report']['macro avg']['f1-score'], flush=True)
    winner = max(trials, key=lambda n: trials[n]['validation']['report']['macro avg']['f1-score'])
    selected = trials[winner]
    columns = selected['features']
    development, test = data[data.year < 2025], data[data.year == 2025]
    model = build(make_preprocessor(original_preprocessor, columns), params['max_depth'], params['learning_rate'])
    model.fit(development[columns], pd.Categorical(development.severity, categories=LABELS).codes)
    probabilities = model.predict_proba(test[columns])
    selected['test'] = evaluate(test.severity, probabilities, weights)
    fitted = {'pipeline':model,'features':columns,'class_order':LABELS.tolist(),
              'decision_weights':bundle['decision_weights'], 'removed':selected['removed']}
    joblib.dump(fitted, ROOT/'ablation_selected.joblib', compress=3)
    restored = joblib.load(ROOT/'ablation_selected.joblib')
    np.testing.assert_array_equal((probabilities*weights).argmax(1),
        (restored['pipeline'].predict_proba(test[columns])*weights).argmax(1))
    output = {'selection':'highest 2024 validation macro F1; fixed model parameters and decision weights',
              'selection_warning':'2024 and 2025 have been previously inspected repeatedly',
              'winner':winner,'trials':trials,'previous_all_features_test':reference['xgboost_test']}
    (ROOT/'ablation_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Selected:',winner,flush=True)
    print(json.dumps({'selected_test':selected['test'], 'baseline_test':reference['xgboost_test']},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
