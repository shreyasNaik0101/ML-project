"""Batch prediction on prepared feature CSVs, using trusted local model bundles."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def predict(bundle, rows):
    features = bundle['features']
    missing = set(features) - set(rows.columns)
    if missing:
        raise ValueError(f'Missing feature columns: {sorted(missing)}')
    values = rows[features].apply(pd.to_numeric, errors='raise').astype(float)
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError('Feature values cannot be infinite')
    flags = [c for c in features if c.startswith('vehicle_type_') or c == 'vehicle_record_missing']
    if flags and not values[flags].isin([0, 1]).all().all():
        raise ValueError('Vehicle flags must be 0 or 1')
    if 'aadf_motor_vehicles_per_day' in features:
        aadf = values['aadf_motor_vehicles_per_day']
        if aadf.isna().any() or aadf.lt(0).any():
            raise ValueError('AADF is required and must be nonnegative')
    model = bundle['pipeline']
    raw = model.predict_proba(values)
    weights = np.array([bundle['decision_weights'][c] for c in model.classes_])
    result = rows[['collision_index']].copy() if 'collision_index' in rows else pd.DataFrame(index=rows.index)
    result['predicted_severity'] = model.classes_[(raw * weights).argmax(axis=1)]
    for i, label in enumerate(model.classes_):
        result[f'raw_score_{label.lower()}'] = raw[:, i]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'artifacts/logistic_regression_without_aadf.joblib')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    args = parser.parse_args()
    try:
        result = predict(joblib.load(args.model), pd.read_csv(args.input, dtype={'collision_index': str}))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f'Saved {len(result)} predictions to {args.output}')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Prediction failed: {error}\n')
