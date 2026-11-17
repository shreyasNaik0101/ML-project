"""Audit matching coverage and export a stratified sample for manual map review."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from modeling import AADF, CATEGORICAL, LABELS

ROOT = Path(__file__).resolve().parent
ID = 'collision_index'


def audit(source, prepared, matches, output):
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source, dtype={ID: str})
    if not raw[ID].is_unique or not prepared[ID].is_unique:
        raise ValueError('Crash IDs must be unique')
    label = raw.collision_severity.map({1: 'Fatal', 2: 'Serious', 3: 'Slight'})
    retained = raw[ID].isin(prepared[ID])
    coverage = pd.crosstab(label, retained).reindex(LABELS, fill_value=0)
    coverage = coverage.reindex(columns=[False, True], fill_value=0)
    coverage.columns = ['unmatched', 'matched']
    coverage['retention_rate'] = coverage.matched / coverage.sum(axis=1)
    coverage.to_csv(output/'class_coverage.csv')
    fields = {}
    for column in CATEGORICAL + ['speed_limit', 'time', 'vehicle_type']:
        frame = raw if column in raw else prepared
        if column not in frame:
            continue
        values = frame[column]
        fields[column] = dict(missing=int(values.isna().sum()), unique_values=int(values.nunique(dropna=False)), most_common_values=values.value_counts(dropna=False).head(20).astype(int).to_dict())
    traffic = prepared[AADF]
    summary = dict(source_rows=len(raw), matched_rows=len(prepared), retention_rate=float(retained.mean()),
        class_coverage=coverage.to_dict(orient='index'), fields=fields,
        aadf=dict(missing=int(traffic.isna().sum()), negative=int(traffic.lt(0).sum()), nonfinite=int((~np.isfinite(traffic)).sum()), quantiles=traffic.quantile([0, .25, .5, .75, 1]).to_dict()),
        speed_limit_unexpected=int((~raw.speed_limit.isin([20, 30, 40, 50, 60, 70, -1])).sum()),
        caveat='Automated flags need DfT codebook review. Matching distance is not proof of correct linkage; manually review the map sample, especially junctions and parallel roads.')
    (output/'quality.json').write_text(json.dumps(summary, indent=2)+'\n')
    sample = matches.merge(raw[[ID, 'longitude', 'latitude', 'junction_detail', 'first_road_class', 'first_road_number']], on=ID, validate='one_to_one').merge(prepared[[ID, 'severity']], on=ID, validate='one_to_one')
    sample = pd.concat([group.sample(min(30, len(group)), random_state=42) for _, group in sample.groupby('severity')])
    sample['manual_match_correct'] = ''
    sample['review_notes'] = ''
    sample.to_csv(output/'matching_review.csv', index=False)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'validation/data_quality')
    args = parser.parse_args()
    from compare_aadf import match_crashes
    audit(ROOT/'data/dft-road-casualty-statistics-collision-2025.csv',
          pd.read_csv(ROOT/'aadf_experiment/matched_crashes.csv.gz', dtype={ID: str}),
          match_crashes(ROOT/'data', ROOT/'data/mrdb-2025.zip'), args.output_dir)
