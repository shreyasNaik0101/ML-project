"""Evaluate disjoint road links separately from the historical crash-level split."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from compare_aadf import match_crashes, POINT_ID
from train_models import run
from train_random_forest import ID

ROOT = Path(__file__).resolve().parent


def partition_links(data):
    if data[POINT_ID].isna().any():
        raise ValueError('Every crash needs a road-link ID')
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_index, test_index = next(splitter.split(data, groups=data[POINT_ID]))
    development = data.iloc[dev_index]
    train_index, validation_index = next(splitter.split(development, groups=development[POINT_ID]))
    result = data.copy()
    result['split'] = 'test'
    result.loc[development.iloc[train_index].index, 'split'] = 'train'
    result.loc[development.iloc[validation_index].index, 'split'] = 'validation'
    groups = [set(rows[POINT_ID]) for _, rows in result.groupby('split')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Road links overlap between partitions')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'validation/unseen_links')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches = match_crashes(ROOT / 'data', ROOT / 'data/mrdb-2025.zip')
    data = pd.read_csv(ROOT / 'aadf_experiment/matched_crashes.csv.gz', dtype={ID: str}, float_precision='round_trip')
    joined = data.merge(matches[[ID, POINT_ID]], on=ID, validate='one_to_one')
    if len(joined) != len(data):
        raise ValueError('Expected every matched crash to have a link ID')
    grouped = partition_links(joined)
    assignments = grouped[[ID, POINT_ID, 'split', 'severity']]
    assignments.to_csv(args.output_dir / 'link_assignments.csv.gz', index=False)
    audit = {'seed': 42, 'group_column': POINT_ID, 'partition_overlap_links': 0,
             'split_crashes': grouped.groupby('split').size().to_dict(),
             'split_links': grouped.groupby('split')[POINT_ID].nunique().to_dict(),
             'split_class_counts': pd.crosstab(grouped['split'], grouped['severity']).to_dict(orient='index'),
             'protocol': 'GroupShuffleSplit 20% test links, then 20% of development links for validation; no seed search. All model families select settings using only grouped training and validation. These are new partitions of an already inspected dataset, not new external data.'}
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    path = args.output_dir / 'prepared_crashes.csv.gz'
    grouped.drop(columns=POINT_ID).to_csv(path, index=False)
    with threadpool_limits(limits=2):
        run(path, args.output_dir / 'models', tune_forest=True)
    print(json.dumps(audit, indent=2))
