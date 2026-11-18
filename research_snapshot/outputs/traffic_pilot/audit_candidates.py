"""Audit candidate WebTRIS matches; geographic proximity is not validation."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
candidate = pd.read_csv(ROOT / 'work/traffic_pilot/near_site2.csv', dtype={'collision_index': str})
crashes = pd.read_csv(ROOT / 'work/yearly-data/dft-road-casualty-statistics-collision-2024.csv',
                      usecols=['collision_index','collision_severity','junction_detail','road_type'],
                      dtype={'collision_index': str})
processed = pd.read_csv(ROOT / 'outputs/additional_years/data_processing/processed_crashes.csv.gz',
                        usecols=['collision_index','year'], dtype={'collision_index': str})
vehicles = pd.read_csv(ROOT / 'work/yearly-data/dft-road-casualty-statistics-vehicle-2024.csv',
                       usecols=['collision_index','vehicle_direction_from','vehicle_direction_to'],
                       dtype={'collision_index': str})
merged = candidate.merge(crashes, on='collision_index', validate='one_to_one')
merged['has_aadf'] = merged.collision_index.isin(processed.loc[processed.year.eq(2024), 'collision_index'])
v = vehicles[vehicles.collision_index.isin(merged.collision_index)]
sites = json.loads((ROOT/'work/traffic_pilot/sites.json').read_text())['sites']
site_description = {str(s['Id']):s.get('Description','') for s in sites}
merged['site_description'] = merged.site_id.astype(str).map(site_description)
cardinal={'Northbound':{8,1,2},'Eastbound':{2,3,4},'Southbound':{4,5,6},'Westbound':{6,7,8}}
def direction_status(group):
    description=group.site_description.iloc[0]
    direction=next((k for k in cardinal if k.lower() in description.lower()),None)
    if direction is None:
        return 'site direction not cardinal'
    codes=set(group.vehicle_direction_to)
    if not codes or any(code not in range(1,9) for code in codes):
        return 'vehicle direction unknown'
    return 'compatible' if codes.issubset(cardinal[direction]) else 'not all vehicles compatible'
direction_rows=v.merge(merged[['collision_index','site_description']],on='collision_index',validate='many_to_one')
directions=direction_rows.groupby('collision_index').apply(direction_status,include_groups=False).rename('direction_status')
directions.to_csv(ROOT/'outputs/traffic_pilot/direction_checks.csv')
direction_summary=directions.value_counts().to_dict()
overlap_ids=set(merged.loc[merged.has_aadf,'collision_index'])
overlap_direction=direction_rows[direction_rows.collision_index.isin(overlap_ids)].groupby('collision_index').apply(direction_status,include_groups=False).value_counts().to_dict()
summary = {'all_2024_crashes':len(crashes), 'candidates':len(merged),
           'candidate_site_count':int(merged.site_id.nunique()),
           'direction_check_all':direction_summary,
           'direction_check_aadf_overlap':overlap_direction,
           'candidate_severity_counts':merged.collision_severity.value_counts().sort_index().to_dict(),
           'aadf_overlap':int(merged.has_aadf.sum()),
           'aadf_overlap_severity':merged.loc[merged.has_aadf,'collision_severity'].value_counts().sort_index().to_dict(),
           'candidate_distance_m_median':float(merged.distance_m.median()),
           'candidate_direction_to_counts':v.vehicle_direction_to.value_counts(dropna=False).sort_index().to_dict(),
           'junction_detail_counts':merged.junction_detail.value_counts(dropna=False).sort_index().to_dict()}
out=ROOT/'outputs/traffic_pilot'
(out/'candidate_audit.json').write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
