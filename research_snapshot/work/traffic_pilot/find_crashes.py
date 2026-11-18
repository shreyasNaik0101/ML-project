import json
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[2]
data=pd.read_csv(root/'work/yearly-data/dft-road-casualty-statistics-collision-2024.csv',
                 usecols=['collision_index','date','time','first_road_class','first_road_number',
                          'location_easting_osgr','location_northing_osgr'],dtype={'collision_index':str})
import re
from scipy.spatial import cKDTree
sites=json.loads((root/'work/traffic_pilot/sites.json').read_text())['sites']
rows=[]
for site in sites:
    road=re.match(r'^(M(\d+)|A(\d+)M)/',site.get('Name',''),re.I)
    coords=re.search(r'GPS Ref:\s*(\d+);\s*(\d+)',site.get('Description') or '',re.I)
    if road and coords and site.get('Status')=='Active':
        rows.append({'site_id':site['Id'],'name':site['Name'],
                     'road_class':1 if road.group(2) else 2,
                     'road_number':int(road.group(2) or road.group(3)),
                     'easting':int(coords.group(1)),'northing':int(coords.group(2))})
stations=pd.DataFrame(rows)
results=[]
for (road_class,road_number),group in data.dropna(subset=['location_easting_osgr','location_northing_osgr']).groupby(['first_road_class','first_road_number']):
    candidates=stations[(stations.road_class==road_class)&(stations.road_number==road_number)]
    if candidates.empty:
        continue
    distance, index=cKDTree(candidates[['easting','northing']].to_numpy()).query(group[['location_easting_osgr','location_northing_osgr']].to_numpy())
    group=group.copy()
    group['distance_m']=distance
    group['site_id']=candidates.iloc[index].site_id.to_numpy()
    group['site_name']=candidates.iloc[index]['name'].to_numpy()
    results.append(group[group.distance_m<=250])
data=pd.concat(results).sort_values('distance_m')
data.to_csv(root/'work/traffic_pilot/near_site2.csv',index=False)
print('2024 motorway crashes within 250m of same-number active sensors:',len(data))
print(data.head(5).to_string(index=False))
