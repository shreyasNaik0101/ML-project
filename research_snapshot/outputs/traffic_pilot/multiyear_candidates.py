"""Link sensors using road number, historic count-point geometry and coarse direction."""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import json,re
import numpy as np
import pandas as pd
import shapefile
from scipy.spatial import cKDTree
from shapely.geometry import Point,shape

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/traffic_pilot'
processed=pd.read_csv(ROOT/'outputs/additional_years/data_processing/processed_crashes.csv.gz',
                      usecols=['collision_index','year','count_point_id'],dtype={'collision_index':str})
catalog=json.loads((ROOT/'work/traffic_pilot/sites.json').read_text())['sites']
allowed={'northbound':{8,1,2},'eastbound':{2,3,4},'southbound':{4,5,6},'westbound':{6,7,8}}
sensors=[]
for site in catalog:
    road=re.match(r'^(M(\d+)|A(\d+)M)/',site.get('Name',''),re.I)
    desc=site.get('Description') or ''
    gps=re.search(r'GPS Ref:\s*(\d+);\s*(\d+)',desc,re.I)
    direction=next((d for d in allowed if d in desc.lower()),None)
    if road and gps and direction and site.get('Status')=='Active':
        sensors.append({'site_id':int(site['Id']),'site_name':site['Name'],'direction':direction,
                        'road_class':1 if road.group(2) else 2,
                        'road_number':int(road.group(2) or road.group(3)),
                        'easting':int(gps.group(1)),'northing':int(gps.group(2))})
sites=pd.DataFrame(sensors)
all_candidates=[]
audit={}
for year in (2022,2023,2024):
    selected=processed[processed.year.eq(year)].copy()
    c=pd.read_csv(ROOT/f'work/yearly-data/dft-road-casualty-statistics-collision-{year}.csv',
                  usecols=['collision_index','date','time','first_road_class','first_road_number',
                           'location_easting_osgr','location_northing_osgr'],dtype={'collision_index':str})
    c=selected.merge(c,on='collision_index',validate='one_to_one').dropna(subset=['location_easting_osgr','location_northing_osgr','time'])
    v=pd.read_csv(ROOT/f'work/yearly-data/dft-road-casualty-statistics-vehicle-{year}.csv',
                  usecols=['collision_index','vehicle_direction_to'],dtype={'collision_index':str})
    v=v[v.collision_index.isin(c.collision_index)]
    direction=v.groupby('collision_index').vehicle_direction_to.agg(lambda s: set(s))
    wanted=set(c.count_point_id.astype(int))
    geom=defaultdict(list)
    path=ROOT/f'work/yearly-data/mrdb-{year}.zip'
    with ZipFile(path) as z:
        shp=next(n for n in z.namelist() if n.lower().endswith('.shp'))
    with shapefile.Reader(f'{path}/{shp}') as reader:
        for rec in reader.iterShapeRecords():
            a=rec.record.as_dict()
            if a['CP_Number'] in wanted and rec.shape.points:
                geom[int(a['CP_Number'])].append(shape(rec.shape.__geo_interface__))
    counts=defaultdict(int)
    for (road_class,number),group in c.groupby(['first_road_class','first_road_number']):
        ss=sites[(sites.road_class==road_class)&(sites.road_number==number)].reset_index(drop=True)
        if ss.empty: continue
        tree=cKDTree(ss[['easting','northing']].to_numpy())
        for row in group.itertuples(index=False):
            indices=tree.query_ball_point([row.location_easting_osgr,row.location_northing_osgr],250)
            if not indices: continue
            counts['has_nearby_cardinal_sensor']+=1
            codes=direction.get(row.collision_index,set())
            if not codes or not codes.issubset(set(range(1,9))):
                counts['vehicle_direction_missing']+=1; continue
            choices=[]
            for idx in indices:
                s=ss.iloc[idx]
                if not codes.issubset(allowed[s.direction]): continue
                cp_geoms=geom.get(int(row.count_point_id),[])
                if not cp_geoms: continue
                point=Point(s.easting,s.northing)
                link_distance=min(g.distance(point) for g in cp_geoms)
                if link_distance>50: continue
                distance=np.hypot(s.easting-row.location_easting_osgr,s.northing-row.location_northing_osgr)
                choices.append((distance,int(s.site_id),s.site_name,s.direction,link_distance))
            if choices:
                d,site_id,name,direct,link_d=min(choices)
                all_candidates.append({'collision_index':row.collision_index,'year':year,'date':row.date,'time':row.time,
                    'site_id':site_id,'site_name':name,'site_direction':direct,'crash_sensor_m':d,
                    'sensor_link_m':link_d,'count_point_id':int(row.count_point_id)})
                counts['matched']+=1
            else: counts['no_direction_and_link_match']+=1
    audit[str(year)]={'aadf_rows':len(selected),**dict(counts)}
    print(year,audit[str(year)],flush=True)
pd.DataFrame(all_candidates).to_csv(OUT/'multiyear_candidates.csv',index=False)
(OUT/'multiyear_match_audit.json').write_text(json.dumps(audit,indent=2))
