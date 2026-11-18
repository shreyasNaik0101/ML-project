"""Exploratory, same-row traffic feature comparison; not a final evaluation."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/traffic_pilot'
FIELD='aadf_motor_vehicles_per_day'
LABELS=['Slight','Serious','Fatal']
candidate=pd.read_csv(ROOT/'work/traffic_pilot/near_site2.csv',dtype={'collision_index':str})
base=pd.read_csv(ROOT/'outputs/additional_years/data_processing/processed_crashes.csv.gz',dtype={'collision_index':str})
data=candidate.merge(base[base.year.eq(2024)],on='collision_index',validate='one_to_one')
data['crash_datetime']=pd.to_datetime(data['date']+' '+data['time'],format='%d/%m/%Y %H:%M',errors='coerce')
records=[]
status={}
for row in data.itertuples(index=False):
    dt=row.crash_datetime
    if pd.isna(dt):
        status['invalid_crash_time']=status.get('invalid_crash_time',0)+1; continue
    p=ROOT/'work/traffic_pilot/readings'/f'{row.site_id}_{dt:%d%m%Y}.json'
    if not p.exists():
        status['download_missing']=status.get('download_missing',0)+1; continue
    payload=json.loads(p.read_text())
    if payload.get('Header',{}).get('row_count',0)>len(payload.get('Rows',[])):
        status['incomplete_pagination']=status.get('incomplete_pagination',0)+1; continue
    usable=[]
    for r in payload.get('Rows',[]):
        try:
            end=pd.Timestamp(r['Report Date'][:10]+' '+r['Time Period Ending'])
            count=float(r['Total Volume']); speed=float(r['Avg mph'])
            # The reported ending minute is inclusive, so treat the period as
            # complete one minute later; add a full 15-minute separation.
            if end+pd.Timedelta(minutes=1)<=dt-pd.Timedelta(minutes=15) and count>=0 and speed>0:
                usable.append((end,count,speed))
        except (KeyError,ValueError,TypeError):
            continue
    if not usable:
        status['no_eligible_reading']=status.get('no_eligible_reading',0)+1; continue
    end,count,speed=max(usable,key=lambda x:x[0])
    # Avoid stale measurements: at most 45 minutes before the collision.
    if dt-(end+pd.Timedelta(minutes=1))>pd.Timedelta(minutes=45):
        status['stale_reading']=status.get('stale_reading',0)+1; continue
    records.append({'collision_index':row.collision_index,'sensor_end':str(end),
                    'volume_15m':count,'average_mph':speed,'density_proxy':4*count/speed})
    status['usable']=status.get('usable',0)+1
data=data.merge(pd.DataFrame(records),on='collision_index',validate='one_to_one')
directions=pd.read_csv(OUT/'direction_checks.csv',dtype={'collision_index':str})
data=data.merge(directions,on='collision_index',validate='one_to_one')
bundle=joblib.load(ROOT/'outputs/additional_years/gradient_boosting_2022_2023.joblib')
features=bundle['features']
results={'candidates':len(candidate),'aadf_overlap_candidates':sum(data.severity.notna())+sum(v for k,v in status.items() if k!='usable'),
         'traffic_reading_status':status,'test_period':'2024-10-01 to 2024-12-31','cohorts':{}}
for cohort,frame in [('all_candidates',data),('direction_compatible',data[data.direction_status.eq('compatible')])]:
    test=frame[frame.crash_datetime.ge('2024-10-01')].copy()
    train=frame[frame.crash_datetime.lt('2024-10-01')].copy()
    before=len(train)
    train=train[~train.site_id.isin(test.site_id)].copy()
    result={'train_size':len(train),'test_size':len(test),
            'training_rows_removed_due_to_test_site':before-len(train),
            'train_counts':train.severity.value_counts().to_dict(),
            'test_counts':test.severity.value_counts().to_dict(),
            'predict_slight_accuracy':float(test.severity.eq('Slight').mean()) if len(test) else None,
            'results':{}}
    results['cohorts'][cohort]=result
    if not (len(train)>=100 and len(test)>=30 and train.severity.nunique()==3):
        result['comparison_not_run']='Insufficient data or missing training classes.'
        continue
    for name in ['aadf','sensor_density_proxy']:
        xtrain=train[features].copy(); xtest=test[features].copy()
        if name=='sensor_density_proxy':
            xtrain[FIELD]=train.density_proxy.to_numpy(); xtest[FIELD]=test.density_proxy.to_numpy()
        model=clone(bundle['pipeline'])
        model.fit(xtrain,train.severity)
        pred=model.predict(xtest)
        result['results'][name]={'accuracy':accuracy_score(test.severity,pred),
            'per_class':classification_report(test.severity,pred,labels=LABELS,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(test.severity,pred,labels=LABELS).tolist()}
(OUT/'sensor_comparison.json').write_text(json.dumps(results,indent=2))
data[['collision_index','site_id','site_name','distance_m','crash_datetime','sensor_end','volume_15m','average_mph','density_proxy','severity']].to_csv(OUT/'exploratory_matches.csv',index=False)
print(json.dumps(results,indent=2),flush=True)
