"""Compare AADF and sensor density on a matched, retrospective development cohort."""
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
candidate=pd.read_csv(OUT/'multiyear_candidates.csv',dtype={'collision_index':str})
features_data=pd.read_csv(ROOT/'outputs/additional_years/data_processing/processed_crashes.csv.gz',dtype={'collision_index':str})
data=candidate.merge(features_data.drop(columns=['year','count_point_id']),on='collision_index',validate='one_to_one')
data['crash_datetime']=pd.to_datetime(data.date+' '+data.time,format='%d/%m/%Y %H:%M',errors='coerce')
records=[]; statuses=[]
for row in data.itertuples(index=False):
    dt=row.crash_datetime
    if pd.isna(dt):
        statuses.append((row.collision_index,'invalid_time')); continue
    days=[dt]
    if dt.hour<2: days.append(dt-pd.Timedelta(days=1))
    rows=[]; missing=0; incomplete=False
    for day in days:
        p=ROOT/'work/traffic_pilot/readings'/f'{row.site_id}_{day:%d%m%Y}.json'
        if not p.exists(): missing+=1; continue
        obj=json.loads(p.read_text())
        if obj.get('Header',{}).get('row_count',0)>len(obj.get('Rows',[])):
            incomplete=True; continue
        rows.extend(obj.get('Rows',[]))
    usable=[]; window_rows=0; bad_measurements=0
    for r in rows:
        try:
            end=pd.Timestamp(r['Report Date'][:10]+' '+r['Time Period Ending'])+pd.Timedelta(minutes=1)
            # Conservative buffer against a possible one-hour clock mismatch;
            # the source time-zone conventions still need authoritative checking.
            if not dt-pd.Timedelta(minutes=105)<=end<=dt-pd.Timedelta(minutes=75): continue
            window_rows+=1
            try:
                volume=float(r['Total Volume']); speed=float(r['Avg mph'])
                if volume>=0 and speed>0: usable.append((end,volume,speed))
                else: bad_measurements+=1
            except (KeyError,TypeError,ValueError): bad_measurements+=1
        except (KeyError,TypeError,ValueError): pass
    if not usable:
        status=('missing_download' if missing else 'incomplete_pagination' if incomplete else
                'no_rows_in_window' if not window_rows else 'invalid_or_missing_traffic_values')
        statuses.append((row.collision_index,status)); continue
    end,volume,speed=max(usable)
    records.append({'collision_index':row.collision_index,'sensor_end_exclusive':str(end),
                    'volume_15m':volume,'average_mph':speed,'density_proxy':4*volume/speed})
    statuses.append((row.collision_index,'usable'))
data=data.merge(pd.DataFrame(records),on='collision_index',validate='one_to_one')
train=data[data.year.isin([2022,2023])].copy()
test=data[data.year.eq(2024)].copy()
before=len(train)
train=train[~train.site_id.isin(test.site_id)].copy()
bundle=joblib.load(ROOT/'outputs/additional_years/gradient_boosting_2022_2023.joblib')
features=bundle['features']
results={'candidate_counts_by_year':candidate.year.value_counts().sort_index().to_dict(),
         'reading_status':pd.Series([s for _,s in statuses]).value_counts().to_dict(),
         'usable_counts_by_year':data.year.value_counts().sort_index().to_dict(),
         'excluded_overlapping_training_sensor_rows':before-len(train),
         'train_size':len(train),'test_size':len(test),
         'train_counts':train.severity.value_counts().to_dict(),'test_counts':test.severity.value_counts().to_dict(),
         'predict_slight_accuracy':float(test.severity.eq('Slight').mean()) if len(test) else None,
         'comparisons':{}}
if len(train)>=100 and len(test)>=30 and train.severity.nunique()==3:
    predictions={}
    with threadpool_limits(limits=2):
        for name in ['aadf','sensor_density_proxy']:
            xtrain=train[features].copy(); xtest=test[features].copy()
            if name=='sensor_density_proxy':
                xtrain[FIELD]=train.density_proxy.to_numpy(); xtest[FIELD]=test.density_proxy.to_numpy()
            model=clone(bundle['pipeline']); model.fit(xtrain,train.severity); pred=model.predict(xtest)
            predictions[name]=pred
            results['comparisons'][name]={'accuracy':float(accuracy_score(test.severity,pred)),
                'per_class':classification_report(test.severity,pred,labels=LABELS,output_dict=True,zero_division=0),
                'confusion_matrix':confusion_matrix(test.severity,pred,labels=LABELS).tolist()}
    truth=test.severity.to_numpy()
    delta=(predictions['sensor_density_proxy']==truth).astype(int)-(predictions['aadf']==truth).astype(int)
    rng=np.random.default_rng(42)
    bootstrap=np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(3000)])
    results['paired_accuracy_difference']={'sensor_minus_aadf':float(delta.mean()),
        'bootstrap_95_percent_interval':np.quantile(bootstrap,[.025,.975]).tolist(),
        'sensor_only_correct':int((delta==1).sum()),'aadf_only_correct':int((delta==-1).sum()),
        'note':'Exploratory row bootstrap; ignores correlations between nearby road sites.'}
else: results['comparison_not_run']='Insufficient matched observations or training classes.'
(OUT/'multiyear_results.json').write_text(json.dumps(results,indent=2))
data[['collision_index','year','site_id','site_name','site_direction','crash_sensor_m','sensor_link_m',
      'crash_datetime','sensor_end_exclusive','volume_15m','average_mph','density_proxy','severity']].to_csv(OUT/'multiyear_usable.csv',index=False)
print(json.dumps(results,indent=2),flush=True)
