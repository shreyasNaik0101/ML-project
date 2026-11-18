"""Add 2021 crash data and compare sampling strategies using the same test cohort."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]/'work'
sys.path.insert(0,str(WORK/'github-ml-project'))
from experiment_more_years import prepare_year
from train_xgboost import LABELS, best_weights, evaluate
from compare_three_models import make_model, fit


def prepare_2021(features):
    base,audit=prepare_year(2021)
    folder=WORK/'yearly-data'
    vehicles=pd.read_csv(folder/'dft-road-casualty-statistics-vehicle-2021.csv',dtype={'collision_index':str})
    crashes=pd.read_csv(folder/'dft-road-casualty-statistics-collision-2021.csv',dtype={'collision_index':str},
                        usecols=['collision_index','time'])
    if vehicles.duplicated(['collision_index','vehicle_reference']).any() or not crashes.collision_index.is_unique:
        raise ValueError('Duplicate IDs found')
    base=base.merge(crashes,on='collision_index',validate='one_to_one')
    clock=pd.to_datetime(base.pop('time'),format='%H:%M',errors='coerce')
    minutes=clock.dt.hour*60+clock.dt.minute
    for harmonic in [2,3]:
        base[f'time_sin_{harmonic}']=np.sin(harmonic*2*np.pi*minutes/1440)
        base[f'time_cos_{harmonic}']=np.cos(harmonic*2*np.pi*minutes/1440)
    categories={'cycle':[1], 'motorcycle':[2,3,4,5,23,97], 'car_taxi':[8,9],
                'bus':[10,11], 'goods':[19,20,21,98], 'other':[16,17,18,22,33,90]}
    for label,codes in categories.items():
        counts=vehicles[vehicles.vehicle_type.isin(codes)].groupby('collision_index').size()
        base['vehicle_composition_'+label]=base.collision_index.map(counts).fillna(0)
    for col in features:
        if col.startswith('vehicle_type_') and col not in base:
            base[col]=0
    # Other years retain unknown / missing categorical values under the same encoding.
    return base[['collision_index','severity','year','count_point_id',*features]],audit


def main():
    root=ROOT.parent
    latest=pd.read_csv(ROOT/'processed_crashes.csv.gz',dtype={'collision_index':str},float_precision='round_trip')
    reference=joblib.load(ROOT/'vehicle_composition_optional.joblib')
    features=reference['features']
    prep=reference['pipeline'].named_steps['prepare']
    historical,audit=prepare_2021(features)
    latest=latest[['collision_index','severity','year','count_point_id',*features]]
    data=pd.concat([historical,latest],ignore_index=True)
    assert data.collision_index.is_unique
    data.to_csv(root/'matched_2021_2025.csv.gz',index=False)
    train_base=data[data.year.isin([2022,2023])]
    prior=data[data.year==2021]
    val=data[data.year==2024]
    test=data[data.year==2025]
    methods={'2022_2023':train_base,'add_2021_all':pd.concat([prior,train_base]),
             'add_2021_serious_fatal':pd.concat([prior[prior.severity.isin(['Serious','Fatal'])],train_base])}
    output={'2021_audit':audit,'year_counts':pd.crosstab(data.year,data.severity).to_dict(orient='index'),
            'protocol':'Validation on 2024, then refit with 2024 and score inspected 2025. All 2021 records are prior to validation.',
            'experiments':{}}
    for family,parameter,weighted in [('Random Forest',14,True),('XGBoost',5,False)]:
        for name,train in methods.items():
            model=make_model(family,parameter,prep)
            fit(model,train,features,weighted)
            score,weights=best_weights(model.predict_proba(val[features]),val.severity)
            valpred=LABELS[(model.predict_proba(val[features])*weights).argmax(1)]
            val_result=evaluate(val.severity,valpred)
            development=pd.concat([train,val])
            fit(model,development,features,weighted)
            pred=LABELS[(model.predict_proba(test[features])*weights).argmax(1)]
            result=evaluate(test.severity,pred)
            key=family.lower().replace(' ','_')+'_'+name
            bundle={'pipeline':model,'features':features,'class_order':LABELS.tolist(),
                    'decision_weights':dict(zip(LABELS.tolist(),weights.tolist()))}
            joblib.dump(bundle,root/f'{key}.joblib',compress=3)
            loaded=joblib.load(root/f'{key}.joblib')
            np.testing.assert_array_equal(pred,LABELS[(loaded['pipeline'].predict_proba(test[features])*weights).argmax(1)])
            output['experiments'][key]={'train_rows':len(train),'validation_macro_f1':score,
                 'validation':val_result,'weights':bundle['decision_weights'],'test':result}
            print(key,'train',len(train),'validation',score,'test',json.dumps(result),flush=True)
    (root/'additional_2021_results.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
