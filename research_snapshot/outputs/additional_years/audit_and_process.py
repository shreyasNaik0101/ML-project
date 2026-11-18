"""Audit raw STATS19 tables and test processing changes using temporal validation."""
import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1] / 'work'
REPO = WORK / 'github-ml-project'
OUT = HERE / 'data_processing'
OUT.mkdir(exist_ok=True)
LABELS = ['Slight', 'Serious', 'Fatal']


def report(y, predicted):
    return {'accuracy': float(accuracy_score(y, predicted)),
            'report': classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, predicted, labels=LABELS).tolist()}


def preprocess():
    existing = pd.read_csv(HERE / 'matched_2022_2025.csv.gz', dtype={'collision_index':str}, float_precision='round_trip')
    traffic = pd.read_csv(REPO / 'data/dft_traffic_counts_aadf.zip', low_memory=False)
    audits, frames = {}, []
    for year in [2022, 2023, 2024, 2025]:
        folder = REPO / 'data' if year == 2025 else WORK / 'yearly-data'
        source = folder / f'dft-road-casualty-statistics-collision-{year}.csv'
        crashes = pd.read_csv(source, dtype={'collision_index':str}, low_memory=False)
        vehicles = pd.read_csv(folder / f'dft-road-casualty-statistics-vehicle-{year}.csv',
                               dtype={'collision_index':str}, low_memory=False)
        if not crashes.collision_index.is_unique or vehicles.duplicated(['collision_index', 'vehicle_reference']).any():
            raise ValueError('Conflicting raw identifiers')
        matched = existing[existing.year == year].copy()
        raw = crashes.set_index('collision_index').loc[matched.collision_index]
        raw.index = matched.index
        assert np.array_equal(raw.collision_severity.map({1:'Fatal', 2:'Serious', 3:'Slight'}), matched.severity)
        catalog = traffic[traffic.year == year-1].set_index('count_point_id')
        expected = catalog.loc[matched.count_point_id, 'all_motor_vehicles'].to_numpy()
        np.testing.assert_array_equal(expected, matched.aadf_motor_vehicles_per_day)
        grouped = vehicles.groupby('collision_index').size()
        counted = crashes.collision_index.map(grouped).fillna(0)
        audit = {'source_rows':len(crashes), 'source_vehicle_rows':len(vehicles),
            'duplicate_crash_ids':int(crashes.collision_index.duplicated().sum()),
            'duplicate_vehicle_keys':int(vehicles.duplicated(['collision_index','vehicle_reference']).sum()),
            'orphan_vehicle_rows':int((~vehicles.collision_index.isin(crashes.collision_index)).sum()),
            'crashes_without_vehicle_records':int(counted.eq(0).sum()),
            'vehicle_count_disagreements':int(counted.ne(crashes.number_of_vehicles).sum()),
            'matched_rows':len(matched),
            'source_class_counts':crashes.collision_severity.value_counts().to_dict(),
            'matched_class_counts':matched.severity.value_counts().to_dict(),
            'raw_field_counts':{c:crashes[c].value_counts(dropna=False).to_dict()
                for c in ['road_type','weather_conditions','road_surface_conditions','speed_limit','collision_injury_based']},
            'vehicle_type_counts':vehicles.vehicle_type.value_counts(dropna=False).to_dict(),
            'vehicle_escooter_counts':vehicles.escooter_flag.value_counts(dropna=False).to_dict(),
            'aadf_method':catalog.loc[matched.count_point_id, 'estimation_method'].value_counts().to_dict(),
            'zero_aadf':int(matched.aadf_motor_vehicles_per_day.eq(0).sum()),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        # Keep unknown categories distinct. Encode numeric missing values as NaN.
        cleaned = matched.copy()
        for col, allowed in {'road_type':[1,2,3,6,7,9,12], 'weather_conditions':list(range(1,10)),
                             'road_surface_conditions':[1,2,3,4,5,6,7,9]}.items():
            values = pd.to_numeric(raw[col], errors='coerce')
            cleaned[col] = values.where(values.isin(allowed), -1)
            audit[col+'_invalid'] = int((~values.isin(allowed+[-1])).sum())
        speed = pd.to_numeric(raw.speed_limit, errors='coerce')
        cleaned['speed_limit'] = speed.where(speed.isin([20,30,40,50,60,70]))
        clock = pd.to_datetime(raw.time, format='%H:%M', errors='coerce')
        minutes = clock.dt.hour*60 + clock.dt.minute
        cleaned['time_sin'] = np.sin(2*np.pi*minutes/1440)
        cleaned['time_cos'] = np.cos(2*np.pi*minutes/1440)
        audit['invalid_clock_matched'] = int(clock.isna().sum())
        # Higher harmonics preserve continuity while allowing several daily peaks.
        cleaned['time_sin_2'] = np.sin(4*np.pi*minutes/1440)
        cleaned['time_cos_2'] = np.cos(4*np.pi*minutes/1440)
        cleaned['time_sin_3'] = np.sin(6*np.pi*minutes/1440)
        cleaned['time_cos_3'] = np.cos(6*np.pi*minutes/1440)
        # Composition is an optional representation derived from vehicle-type records.
        for name, codes in {'cycle':[1], 'motorcycle':[2,3,4,5,23,97], 'car_taxi':[8,9],
                            'bus':[10,11], 'goods':[19,20,21,98], 'other':[16,17,18,22,33,90]}.items():
            counts = vehicles[vehicles.vehicle_type.isin(codes)].groupby('collision_index').size()
            cleaned['vehicle_composition_'+name] = cleaned.collision_index.map(counts).fillna(0)
        audits[year] = audit
        frames.append(cleaned)
        print('Audited year',year,flush=True)
    processed = pd.concat(frames,ignore_index=True)
    processed.to_csv(OUT / 'processed_crashes.csv.gz',index=False)
    (OUT/'audit.json').write_text(json.dumps(audits,indent=2,default=str))
    return existing, processed


def select_weights(model, data, features):
    probabilities = model.predict_proba(data[features])
    classes = model.classes_.tolist()
    actual = pd.Categorical(data.severity, categories=classes).codes
    counts = np.bincount(actual, minlength=3)
    best = None
    for serious in [0.75,1,1.25,1.5,2,3]:
        for fatal in [1,2,4,6,8,12]:
            weights = np.array([{'Slight':1,'Serious':serious,'Fatal':fatal}[c] for c in classes])
            pred = (probabilities*weights).argmax(1)
            correct = np.bincount(actual[pred==actual],minlength=3)
            predcounts = np.bincount(pred,minlength=3)
            f1 = float(np.mean(2*correct/np.maximum(counts+predcounts,1)))
            if best is None or f1>best[0]:
                best=(f1, weights)
    pred = np.array(classes)[(probabilities*best[1]).argmax(1)]
    return best[1], report(data.severity, pred)


def run():
    raw, processed = preprocess()
    bundle = joblib.load(HERE/'gradient_boosting_2022_2023.joblib')
    base = bundle['features']
    time_cols = ['time_sin_2','time_cos_2','time_sin_3','time_cos_3']
    variants = [('baseline',raw,base),('validated_categories',processed,base),
                ('time_representation',processed,base+time_cols),
                ('vehicle_composition_optional',processed,base+time_cols+
                 [c for c in processed if c.startswith('vehicle_composition_')])]
    results, fitted = {}, {}
    for name,data,features in variants:
        train=data[data.year.isin([2022,2023])]
        val=data[data.year==2024]
        model=clone(bundle['pipeline'])
        numeric=[c for c in features if c not in ['road_type','weather_conditions','road_surface_conditions']
                 and not c.startswith('vehicle_type_')]
        # Add transformed numeric columns to the existing training-only imputer.
        from sklearn.compose import ColumnTransformer
        transforms = model.named_steps['prepare'].transformers
        transforms = [(name_,transform,numeric if name_=='numeric' else cols)
                      for name_,transform,cols in transforms]
        model.set_params(prepare=ColumnTransformer(transforms))
        model.fit(train[features],train.severity)
        weights,validation=select_weights(model,val,features)
        results[name]={'validation':validation,'decision_weights':dict(zip(model.classes_,weights.tolist())),
                       'features':features}
        fitted[name]=(model,weights,data,features)
        print(name,'validation macro F1',validation['report']['macro avg']['f1-score'],flush=True)
    # Select each candidate before accessing 2025 test scores.
    winner=max(results,key=lambda n:results[n]['validation']['report']['macro avg']['f1-score'])
    strict=max([n for n in results if n!='vehicle_composition_optional'],
               key=lambda n:results[n]['validation']['report']['macro avg']['f1-score'])
    for name in dict.fromkeys(['baseline',strict,winner]):
        model,weights,data,features=fitted[name]
        development=data[data.year<2025]
        test=data[data.year==2025]
        model.fit(development[features],development.severity)
        pred=model.classes_[(model.predict_proba(test[features])*weights).argmax(1)]
        results[name]['test']=report(test.severity,pred)
        joblib.dump(dict(pipeline=model,features=features,
                        decision_weights=dict(zip(model.classes_,weights.tolist()))),
                    OUT/f'{name}.joblib',compress=3)
        loaded=joblib.load(OUT/f'{name}.joblib')
        np.testing.assert_array_equal(pred,loaded['pipeline'].classes_[
            (loaded['pipeline'].predict_proba(test[features])*weights).argmax(1)])
    results['selected_variant']=winner
    results['selected_strict_variant']=strict
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
