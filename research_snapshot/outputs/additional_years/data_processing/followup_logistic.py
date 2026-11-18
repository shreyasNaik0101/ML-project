"""Tune logistic regression using 2024 validation, then evaluate on 2025."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_xgboost import LABELS, evaluate

ROOT=Path(__file__).resolve().parent


def main():
    data=pd.read_csv(ROOT/'processed_crashes.csv.gz',dtype={'collision_index':str},float_precision='round_trip')
    base=joblib.load(ROOT/'baseline.joblib')
    enriched=joblib.load(ROOT/'vehicle_composition_optional.joblib')
    train=data[data.year.isin([2022,2023])]
    validation=data[data.year==2024]
    actual=pd.Categorical(validation.severity,categories=LABELS).codes
    counts=np.bincount(actual,minlength=3)
    y_train=pd.Categorical(train.severity,categories=LABELS).codes
    class_weights=[None,{0:5,1:2,2:1},{0:10,1:2,2:1}]
    trials=[]
    for variant,bundle in [('original',base),('composition_time',enriched)]:
        features=bundle['features']
        prep=bundle['pipeline'].named_steps['prepare']
        for regularization in [0.1,1,10]:
            for class_weight in class_weights:
                model=Pipeline([('prepare',clone(prep)),('scale',StandardScaler()),
                                ('model',LogisticRegression(C=regularization,class_weight=class_weight,max_iter=1500))])
                model.fit(train[features],y_train)
                probabilities=model.predict_proba(validation[features])
                best=None
                for serious in np.linspace(1,4,16):
                    for fatal in [1,2,3,4,6,8,12,16]:
                        weights=np.array([fatal,serious,1.0])
                        predictions=(probabilities*weights).argmax(1)
                        correct=np.bincount(actual[predictions==actual],minlength=3)
                        totals=np.bincount(predictions,minlength=3)
                        accuracy=float(correct.sum()/len(actual))
                        fatal_recall=float(correct[0]/counts[0])
                        macro=float((2*correct/np.maximum(counts+totals,1)).mean())
                        if accuracy>=0.70 and fatal_recall>=0.20:
                            candidate=(macro,accuracy,fatal_recall,serious,fatal)
                            if best is None or candidate>best:
                                best=candidate
                trials.append({'variant':variant,'C':regularization,'class_weight':class_weight,
                               'best_feasible':best})
                print(variant,regularization,class_weight,best,flush=True)
    feasible=[t for t in trials if t['best_feasible'] is not None]
    if not feasible:
        raise ValueError('No candidate met the validation accuracy and fatal recall targets')
    selected=max(feasible,key=lambda t:t['best_feasible'][:3])
    macro,accuracy,recall,serious,fatal=selected['best_feasible']
    bundle=base if selected['variant']=='original' else enriched
    features=bundle['features']
    model=Pipeline([('prepare',clone(bundle['pipeline'].named_steps['prepare'])),
                    ('scale',StandardScaler()),('model',LogisticRegression(C=selected['C'],
                    class_weight=selected['class_weight'],max_iter=1500))])
    model.fit(train[features],y_train)
    weights=np.array([fatal,serious,1.0])
    validation_result=evaluate(validation.severity,LABELS[(model.predict_proba(validation[features])*weights).argmax(1)])
    dev=data[data.year<2025]
    model.fit(dev[features],pd.Categorical(dev.severity,categories=LABELS).codes)
    test=data[data.year==2025]
    predicted=LABELS[(model.predict_proba(test[features])*weights).argmax(1)]
    result=evaluate(test.severity,predicted)
    artifact={'pipeline':model,'features':features,'class_order':LABELS.tolist(),
              'decision_weights':dict(zip(LABELS.tolist(),weights.tolist()))}
    joblib.dump(artifact,ROOT/'logistic_followup.joblib',compress=3)
    loaded=joblib.load(ROOT/'logistic_followup.joblib')
    np.testing.assert_array_equal(predicted,LABELS[(loaded['pipeline'].predict_proba(test[features])*weights).argmax(1)])
    output={'selection':'Highest validation macro F1 with accuracy >=70% and fatal recall >=20%; 2024 only',
            'selected':selected,'trials':trials,'validation':validation_result,'test':result,
            'note':'The 2025 test cohort was examined previously; links may overlap across years.'}
    (ROOT/'logistic_followup_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'selected':selected,'validation':validation_result,'test':result},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
