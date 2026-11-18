"""Select XGBoost score weights for 70% validation accuracy and report class trade-offs."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from train_xgboost import LABELS, evaluate

ROOT=Path(__file__).resolve().parent


def main():
    data=pd.read_csv(ROOT/'processed_crashes.csv.gz',dtype={'collision_index':str},float_precision='round_trip')
    saved=joblib.load(ROOT/'xgboost.joblib')
    features=saved['features']
    train=data[data.year.isin([2022,2023])]
    val=data[data.year==2024]
    test=data[data.year==2025]
    model=clone(saved['pipeline'])
    model.fit(train[features],pd.Categorical(train.severity,categories=LABELS).codes)
    probs=model.predict_proba(val[features])
    actual=pd.Categorical(val.severity,categories=LABELS).codes
    counts=np.bincount(actual,minlength=3)
    choices=[]
    for serious in np.linspace(1,4,31):
        for fatal in np.arange(1,21):
            w=np.array([fatal,serious,1.0])
            pred=(probs*w).argmax(1)
            hits=np.bincount(actual[pred==actual],minlength=3)
            predcounts=np.bincount(pred,minlength=3)
            macro=float(np.mean(2*hits/np.maximum(counts+predcounts,1)))
            accuracy=float(hits.sum()/len(actual))
            recall=float(hits[0]/counts[0])
            if accuracy >= 0.72 and recall >= 0.10:
                choices.append((macro,accuracy,recall,serious,fatal))
    if not choices:
        raise ValueError('No candidate met both validation targets')
    chosen=max(choices)
    weights=np.array([chosen[4],chosen[3],1.0])
    val_result=evaluate(val.severity,LABELS[(probs*weights).argmax(1)])
    dev=data[data.year<2025]
    model.fit(dev[features],pd.Categorical(dev.severity,categories=LABELS).codes)
    pred=LABELS[(model.predict_proba(test[features])*weights).argmax(1)]
    result=evaluate(test.severity,pred)
    bundle={'pipeline':model,'features':features,'class_order':LABELS.tolist(),
            'decision_weights':dict(zip(LABELS.tolist(),weights.tolist()))}
    joblib.dump(bundle,ROOT/'xgboost_accuracy_72_validation.joblib',compress=3)
    loaded=joblib.load(ROOT/'xgboost_accuracy_72_validation.joblib')
    np.testing.assert_array_equal(pred,LABELS[(loaded['pipeline'].predict_proba(test[features])*weights).argmax(1)])
    record={'criterion':'Maximum validation macro F1 among decision weights with accuracy >=72% and fatal recall >=10%; both constraints on 2024 only.',
            'weights':bundle['decision_weights'],'validation':val_result,'test':result,
            'test_note':'2025 was inspected in earlier experiments and road links may overlap across years.'}
    (ROOT/'accuracy_72_validation_results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
