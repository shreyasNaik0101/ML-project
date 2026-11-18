"""Compare accuracy-oriented classifiers with shared temporal validation."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from compare_three_models import make_model, fit
from train_xgboost import LABELS, evaluate

ROOT=Path(__file__).resolve().parent


def main():
    data=pd.read_csv(ROOT/'processed_crashes.csv.gz',dtype={'collision_index':str},float_precision='round_trip')
    ref=joblib.load(ROOT/'vehicle_composition_optional.joblib')
    features=ref['features']
    prep=ref['pipeline'].named_steps['prepare']
    train=data[data.year.isin([2022,2023])]
    val=data[data.year==2024]
    actual=pd.Categorical(val.severity,categories=LABELS).codes
    counts=np.bincount(actual,minlength=3)
    dev=data[data.year<2025]
    test=data[data.year==2025]
    families={'Logistic Regression':[0.1,1], 'Random Forest':[8,14], 'XGBoost':[3,5]}
    output={'selection':'Maximum macro F1 on 2024 with accuracy >=72% and fatal recall >=10%; both constraints apply only to validation.',
            'models':{},'class_order':LABELS.tolist(),
            'note':'2025 has been inspected previously and road links can recur between years.'}
    for family,params in families.items():
        trials=[]
        for parameter in params:
            for weighted in [False,True]:
                model=make_model(family,parameter,prep)
                fit(model,train,features,weighted)
                probs=model.predict_proba(val[features])
                best=None
                for serious in np.linspace(1,4,31):
                    for fatal in range(1,21):
                        weights=np.array([fatal,serious,1.0])
                        pred=(probs*weights).argmax(axis=1)
                        hits=np.bincount(actual[pred==actual],minlength=3)
                        predicted=np.bincount(pred,minlength=3)
                        accuracy=float(hits.sum()/len(actual))
                        fatal_recall=float(hits[0]/counts[0])
                        macro=float((2*hits/np.maximum(counts+predicted,1)).mean())
                        if accuracy>=.72 and fatal_recall>=.10:
                            trial=(macro,accuracy,fatal_recall,float(serious),fatal)
                            if best is None or trial>best:
                                best=trial
                trials.append({'parameter':parameter,'weighted':weighted,'best':best})
                print(family,parameter,weighted,best,flush=True)
        feasible=[trial for trial in trials if trial['best'] is not None]
        if not feasible:
            output['models'][family]={'trials':trials,'failure':'No candidate reached validation targets'}
            continue
        selected=max(feasible,key=lambda t:t['best'])
        _,_,_,serious,fatal=selected['best']
        weights=np.array([fatal,serious,1.0])
        model=make_model(family,selected['parameter'],prep)
        fit(model,dev,features,selected['weighted'])
        pred=LABELS[(model.predict_proba(test[features])*weights).argmax(axis=1)]
        bundle={'pipeline':model,'features':features,'class_order':LABELS.tolist(),
                'decision_weights':dict(zip(LABELS.tolist(),weights.tolist()))}
        name=family.lower().replace(' ','_')
        joblib.dump(bundle,ROOT/f'accuracy_comparison_{name}.joblib',compress=3)
        loaded=joblib.load(ROOT/f'accuracy_comparison_{name}.joblib')
        np.testing.assert_array_equal(pred,LABELS[(loaded['pipeline'].predict_proba(test[features])*weights).argmax(axis=1)])
        output['models'][family]={'selected':selected,'trials':trials,'test':evaluate(test.severity,pred)}
        print('TEST',family,output['models'][family]['test'],flush=True)
    (ROOT/'accuracy_model_comparison.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
