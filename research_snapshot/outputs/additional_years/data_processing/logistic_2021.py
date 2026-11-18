"""Compare additional 2021 records using the shared logistic-regression setup."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from compare_three_models import make_model, fit
from train_xgboost import LABELS, best_weights, evaluate

ROOT=Path(__file__).resolve().parent


def main():
    data=pd.read_csv(ROOT.parent/'matched_2021_2025.csv.gz',dtype={'collision_index':str},float_precision='round_trip')
    reference=joblib.load(ROOT/'vehicle_composition_optional.joblib')
    features=reference['features']
    preprocessor=reference['pipeline'].named_steps['prepare']
    base=data[data.year.isin([2022,2023])]
    old=data[data.year==2021]
    validation=data[data.year==2024]
    test=data[data.year==2025]
    groups={'no_2021':base,'all_2021':pd.concat([old,base]),
            'serious_fatal_2021':pd.concat([old[old.severity.isin(['Serious','Fatal'])],base])}
    results={}
    for name,train in groups.items():
        model=make_model('Logistic Regression',1,preprocessor)
        fit(model,train,features,False)
        val_prob=model.predict_proba(validation[features])
        score,weights=best_weights(val_prob,validation.severity)
        validation_result=evaluate(validation.severity,LABELS[(val_prob*weights).argmax(1)])
        fit(model,pd.concat([train,validation]),features,False)
        pred=LABELS[(model.predict_proba(test[features])*weights).argmax(1)]
        result=evaluate(test.severity,pred)
        bundle={'pipeline':model,'features':features,'class_order':LABELS.tolist(),
                'decision_weights':dict(zip(LABELS.tolist(),weights.tolist()))}
        joblib.dump(bundle,ROOT.parent/f'logistic_{name}.joblib',compress=3)
        loaded=joblib.load(ROOT.parent/f'logistic_{name}.joblib')
        np.testing.assert_array_equal(pred,LABELS[(loaded['pipeline'].predict_proba(test[features])*weights).argmax(1)])
        results[name]={'train_rows':len(train),'validation_macro_f1':score,'validation':validation_result,
                       'decision_weights':bundle['decision_weights'],'test':result}
        print(name,json.dumps(result),flush=True)
    (ROOT.parent/'logistic_2021_results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
