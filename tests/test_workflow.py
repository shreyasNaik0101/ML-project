import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from modeling import choose_score_weights
from predict import predict
from scenario import prepare
from train_models import validate_data
from evaluate_unseen_links import partition_links

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / 'aadf_experiment/matched_crashes.csv.gz', dtype={'collision_index': str})

    def test_shared_split_and_no_leakage_features(self):
        validate_data(self.data)
        self.assertEqual(len(self.data), 19397)
        for path in (ROOT / 'artifacts').glob('*.joblib'):
            bundle = joblib.load(path)
            self.assertFalse({'severity', 'split', 'collision_index'} & set(bundle['features']))

    def test_readable_scenario(self):
        import json
        scenario = json.loads((ROOT / 'examples/scenario.json').read_text())
        bundle = joblib.load(ROOT / 'artifacts/logistic_regression_with_aadf.joblib')
        rows = prepare(scenario, bundle['features'])
        self.assertEqual(rows.loc[0, 'vehicle_type_9'], 1)
        self.assertEqual(rows.loc[0, 'vehicle_type_1'], 1)
        self.assertAlmostEqual(rows.loc[0, 'time_sin'] ** 2 + rows.loc[0, 'time_cos'] ** 2, 1)
        self.assertEqual(len(predict(bundle, rows)), 1)
        with self.assertRaises(ValueError):
            prepare(dict(scenario, time='25:00'), bundle['features'])
        with self.assertRaises(ValueError):
            prepare(dict(scenario, vehicles=[]), bundle['features'])

    def test_grouped_partitions_are_disjoint(self):
        rows = pd.DataFrame({'count_point_id': np.repeat(np.arange(50), 3)})
        result = partition_links(rows)
        groups = [set(part['count_point_id']) for _, part in result.groupby('split')]
        self.assertEqual(len(groups), 3)
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        self.assertEqual(len(result), len(rows))

    def test_reject_unexpected_predictors(self):
        with self.assertRaisesRegex(ValueError, 'leak'):
            validate_data(self.data.assign(number_of_casualties=1))

    def test_scenario_rejects_wrong_json_types(self):
        import json
        scenario = json.loads((ROOT / 'examples/scenario.json').read_text())
        bundle = joblib.load(ROOT / 'artifacts/logistic_regression_with_aadf.joblib')
        for value in ([], None, dict(scenario, time=None), dict(scenario, vehicles=[{}])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare(value, bundle['features'])

    def test_weights_follow_class_order(self):
        classes = ['Fatal', 'Serious', 'Slight']
        raw = np.array([[.2, .3, .5], [.8, .1, .1], [.1, .7, .2]])
        actual = ['Slight', 'Fatal', 'Serious']
        score, weights = choose_score_weights(raw, actual, classes)
        predicted = np.array(classes)[(raw * [weights[c] for c in classes]).argmax(axis=1)]
        self.assertAlmostEqual(score[0], f1_score(actual, predicted, average='macro'))
        self.assertEqual(weights['Slight'], 1.)

    def test_all_exported_models_predict_and_round_trip(self):
        paths = list((ROOT / 'artifacts').glob('*.joblib'))
        self.assertEqual(len(paths), 6, 'Run python train_models.py first')
        rows = self.data.head(10)
        for path in paths:
            with self.subTest(model=path.name):
                bundle = joblib.load(path)
                result = predict(bundle, rows)
                model = bundle['pipeline']
                weights = [bundle['decision_weights'][c] for c in model.classes_]
                expected = model.classes_[(model.predict_proba(rows[bundle['features']]) * weights).argmax(axis=1)]
                np.testing.assert_array_equal(result['predicted_severity'], expected)
                np.testing.assert_allclose(result.filter(like='raw_score').sum(axis=1), 1.)
                self.assertEqual(result['collision_index'].tolist(), rows['collision_index'].tolist())

    def test_reject_duplicate_ids_and_invalid_split(self):
        with self.assertRaises(ValueError):
            validate_data(pd.concat([self.data, self.data.head(1)]))
        with self.assertRaises(ValueError):
            validate_data(self.data.assign(split='train'))

    def test_reject_missing_features_and_bad_flags(self):
        bundle = joblib.load(ROOT / 'artifacts/logistic_regression_with_aadf.joblib')
        with self.assertRaisesRegex(ValueError, 'Missing feature'):
            predict(bundle, self.data.drop(columns=['speed_limit']).head(1))
        with self.assertRaisesRegex(ValueError, 'flags'):
            predict(bundle, self.data.head(1).assign(vehicle_type_9=2))
        with self.assertRaisesRegex(ValueError, 'AADF'):
            predict(bundle, self.data.head(1).assign(aadf_motor_vehicles_per_day=-1))


if __name__ == '__main__':
    unittest.main()
