import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from experiments.temporal_enhancements.run import LABELS, NUMERIC, metrics, operating_points, split_data, make_model, fit, run


class TemporalEnhancementTests(unittest.TestCase):
    def test_selection_does_not_use_2025_labels(self):
        import tempfile
        from pathlib import Path
        from contextlib import redirect_stdout
        from io import StringIO
        rows = pd.DataFrame({name: [1.]*90 for name in NUMERIC})
        for name in ['road_type','weather_conditions','road_surface_conditions']:
            rows[name] = [1,2,3]*30
        rows['collision_index'] = np.arange(90).astype(str)
        rows['count_point_id'] = np.arange(90)
        rows['year'] = [2021]*30 + [2024]*30 + [2025]*30
        rows['severity'] = ['Fatal','Serious','Slight']*30
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            path = Path(tmp)/'data.csv.gz'
            rows.to_csv(path, index=False)
            first = run(path, Path(tmp)/'first', limit=1)
            rows.loc[rows.year == 2025, 'severity'] = 'Slight'
            rows.to_csv(path, index=False)
            second = run(path, Path(tmp)/'second', limit=1)
            self.assertEqual(first['selected'], second['selected'])
            self.assertFalse((Path(tmp)/'first/inspected_test.json').exists())

    def test_catboost_pipeline_supports_sklearn_and_reload(self):
        import joblib
        import tempfile
        from pathlib import Path
        rows = pd.DataFrame({'road_type':[1,2,3]*4,'weather_conditions':[1]*12,
                             'road_surface_conditions':[1]*12,'speed_limit':[30,50,70]*4,
                             'severity':['Fatal','Serious','Slight']*4})
        columns = list(rows.columns[:-1])
        config = {'family':'cat','params':{'depth':2},'weighted':False}
        model = fit(make_model(config, columns), config, rows, columns)
        raw = model.predict_proba(rows[columns])
        np.testing.assert_allclose(raw.sum(axis=1), 1.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(rows[columns]), raw)

    def test_temporal_partitions_and_duplicate_ids(self):
        data = pd.DataFrame({'collision_index': list('abcdef'), 'year': [2021,2022,2023,2024,2025,2025],
                             'severity': ['Slight','Serious','Fatal']*2})
        train, validation, test = split_data(data)
        self.assertEqual(train.year.tolist(), [2021,2022,2023])
        self.assertEqual(validation.year.tolist(), [2024])
        self.assertEqual(test.year.tolist(), [2025,2025])
        with self.assertRaises(ValueError):
            split_data(pd.concat([data, data.head(1)]))

    def test_score_and_label_order(self):
        y = ['Slight','Fatal','Serious','Fatal']
        codes = np.array([2,0,2,1])
        score = metrics(y, codes)
        self.assertAlmostEqual(score['accuracy'], accuracy_score(y, LABELS[codes]))
        self.assertAlmostEqual(score['macro_f1'], f1_score(y, LABELS[codes], labels=LABELS, average='macro', zero_division=0))
        self.assertEqual(score['confusion_matrix'], [[1,1,0],[0,0,1],[0,0,1]])

    def test_constraints_are_enforced_not_relaxed(self):
        y = ['Fatal','Serious','Slight']*2
        result = operating_points(y, np.tile([0.,0.,1.], (6,1)))
        self.assertIsNone(result['constrained'])
        result = operating_points(y, np.eye(3)[[0,1,2]*2])
        self.assertGreaterEqual(result['constrained']['recall']['Fatal'], .22)
        self.assertGreaterEqual(result['constrained']['recall']['Serious'], .33)

    def test_training_preprocessing_ignores_future_rows(self):
        columns = ['road_type','weather_conditions','road_surface_conditions','speed_limit']
        rows = pd.DataFrame({'road_type':[1,2,3]*4,'weather_conditions':[1]*12,
                             'road_surface_conditions':[1]*12,'speed_limit':[30,50,70]*4,
                             'severity':['Fatal','Serious','Slight']*4})
        config = {'family':'rf','params':{'n_estimators':5},'weighted':True}
        model = fit(make_model(config, columns), config, rows, columns)
        encoder = model.named_steps['prepare'].named_transformers_['categories'].named_steps['encode']
        self.assertNotIn(99, encoder.categories_[0])
        self.assertEqual(model.predict_proba(rows[columns].assign(road_type=99)).shape, (12,3))


if __name__ == '__main__':
    unittest.main()
