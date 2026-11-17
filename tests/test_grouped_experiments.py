import unittest
import numpy as np
import pandas as pd
from grouped_experiments import (Features, folds_for, resample_indices, decision_rules, pipeline, metrics)
from modeling import LABELS, AADF


class GroupedExperimentTests(unittest.TestCase):
    def test_folds_keep_links_together(self):
        data = pd.DataFrame({'collision_index': np.arange(90), 'count_point_id': np.repeat(np.arange(30), 3), 'severity': LABELS*30})
        folds = folds_for(data)
        seen = []
        for train, valid in folds:
            self.assertFalse(set(data.iloc[train].count_point_id) & set(data.iloc[valid].count_point_id))
            seen.extend(valid)
        self.assertEqual(sorted(seen), list(range(90)))
        with self.assertRaises(ValueError):
            folds_for(data.assign(count_point_id=np.nan))

    def test_resampling_only_uses_supplied_training_rows(self):
        y = ['Slight']*10 + ['Serious']*4 + ['Fatal']*2
        for method, size in [('none', 16), ('over', 30), ('under', 6)]:
            indices = resample_indices(y, method, 42)
            self.assertEqual(len(indices), size)
            self.assertTrue(np.all((indices >= 0) & (indices < len(y))))
            np.testing.assert_array_equal(indices, resample_indices(y, method, 42))

    def test_infeasible_requirements_do_not_silently_relax(self):
        rules = decision_rules(LABELS, np.tile([.9, .09, .01], (3, 1)), 1., 1.)
        self.assertIsNone(rules['constrained'])
        self.assertIn('accuracy', rules)

    def test_decision_rule_metrics_match_sklearn(self):
        probabilities = np.random.default_rng(42).dirichlet(np.ones(3), size=30)
        actual = LABELS*10
        for point in decision_rules(actual, probabilities, 0., 0.).values():
            prediction = np.asarray(LABELS)[(probabilities*np.array([point['weights'][c] for c in LABELS])).argmax(axis=1)]
            expected = metrics(actual, prediction)
            self.assertAlmostEqual(point['macro_f1'], expected['macro_f1'])
            self.assertAlmostEqual(point['accuracy'], expected['accuracy'])

    def test_fatal_as_slight_and_fixed_label_order(self):
        result = metrics(['Fatal', 'Serious', 'Slight'], ['Slight']*3)
        self.assertEqual(result['fatal_as_slight'], 1)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0]]*3)

    def test_search_selection_is_independent_of_holdout_labels(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import joblib
        from grouped_experiments import run
        data = pd.DataFrame({'collision_index': np.arange(99), 'count_point_id': np.repeat(np.arange(33), 3),
            'severity': LABELS*33, 'split': ['train']*72 + ['validation']*18 + ['test']*9,
            'road_type': [1]*99, 'weather_conditions': [1]*99, 'road_surface_conditions': [1]*99,
            'speed_limit': [30, 50, 70]*33, 'time_sin': [0.]*99, 'time_cos': [1.]*99, AADF: [100.]*99})
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            first = run(data, Path(tmp)/'first', trials=1, minimum_recall=0.)
            data.loc[data.split == 'test', 'severity'] = 'Slight'
            second = run(data, Path(tmp)/'second', trials=1, minimum_recall=0.)
            self.assertEqual(first['selected'], second['selected'])
            bundle = joblib.load(Path(tmp)/'first/selected.joblib')
            self.assertEqual(len(bundle['pipeline'].predict(data[bundle['features']])), 99)

    def test_calendar_features_are_optional_ablation(self):
        columns = ['road_type', 'weather_conditions', 'road_surface_conditions', 'speed_limit', 'time_sin', 'time_cos', AADF, 'month', 'weekend']
        original = pipeline(columns, 'decision_tree', {'max_depth': 4}, False)
        engineered = pipeline(columns, 'decision_tree', {'max_depth': 4}, True)
        self.assertNotIn('month', original.named_steps['preprocess'].transformers[1][2])
        self.assertIn('month', engineered.named_steps['preprocess'].transformers[1][2])

    def test_feature_transform_and_unseen_categories(self):
        rows = pd.DataFrame({'road_type': [1, 2, 1, 2, 1, 2], 'weather_conditions': [1]*6,
            'road_surface_conditions': [1]*6, 'speed_limit': [30, 40, 50, 60, 70, 30],
            'time_sin': [0.]*6, 'time_cos': [1.]*6, AADF: [100.]*6,
            'vehicle_type_1': [0, 1]*3})
        copy = rows.copy(deep=True)
        transformed = Features(True).transform(rows)
        pd.testing.assert_frame_equal(rows, copy)
        self.assertAlmostEqual(transformed[AADF].iloc[0], np.log1p(100))
        self.assertEqual(transformed.cycles.tolist(), rows.vehicle_type_1.tolist())
        model = pipeline(list(rows), 'logistic_regression', {'C': 1.}, True)
        model.fit(rows, LABELS*2)
        raw = model.predict_proba(rows.assign(road_type=99))
        np.testing.assert_allclose(raw.sum(axis=1), 1.)
