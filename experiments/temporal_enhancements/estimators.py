"""Scikit-learn adapter for CatBoost 1.2.8 (which predates sklearn 1.8 tags)."""
from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, depth=7):
        self.depth = depth

    def fit(self, X, y, sample_weight=None):
        from catboost import CatBoostClassifier
        self.model_ = CatBoostClassifier(iterations=600, learning_rate=.05, l2_leaf_reg=5,
                                        depth=self.depth, loss_function='MultiClass', verbose=False,
                                        random_seed=42, thread_count=2, allow_writing_files=False)
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X).reshape(-1)
