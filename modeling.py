"""Shared validation-only model selection for the matched crash experiment."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LABELS = ["Slight", "Serious", "Fatal"]
TARGET = "severity"
AADF = "aadf_motor_vehicles_per_day"
CATEGORICAL = ["road_type", "weather_conditions", "road_surface_conditions"]
BASE_NUMERIC = ["speed_limit", "time_sin", "time_cos"]

def make_pipeline(features, complexity, class_weight, family="logistic_regression"):
    numeric = BASE_NUMERIC + ([AADF] if AADF in features else [])
    vehicles = [column for column in features if column not in CATEGORICAL + numeric]
    categories = Pipeline([
        ("missing", SimpleImputer(strategy="constant", fill_value=-1)),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numbers = Pipeline([
        ("missing", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("categorical", categories, CATEGORICAL),
        ("numeric", numbers, numeric),
        ("vehicle_flags", "passthrough", vehicles),
    ])
    classifier = (LogisticRegression(C=complexity, class_weight=class_weight, max_iter=2000)
                  if family == 'logistic_regression' else
                  DecisionTreeClassifier(max_depth=complexity, min_samples_leaf=10,
                                         class_weight=class_weight, random_state=42))
    return Pipeline([("preprocess", preprocessing), ("model", classifier)])
def choose_score_weights(probabilities, actual, classes):
    # Keep the Slight multiplier at 1 while searching the other two.
    y = pd.Categorical(actual, categories=classes).codes
    counts = np.bincount(y, minlength=len(classes))
    serious, fatal = [classes.index(label) for label in ("Serious", "Fatal")]
    multipliers = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    best = None
    for serious_weight in multipliers:
        for fatal_weight in multipliers:
            weights = np.ones(len(classes))
            weights[serious], weights[fatal] = serious_weight, fatal_weight
            prediction = (probabilities * weights).argmax(axis=1)
            predicted_counts = np.bincount(prediction, minlength=len(classes))
            correct_counts = np.bincount(y[y == prediction], minlength=len(classes))
            macro_f1 = (2 * correct_counts / (counts + predicted_counts)).mean()
            accuracy = correct_counts.sum() / len(y)
            score = (float(macro_f1), float(accuracy))
            if best is None or score > best[0]:
                best = score, dict(zip(classes, weights.tolist()))
    return best


