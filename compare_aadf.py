"""Match STATS19 crashes to DfT road links and measure the effect of adding AADF."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import shapefile
from shapely import STRtree, distance, points
from shapely.geometry import shape
from sklearn.metrics import average_precision_score

from train_random_forest import ID, LABELS, NUMERIC, PREFIX, TARGET, build_forest, evaluate, plt


TRAFFIC = "aadf_motor_vehicles_per_day"
POINT_ID = "count_point_id"
MAX_DISTANCE_M = 50
MIN_GAP_M = 10


def match_crashes(source, geometry):
    traffic = pd.read_csv(source / "dft_traffic_counts_aadf.zip", usecols=[
        POINT_ID, "year", "all_motor_vehicles", "estimation_method",
    ]).query("year == 2025").set_index(POINT_ID)
    traffic = traffic.rename(columns={"all_motor_vehicles": TRAFFIC})
    points_catalog = pd.read_csv(source / "count_points.zip", usecols=[POINT_ID])
    if not traffic.index.is_unique or not traffic.index.isin(points_catalog[POINT_ID]).all():
        raise ValueError("2025 traffic figures must have unique IDs in the count point catalog")
    if traffic[TRAFFIC].isna().any() or traffic[TRAFFIC].lt(0).any():
        raise ValueError("Missing or negative traffic volumes")

    coordinates = ["location_easting_osgr", "location_northing_osgr"]
    crashes = pd.read_csv(source / f"{PREFIX}-collision-2025.csv", usecols=[
        ID, "first_road_class", "first_road_number", *coordinates,
    ], dtype={ID: str})
    number = crashes["first_road_number"].astype(str)
    road_class = crashes["first_road_class"]
    crashes["road"] = ("A" + number).where(road_class == 3)
    crashes.loc[road_class == 1, "road"] = "M" + number[road_class == 1]
    crashes.loc[road_class == 2, "road"] = "A" + number[road_class == 2] + "(M)"
    eligible = crashes.loc[crashes["first_road_number"].gt(0)].dropna(
        subset=["road", *coordinates]
    )

    with ZipFile(geometry) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".shp"))
    roads = defaultdict(list)
    with shapefile.Reader(f"{geometry}/{member}") as reader:
        for record in reader.iterShapeRecords():
            attributes = record.record.as_dict()
            cp, road = attributes["CP_Number"], attributes["RoadNumber"]
            if cp in traffic.index and road and record.shape.points:
                key = road.upper().replace(" ", "")
                roads[key].append((int(cp), shape(record.shape.__geo_interface__)))

    matches = []
    for road, crashes_on_road in eligible.groupby("road"):
        if road not in roads:
            continue
        count_ids, segments = zip(*roads[road])
        crash_points = points(crashes_on_road[coordinates].to_numpy())
        tree = STRtree(segments)
        # Search beyond the cutoff so a competing link outside 50 m can still
        # make a match ambiguous.
        crash_pos, segment_pos = tree.query(
            crash_points, predicate="dwithin", distance=MAX_DISTANCE_M + MIN_GAP_M
        )
        if not len(crash_pos):
            continue
        pairs = pd.DataFrame({
            ID: crashes_on_road[ID].to_numpy()[crash_pos],
            POINT_ID: np.asarray(count_ids)[segment_pos],
            "distance_m": distance(crash_points[crash_pos], tree.geometries[segment_pos]),
        }).sort_values("distance_m").drop_duplicates([ID, POINT_ID])
        rank = pairs.groupby(ID).cumcount()
        nearest = pairs.loc[rank == 0].set_index(ID)
        runner_up = pairs.loc[rank == 1].set_index(ID)["distance_m"]
        gap = runner_up.reindex(nearest.index, fill_value=np.inf) - nearest["distance_m"]
        matches.append(nearest.loc[(nearest["distance_m"] <= MAX_DISTANCE_M) & (gap > MIN_GAP_M)])
    if not matches:
        raise ValueError('No crashes matched eligible road links under the distance rules')
    result = pd.concat(matches).reset_index().join(traffic, on=POINT_ID)
    if not result[ID].is_unique or result[TRAFFIC].isna().any():
        raise ValueError("Expected exactly one valid traffic record for each matched crash")
    return result


def fit_and_score(data, features, numeric, settings, model_path):
    development = data.loc[data["split"] != "test"]
    test = data.loc[data["split"] == "test"]
    forest = build_forest(
        features, settings["depth"], settings["training_class_weight"], jobs=4, numeric=numeric
    )
    forest.fit(development[features], development[TARGET])
    forest.set_params(model__n_jobs=1)
    probabilities = forest.predict_proba(test[features])
    classes = forest.classes_.tolist()
    weights = np.array([settings["decision_weights"][label] for label in classes])
    scores = probabilities * weights
    predicted = forest.classes_[scores.argmax(axis=1)]
    result = evaluate(test[TARGET], predicted)
    # Score fatal ranking independently of the final three-class decision rule.
    result["fatal_average_precision"] = average_precision_score(
        test[TARGET].eq("Fatal"), probabilities[:, classes.index("Fatal")]
    )
    joblib.dump({
        "pipeline": forest, "features": features, "decision_weights": settings["decision_weights"],
    }, model_path, compress=3)
    saved = joblib.load(model_path)
    reloaded = saved["pipeline"]
    np.testing.assert_array_equal(
        predicted,
        reloaded.classes_[(reloaded.predict_proba(test[saved["features"]]) * weights).argmax(axis=1)],
    )
    return result


def plot_comparison(results, destination):
    names = ["Accuracy", "Macro F1", "Fatal precision", "Fatal recall", "Fatal avg. precision"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
    positions = np.arange(len(names))
    for index, (key, label, color) in enumerate([
        ("without_aadf", "Without AADF", "#355777"),
        ("with_aadf", "With AADF", "#14816e"),
    ]):
        metrics = results[key]
        fatal = metrics["per_class"]["Fatal"]
        values = [metrics["accuracy"], metrics["macro_f1"], fatal["precision"],
                  fatal["recall"], metrics["fatal_average_precision"]]
        bars = ax.barh(positions + (index - 0.5) * 0.34, values, height=0.3, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=4, fontsize=9)
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_xlim(0, 0.9)
    ax.set_title("Does average daily traffic volume improve crash severity prediction?",
                 loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("Score on the same matched test crashes")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True, help="DfT mrdb-2025.zip")
    parser.add_argument("--output-dir", type=Path, default=root / "aadf_experiment")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(root / "team_crashes.csv.gz", dtype={ID: str}, float_precision="round_trip")
    features = json.loads((root / "metrics.json").read_text(encoding="utf-8"))["features"]
    settings = json.loads((root / "tuned_metrics.json").read_text(encoding="utf-8"))["selected"]
    matches = match_crashes(args.data_dir, args.geometry)
    matched = data.merge(matches[[ID, TRAFFIC]], on=ID, validate="one_to_one")
    if len(matched) != len(matches):
        raise ValueError("Match IDs must all exist in the shared crash dataset")
    matched.to_csv(args.output_dir / "matched_crashes.csv.gz", index=False)
    results = {
        "classes": LABELS,
        "match_rules": {"max_distance_m": MAX_DISTANCE_M, "minimum_gap_m_exclusive": MIN_GAP_M},
        "source_crashes": len(data),
        "matched_crashes": len(matched),
        "matched_links": int(matches[POINT_ID].nunique()),
        "estimate_method_counts": matches["estimation_method"].value_counts().to_dict(),
        "split_counts": pd.crosstab(matched["split"], matched[TARGET]).to_dict(orient="index"),
        "locked_settings": settings,
    }
    for key, model_features, numeric in [
        ("without_aadf", features, NUMERIC),
        ("with_aadf", [*features, TRAFFIC], [*NUMERIC, TRAFFIC]),
    ]:
        results[key] = fit_and_score(
            matched, model_features, numeric, settings, args.output_dir / f"{key}.joblib"
        )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(results, args.output_dir / "comparison.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
