"""Translate a readable crash scenario into the model's prepared feature schema."""
import argparse
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from predict import predict

ROAD = {'roundabout': 1, 'one_way': 2, 'dual_carriageway': 3, 'single_carriageway': 6, 'slip_road': 7, 'unknown': 9}
WEATHER = {'fine': 1, 'rain': 2, 'snow': 3, 'fine_high_wind': 4, 'rain_high_wind': 5, 'snow_high_wind': 6, 'fog': 7, 'other': 8, 'unknown': 9}
SURFACE = {'dry': 1, 'wet': 2, 'snow': 3, 'ice': 4, 'flood': 5, 'unknown': None}
VEHICLES = {'bicycle': 1, 'motorcycle_50cc': 2, 'motorcycle_125cc': 3, 'motorcycle_500cc': 4, 'motorcycle_over_500cc': 5, 'taxi': 8, 'car': 9, 'minibus': 10, 'bus': 11, 'van': 19, 'unknown': 99}


def prepare(scenario, features):
    if not isinstance(scenario, dict):
        raise ValueError('Scenario must be a JSON object')
    for field, choices in [('road_type', ROAD), ('weather', WEATHER), ('surface', SURFACE)]:
        if not isinstance(scenario.get(field), str) or scenario[field] not in choices:
            raise ValueError(f'{field} must be one of: {", ".join(choices)}')
    row = {f: 0 for f in features}
    try:
        row['road_type'] = ROAD[scenario['road_type']]
        row['weather_conditions'] = WEATHER[scenario['weather']]
        row['road_surface_conditions'] = SURFACE[scenario['surface']]
        if not isinstance(scenario['time'], str) or not re.fullmatch(r'[0-9]{2}:[0-9]{2}', scenario['time']):
            raise ValueError('Time must be a string using HH:MM')
        parts = scenario['time'].split(':')
        if len(parts) != 2:
            raise ValueError('Time must use HH:MM')
        hour, minute = map(int, parts)
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError('Time must use HH:MM with valid hours and minutes')
        speed = scenario['speed_limit_mph']
        if speed not in (20, 30, 40, 50, 60, 70):
            raise ValueError('Speed limit must be 20, 30, 40, 50, 60 or 70 mph')
        row['speed_limit'] = speed
        angle = 2 * np.pi * (hour * 60 + minute) / 1440
        row['time_sin'], row['time_cos'] = np.sin(angle), np.cos(angle)
        vehicles = scenario['vehicles']
        if not isinstance(vehicles, list) or not vehicles:
            raise ValueError('Provide at least one vehicle in a list')
        for vehicle in vehicles:
            if not isinstance(vehicle, str):
                raise ValueError('Each vehicle must be a category name')
            feature = f'vehicle_type_{VEHICLES[vehicle]}'
            if feature not in features:
                raise ValueError(f'Vehicle type absent from trained model: {vehicle}')
            row[feature] = 1
        if 'aadf_motor_vehicles_per_day' in features:
            row['aadf_motor_vehicles_per_day'] = scenario['aadf_motor_vehicles_per_day']
    except KeyError as error:
        raise ValueError(f'Missing field or unsupported category: {error.args[0]}') from error
    return pd.DataFrame([row], columns=features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--options', action='store_true', help='List supported scenario inputs without loading a model')
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'artifacts/logistic_regression_without_aadf.joblib')
    args = parser.parse_args()
    if args.options:
        print(json.dumps({'road_type': list(ROAD), 'weather': list(WEATHER),
                          'surface': list(SURFACE), 'vehicles': list(VEHICLES),
                          'speed_limit_mph': [20, 30, 40, 50, 60, 70],
                          'time': 'HH:MM in 24-hour format',
                          'aadf_motor_vehicles_per_day': 'Nonnegative daily volume, required only by AADF models'}, indent=2))
        parser.exit()
    try:
        bundle = joblib.load(args.model)
        rows = prepare(json.loads(args.input.read_text()), bundle['features'])
        print(predict(bundle, rows).to_json(orient='records', indent=2))
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f'Prediction failed: {error}\n')
