# Traffic data investigation

## What was checked

The current traffic feature uses annual average daily flow (AADF). National Highways' WebTRIS API provides historic 15-minute vehicle counts and average speeds at monitoring sites. This could provide a more time-specific estimate of traffic density within the existing traffic-density input group.

An exploratory location match compared 2024 STATS19 collision coordinates to the current catalogue of active motorway-named WebTRIS sites. It selected the nearest sensor with the same road class and number within 250 metres. This produced 1,795 candidate collisions. The matching did not use severity labels. These are candidates, not validated matches: proximity and road number do not prove that the sensor measured the same carriageway or road segment, and today's sensor status does not establish historic availability.

For three candidates, the API returned 96 quarter-hour records for the collision date. The latest reported interval ending at least 15 minutes before the recorded collision time was:

| Sensor | Collision time | Earlier interval ending | Vehicles in interval | Mean speed (mph) |
| --- | --- | --- | ---: | ---: |
| 2209 (M25) | 13 Feb 2024, 13:50 | 13:29 | 1,592 | 55 |
| 3331 (M1) | 14 Dec 2024, 17:39 | 17:14 | 1,204 | 60 |
| 10287 (M25) | 2 Jul 2024, 09:50 | 09:29 | 647 | 51 |

These examples establish data availability only, not a prediction improvement. Confirm the API's interval conventions, sensor validity and carriageway before using a measurement. Average observed speed is not the speed-limit feature. A density approximation could be calculated as hourly flow divided by mean speed, but only where both are valid and the spatial and temporal assumptions hold; units would be vehicles per mile over the observed lanes.

## Next comparison

1. Validate sensor-to-road matching and quantify coverage and missingness by year, road type and severity, using development data only. Check whether enough serious and fatal crashes are represented.
2. On the same matched crashes, compare the existing AADF feature with the pre-crash sensor-based traffic estimate, keeping all other inputs, preprocessing and model settings consistent. Split by time or appropriate road groups and prevent overlapping sensor/time observations across splits. Fit preprocessing only on training folds. Report overall accuracy, confusion matrices, per-class recall and precision, support, and uncertainty, not accuracy alone.
3. Keep the original AADF approach for crashes with no reliable sensor match. Investigate whether differences are caused by the new feature or by restricting evaluation to motorways.
4. Freeze all model and matching choices before a genuinely new final test set becomes available. Previously used 2025 data cannot be presented as untouched if its results have already guided development. A later official year may be suitable when published, after checking coverage and schema.

The seven input groups remain road type, weather, time, traffic density, vehicle type, speed limit and road conditions. Existing crash-condition fields need an audit for unknown values and whether their information would be available at the intended prediction time; this investigation did not establish that additional weather or road-condition data improve the model.

## Sources and reproducibility

- National Highways [WebTRIS API documentation](https://webtris.nationalhighways.co.uk/api/swagger/ui/index).
- National Highways [WebTRIS updates](https://webtris.nationalhighways.co.uk/Home/News), for service availability.
- US FHWA [traffic flow relationships](https://ops.fhwa.dot.gov/publications/fhwahop08054/sect3.htm).
- Exploratory matching script and downloaded API samples: `work/traffic_pilot/`. Source collision data: `work/yearly-data/dft-road-casualty-statistics-collision-2024.csv`.

No model was trained or evaluated as part of this pilot, and no untouched final test set has yet been established.
