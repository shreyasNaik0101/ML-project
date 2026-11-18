# Data provenance and preparation

`final_matched_2021_2025.csv.gz` is the prepared project dataset containing one row per eligible collision. It includes metadata and the feature representation derived from the seven input groups. Use `collision_index`, `year`, `count_point_id` and `severity` as metadata or target, not as predictors.

The `raw/` folder contains compressed public source files for collision and vehicle records from 2021-2025, annual average daily flow counts, and yearly major-road geometry archives. Collision and vehicle records are published by the UK Department for Transport as [road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data); annual traffic data and road geometry are from [road traffic statistics downloads](https://roadtraffic.dft.gov.uk/downloads). Consult the provider documentation and applicable Open Government Licence terms when reusing these files.

The original project joined vehicles to collisions by `collision_index`, checked identifiers and labels, matched road numbers and geometry to an eligible traffic count point within 50 metres with an ambiguity check, and used the previous year's AADF. It encoded time cyclically and aggregated vehicle-type indicators and counts. For the reported experiment, the prepared dataset includes all eligible matched 2021 records alongside 2022-2025 records.

The notebook shows examples of the original tables and join, checks and examines the prepared dataset, and fits model preprocessing in the model notebooks. It does **not** rebuild the full historical geospatial match. Previous scripts, detailed audits and intermediate experiments are preserved on the `experiment/traffic-sensor-multiyear` branch. Exact source releases, matching decisions and reporting changes can affect replication.

The selected major-road cohort does not cover every road or all collisions. AADF measures typical daily flow, not instantaneous density. Vehicle composition may only be available after the event. The 2025 cohort was previously inspected and is not a fresh external test.
