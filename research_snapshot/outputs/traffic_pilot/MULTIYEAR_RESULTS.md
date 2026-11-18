# Multi-year traffic sensor comparison

## Findings

On the same 148 selected 2024 crashes, the sensor-based traffic proxy did not improve this model comparison. These are exploratory development results, not a fresh final test: 2024 has been used previously in this project, and the sample is small and selective.

| Evaluation measure | Previous-year AADF | Sensor traffic proxy |
| --- | ---: | ---: |
| Overall accuracy | 70.9% (105/148) | 70.3% (104/148) |
| Slight recall | 89.3% (100/112) | 90.2% (101/112) |
| Serious recall | 16.7% (5/30) | 10.0% (3/30) |
| Fatal recall | 0% (0/6) | 0% (0/6) |
| Macro F1 | 0.345 | 0.319 |

An always-Slight predictor scores 75.7% accuracy on this evaluation group but catches no Serious or Fatal crashes. Only five Fatal crashes were available for training and six for testing, so the fatal numbers cannot establish reliable performance. These results are not directly comparable with earlier project scores because the cohort and training set are different.

The accuracy difference (sensor minus AADF) is -0.7 percentage points. An exploratory paired row-bootstrap interval is approximately -6.1 to +4.7 percentage points; it does not account for correlations between nearby sites, so it may be too narrow. This experiment does not establish a real difference in accuracy.

## Matching and completeness

Starting from the existing AADF-matched project data, the improved rule looks for a current active WebTRIS sensor on the same numbered motorway within 250 m, whose recorded cardinal direction broadly agrees with all involved vehicles' recorded directions. It also requires the sensor to be within 50 m of the crash's previously matched count-point road geometry for that year. If several sensors qualify, the nearest is selected. These checks yielded 289, 283 and 236 candidates in 2022, 2023 and 2024, respectively.

Matching uses information from the collision's vehicle records, so this is a retrospective selected cohort rather than a method proven usable for live predictions. A compass match and road-link proximity do not prove the correct carriageway, road segment or historical sensor operation. Clockwise/non-cardinal sensors and uncertain vehicle directions were excluded; the current sensor catalogue can omit historically relevant sensors. These restrictions and the original AADF matching select motorway locations and may alter severity prevalence.

Among 808 candidates, 502 had an eligible traffic value, 280 had missing or invalid count/speed values in the selected window, and 26 had at least one missing required sensor-day download with no usable value. The latter includes requests that failed after retries; these categories are not proof of why readings were missing. Usable readings were available for 183 crashes in 2022, 171 in 2023 and 148 in 2024. The collector also fetched previous-day readings for early-morning crashes where necessary.

## Comparison design and limitations

- The training period is 2022–2023; the evaluation period is 2024. Fifty-four training rows with sensor IDs appearing in the test period were excluded, leaving 300 training and 148 evaluation crashes. Nearby different sensors or road links can still be related across periods.
- Both runs fit the same histogram gradient boosting model and the same seven-group feature representation on the same rows, with the default class decision rule. Their only changed input is previous-year AADF versus `4 × 15-minute count / mean mph`, an approximate vehicles-per-mile measure across monitored lanes. It is not a direct density measurement.
- The source-reported interval end was treated as inclusive. To reduce potential leakage from a one-hour difference in clock conventions, the eligible recorded interval must end 75–105 minutes before the recorded crash time. This is conservative only under the assumed maximum one-hour offset. WebTRIS and STATS19 time-zone and daylight-saving conventions still need authoritative verification before claiming the measurement is definitely pre-crash.
- Average speed must be positive and count nonnegative. Detailed interval quality, lane counts, sensor health, and geometry have not been independently validated. Missing readings were excluded rather than filled, and both models use the same resulting cases.
- [WebTRIS states](https://webtris.nationalhighways.co.uk/Home/Faqs) that readings are usually uploaded about a month in arrears, so its historical API alone is not an immediate real-time traffic feed.
- A truly untouched final set remains unavailable in the previously inspected years. [DfT's release announcement](https://www.gov.uk/government/statistics/announcements/reported-road-casualties-great-britain-provisional-results-2026) lists annual provisional 2026 statistics for May 2027; check the actual release of compatible record-level crash, vehicle and traffic data before relying on it. Freeze the whole method and choices before opening a new final test set.

## Artifacts

`multiyear_candidates.py` builds the selected candidate links; `collect_multiyear.py` retrieves sensor days; `compare_multiyear.py` performs the reading audit and paired comparison. Results and confusion matrices are in `multiyear_results.json`, and candidate and usable matches are in the corresponding CSV files. Cached readings and the sensor catalogue remain in `work/traffic_pilot/`; source road geometry and STATS19 files remain in `work/yearly-data/`. The comparison has not created a replacement production model.
