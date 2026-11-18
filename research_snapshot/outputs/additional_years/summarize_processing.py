"""Summarize the completed data audit and controlled processing experiments."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent / 'data_processing'
audit=json.loads((HERE/'audit.json').read_text())
results=json.loads((HERE/'results.json').read_text())
old=results['baseline']['test']
new=results['vehicle_composition_optional']['test']
labels=['Slight','Serious','Fatal']
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
x=np.arange(3)
for i,(name,record,color) in enumerate([('Previous processing',old,'#355777'),('Vehicle composition + time',new,'#16816e')]):
    values=[record['report'][c]['recall']*100 for c in labels]
    bars=axes[0].bar(x+(i-.5)*.34,values,width=.3,label=name,color=color)
    axes[0].bar_label(bars,fmt='%.1f%%',padding=3,fontsize=8)
axes[0].set_xticks(x,labels)
axes[0].set_ylim(0,100)
axes[0].set_ylabel('Actual cases correctly identified (%)')
axes[0].set_title('Recall on the same 2025 crashes')
axes[0].legend(fontsize=7,frameon=False)
for label in labels:
    code={'Fatal':'1','Serious':'2','Slight':'3'}[label]
    values=[audit[str(y)]['matched_class_counts'][label]/audit[str(y)]['source_class_counts'][code]*100 for y in range(2022,2026)]
    axes[1].plot(range(2022,2026),values,marker='o',label=label)
axes[1].set_xticks(range(2022,2026))
axes[1].set_title('Retention after traffic matching')
axes[1].set_ylabel('Source crashes retained (%)')
axes[1].legend(frameon=False)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.savefig(HERE/'processing_comparison.png',dpi=180)
plt.close(fig)

rows=[]
for y,a in audit.items():
    rows.append(f"| {y} | {a['source_rows']:,} | {a['source_vehicle_rows']:,} | {a['matched_rows']:,} | {a['raw_field_counts']['collision_injury_based']['1']/a['source_rows']:.1%} |")
text='''# Data processing review

The audit covered all 2022-2025 source collisions and vehicles, and tested changes using 2022-2023 training and 2024 validation. Model complexity was kept fixed at the previous histogram-gradient-boosting settings. The selected data representation was refitted on 2022-2024 and evaluated on the previously inspected 2025 cohort.

## Raw data integrity

| Year | Crashes | Vehicle records | Retained by traffic match | Injury-based reporting |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

All four years have unique crash IDs and unique crash/vehicle keys. Every vehicle row joins to a crash. Every crash has vehicle records, and the recorded vehicle totals agree with those records. Similar input values across different crashes are legitimate repeated observations and were retained.

## Cleaning and coding

The supplied files already use a common published schema. Allowed-value checks on road type, weather and surface found no unexpected codes in the matched cohort. All matched clock times parsed correctly and all matched AADF values were positive. Each cached traffic value was independently checked against its count-point ID and previous year's raw traffic table. Every retained severity label was checked against the raw collision severity.

Explicit unknown values (such as weather code 9) were kept distinct from invalid or missing values. Missing road surface codes remain present: the full source files contain 694, 1,064, 704 and 634 such records from 2022 through 2025. Unknown weather and surface also occur and were preserved. Rechecking categories produced the same validation score as the baseline, so this audit provides assurance rather than an artificial accuracy increase.

## Changes tested

1. Validated categorical codes, speed limits and raw clock reconstruction.
2. Two extra sine/cosine harmonics derived from the existing time-of-day input.
3. Vehicle composition counts for cycles, motorcycles, cars/taxis, buses, goods vehicles and other vehicles, alongside the existing type-presence flags and time features. These counts distinguish two cars from one car. They require the complete vehicle-type list for each crash and expand the representation of the vehicle-type input. This optional representation should be documented if the assignment expects a single vehicle-type selection.

Validation macro F1 was 0.4238 for the baseline and validated categories, 0.4203 for the extra time features, and 0.4330 for vehicle composition plus time. Thus the composition variant was selected on validation, before this run evaluated 2025. The experiment combines composition and extra time features, so its gain cannot be assigned exclusively to composition. The strict baseline remains the selected variant among those without vehicle counts.

| Same 2025 cohort | Previous processing | Composition plus time |
|---|---:|---:|
| Accuracy | 69.38% | 67.00% |
| Macro F1 | 0.4232 | 0.4284 |
| Slight recall | 80.30% (11,755/14,639) | 74.36% (10,886/14,639) |
| Serious recall | 34.08% (1,438/4,219) | 43.94% (1,854/4,219) |
| Serious precision | 34.03% | 33.47% |
| Serious F1 | 0.3406 | 0.3800 |
| Fatal recall | 15.69% (32/204) | 15.69% (32/204) |

The optional representation identified 416 more Serious crashes while missing 869 more Slight crashes. It improved class-balanced F1 modestly and reduced overall accuracy. It did not improve every class.

## Data limitations exposed

- The source spans 412,714 crashes, but traffic matching retains 79,100. This is a selected major-road cohort. The plot breaks retention down by year and severity. Increasing the radius without a linkage-quality study could introduce incorrect traffic values.
- Injury-based reporting grows from 52.6% in 2022 to 86.5% in 2025. DfT explains that reporting systems can change the recorded Slight/Serious balance. The reporting indicator and adjusted severity fields were used for audit only. Labels were preserved; statistical adjustment estimates were not converted into invented individual labels.
- Published vehicle category 33 already appears in all four supplied files. Older files therefore should not be recoded just because a category was introduced in the later specification. The audit also records e-scooter flags separately; category 33 includes more than e-scooters.
- Matching rules and the numeric AADF join were checked in code, but road geometry has not been independently reviewed on a map. Junction and parallel-road ambiguity remains a source of potential error.
- The cached feature schema includes vehicle categories observed in 2025. No 2025 labels were used to learn preprocessing, weights or the selected representation in this experiment. The 2025 cohort was previously inspected in project development, and road links can recur across years. A fresh external evaluation remains needed.
- Only the seven input groups were modelled. Reporting-system identifiers, location, crash severity adjustments and casualty counts were excluded from predictors. AADF remains annual traffic volume, used as a density proxy.

## Files and verification

- audit.json records source hashes, counts, class coverage, missing/unknown distributions, coding checks and vehicle joins.
- processed_crashes.csv.gz contains all tested representations plus metadata. Use only the chosen bundle's features for modelling. Year, crash IDs, road-link IDs and severity are metadata or target.
- results.json records validation reports and held-out reports for the baseline and selected variant.
- vehicle_composition_optional.joblib contains the selected pipeline, ordered feature list and decision weights. The pipeline learns numeric imputation from development rows. Reloaded predictions reproduced the recorded test results.
- processing_comparison.png plots recall changes and matching retention.

Use the audit_and_process.py script in the parent folder to repeat the analysis with the existing workspace source files and sklearn 1.9.1. Bundles predict using the pipeline's scores multiplied by decision_weights in pipeline.classes_ order. Scores are uncalibrated.

Sources: [DfT source data and codebook](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data), [effects of the STATS19 specification transition](https://www.gov.uk/government/publications/stats19-collection-and-use-impacts-of-the-2024-specification/stats19-collection-and-use-initial-impacts-of-transition-to-the-2024-specification), and [road traffic data](https://roadtraffic.dft.gov.uk/downloads).
'''
(HERE/'README.md').write_text(text,encoding='utf-8')
