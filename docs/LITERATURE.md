# Literature notes

These notes distinguish published findings from this project's design choices. They are an initial source-backed survey, not a claim that every team member has completed the assignment's two-paper review requirement.

| Study | Evidence consulted | Relevant finding | Application here |
|---|---|---|---|
| I. C. Obasi and C. Benson, “Evaluating the effectiveness of machine learning techniques in forecasting the severity of traffic accidents,” Heliyon 9(8), e18812, 2023. [DOI](https://doi.org/10.1016/j.heliyon.2023.e18812) | Published abstract via [PubMed](https://pubmed.ncbi.nlm.nih.gov/37560691/) | Compares logistic regression, Random Forest, Naive Bayes and neural networks on UK crashes from 2005–2014; reports 87% accuracy for LR and RF. | Supports comparing a linear model with a forest. Its accuracy cannot be directly compared with our different years, features, population and split. We add macro F1 and fatal precision/recall. |
| “Accident severity prediction modeling for road safety using random forest algorithm: an analysis of Indian highways.” [Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10787871/) | Indexed article abstract; full-text retrieval was blocked during preparation | Reports 67% training accuracy and 41.47% test accuracy, with possible bias or imbalance identified in its conclusion. | Motivates reporting held-out performance and minority-class metrics rather than training accuracy. It does not establish transferability of a UK model to India. |

## Questions for each member's full-paper review

Record the unit of analysis, dataset period, labels and class balance; establish which fields are available at prediction time; check whether preprocessing and resampling happen within training folds; identify the split and tuning protocol; record class-level metrics and any calibration or external validation. Explain one design change the evidence motivates, and one limitation preventing direct comparison.

## Project contribution

This project tests an explicitly defined AADF proxy on exactly the same matched test crashes with and without that feature. It separates raw model scores from weighted decisions and reports the trade-off between overall classification and fatal-case detection. Its contribution is a reproducible comparison; it does not claim a new classifier or an improvement over the cited studies.
