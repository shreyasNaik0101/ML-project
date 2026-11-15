# Literature review

Eight research papers are summarized below. Reviews identify what each source supports, how it informs this implementation and where comparisons stop being valid. Full-text access and abstract-only access are distinguished. No individual team member's authorship or reading activity is asserted.

## 1 Comparative model families

**M.-M. Chen and M.-C. Chen.** “Modeling Road Accident Severity with Comparisons of Logistic Regression, Decision Tree and Random Forest.” *Information* 11(5), 270, 2020. [Publisher full text](https://www.mdpi.com/2078-2489/11/5/270), DOI 10.3390/info11050270.

The study compares LR, CART and RF, combining model assessment with predictor selection. Its outcome separates serious and minor accidents and its data come from Taiwan highways. Accuracy, sensitivity and specificity expose different aspects of performance. This supports our choice of complementary model families and class-specific evaluation. Our three-class Great Britain target differs; rankings and score magnitudes cannot be transferred directly. Variable importance or association also does not establish an intervention's causal effect. **Evidence:** publisher full text, abstract and methods.

## 2 Imbalance and undersampling

**N. Fiorentini and M. Losa.** “Handling Imbalanced Data in Road Crash Severity Prediction by Machine Learning Algorithms.” *Infrastructures* 5(7), 61, 2020. [Publisher full text](https://www.mdpi.com/2412-3811/5/7/61), DOI 10.3390/infrastructures5070061.

The paper compares ordinary and majority-undersampled training for four classifiers. It evaluates minority detection using recall, precision, F1 and confusion matrices alongside accuracy. The methodological lesson is that a majority-dominated accuracy score can conceal poor severe-crash detection. We implement class weighting and validation-selected decision weights rather than its resampling method. A future resampling experiment must fit the sampler inside development folds and preserve an untouched test distribution. Its results do not prove that our weights are optimal. **Evidence:** publisher full text, abstract and methods.

## 3 Ordinal outcomes

**S. Kim, Y. Lym and K.-J. Kim.** “Developing Crash Severity Model Handling Class Imbalance and Implementing Ordered Nature: Focusing on Elderly Drivers.” *International Journal of Environmental Research and Public Health* 18(4), 1966, 2021. [Publisher full text](https://www.mdpi.com/1660-4601/18/4/1966), DOI 10.3390/ijerph18041966.

The study considers ordered and unordered LR/RF variants, with and without imbalance handling, for Ohio crashes involving older drivers. Balancing helps severe-outcome prediction, while incorporating order has model-dependent effects. Severity labels have an ordering, so nominal multiclass classification is a design choice, not the only available formulation. Our current implementation does not include an ordinal classifier. An extension should compare ordinal models on the same splits and evaluate severe-underprediction costs. The population-specific findings cannot establish performance on all road users. **Evidence:** publisher full text and abstract.

## 4 Earlier UK comparison

**I. C. Obasi and C. Benson.** “Evaluating the effectiveness of machine learning techniques in forecasting the severity of traffic accidents.” *Heliyon* 9(8), e18812, 2023. [Published abstract](https://pubmed.ncbi.nlm.nih.gov/37560691/), DOI 10.1016/j.heliyon.2023.e18812.

The authors compare LR, RF, Naive Bayes and neural networks using UK crash data from 2005–2014. LR and RF both achieve the abstract's reported 87% accuracy. This supports keeping simpler reference models instead of assuming complexity guarantees better prediction. That figure is not a benchmark our models can be judged against directly: years, input fields, class distributions and evaluation procedures differ. We report macro F1 and fatal-case counts explicitly. **Evidence:** published abstract; full-text methods were not independently audited for this review.

## 5 Small samples and preprocessing order

**P.-F. Kuo, W.-T. Hsu, D. Lord and I G. B. Putra.** “Classification of autonomous vehicle crash severity: Solving the problems of imbalanced datasets and small sample size.” *Accident Analysis & Prevention* 205, 107666, 2024. [Published abstract](https://pubmed.ncbi.nlm.nih.gov/38901160/), DOI 10.1016/j.aap.2024.107666.

The study uses 266 California autonomous-vehicle reports, including 51 injury cases, and external environmental features. It examines resampling and feature-selection order. The abstract reports the strongest configuration with SMOTE and balancing first. This motivates explicitly documenting pipeline order and sample size. Our study uses a much larger, different population and does not implement this resampling procedure. Any future feature selection must remain within training folds. Some listed predictors may be available only after the crash, so a use-case audit is essential. **Evidence:** published abstract.

## 6 Decision thresholds and balancing

“Machine learning approaches to traffic accident severity prediction: Addressing class imbalance.” *Machine Learning with Applications* 22, 100792, 2025. [Publisher record](https://www.sciencedirect.com/science/article/pii/S2666827025001756), DOI 10.1016/j.mlwa.2025.100792.

The study examines Australian pedestrian injury data and compares LR, SVM, decision tree and XGBoost with resampling and decision-threshold methods. Its relevance is the distinction between changing the training distribution and changing decisions after fitting. Our class multipliers likewise change decisions without calibrating the raw scores. The publisher's accessible abstract and indexed results do not support treating every reported aggregate as directly comparable, and the abstract and indexed detailed results need reconciliation before citing a single best-model claim. We therefore use the methodological observation only. **Evidence:** publisher abstract and indexed methods/results excerpts; direct full-text retrieval was blocked.

## 7 Indian highway generalization

**H. Khanum, A. Garg and M. I. Faheem.** “Accident severity prediction modeling for road safety using random forest algorithm: an analysis of Indian highways.” *F1000Research* 12, 494, version 2, 2023. [Publisher version](https://f1000research.com/articles/12-494/v2), DOI 10.12688/f1000research.133594.2.

The study applies RF to Indian highway severity prediction. Indexed findings report a substantial training-to-test accuracy gap, motivating careful held-out evaluation. This provides local-domain context but does not validate deploying a Great Britain model in India. The project should collect appropriately licensed Indian data and reserve an external evaluation population before making that claim. The versioned article and its review record should be checked when preparing a formal comparison. **Evidence:** publisher record and indexed article abstract; not a full independent reanalysis.

## 8 Forest optimization and interpretation

**X. Wang, Y. Su, Z. Zheng and L. Xu.** “Prediction and interpretive of motor vehicle traffic crashes severity based on random forest optimized by meta-heuristic algorithm.” *Heliyon* 10(16), e35595, 2024. [Published record](https://pubmed.ncbi.nlm.nih.gov/39224374/), DOI 10.1016/j.heliyon.2024.e35595.

The paper studies an optimized RF severity classifier with interpretation as part of its stated scope. It motivates treating optimization and explanation as separate steps that require their own validation. Our implementation uses a transparent, finite validation search and does not reproduce its optimizer. Feature rankings cannot be equated with causal safety effects or universal rankings across countries. No numerical superiority claim from this paper is used in our manuscript. **Evidence:** published title, metadata and indexed topic information; detailed experimental claims are deliberately not inferred.

## Synthesis

The sources motivate three implementation decisions: compare a linear baseline with trees, report rare-class behavior separately, and document preprocessing and decision rules. The project's additional experiment keeps count-point links disjoint, while the AADF ablation holds test rows fixed within each protocol. Model ranking, transferability and traffic-feature value remain empirical questions.

## Member handoff

These eight notes can support up to four pairs of papers. Assign actual members after confirming the team roster, and have each member read the relevant full papers and approve their own review. This file does not certify reading activity or allocate credit without the team's confirmation.
