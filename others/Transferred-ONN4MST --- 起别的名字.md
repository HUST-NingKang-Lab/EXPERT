# Transferred-ONN4MST --- 起别的名字

## Abstract

- The significant correlation between taxonomic structures and microbiome sources makes possible for accurate microbiome source tracking.
- We have proposed ONN4MST in previews work. It has show highly priori accuracy comparing with other methods. But it still faces several limitations (preference to well-studied microbiome source and weak flexibility) when applied to real world problems.
- Today, relying on advanced transfer learning techniques. We now propose Transferred-ONN4MST.
- It has been a great success on two fronts: Great flexibility like unsupervised learning methods (FEAST, JSD, SourceTracker) and excellent accuracy even higher than ONN4MST. Thus it can provide better solutions under some special circumstances.
- The result showed that it has greatly improved accuracy on small microbiome comparing to ONN4MST. It also has faster speed (50%) and lower memory (50%) usage than ONN4MST.

## Introduction

...

Highlight:

- great flexibility relying on transfer learning
- customizable ontology and phylogenetic tree
- less data needed when retraining
- great efficiency and low memory usage
- higher accuracy on small microbiome

## Materials and Methods

Methods to be compared:

- ONN4MST, FEAST, SourceTracker, JSD

Experiment design:

- 80% data to train feature extractor（combined dataset）
- 20% data to perform K-fold validation for each method.
- also perform experiment for testing $R^2$ of source contribution calculation
- try FEAST dataset
- split dataset into large, middle and small microbiome to test accuracy & $R^2$
- pollution source tracking (customize ontology)

## Results and Discussion

- Figure 1：Boxplot for each metric of methods (K-fold -> classification）
- Figure 2: $R^2$ plot for source contribution calculation of methods (K-fold -> classification and contribution calculation)
- Figure 3: Variances of metrics and $R^2$ for methods across datasets(combined, 1e4, 1e3, 1e2, 1e1 FEAST dataset) (**Customize ontology case 1**)
- Figure 4: Error analysis and label-based evaluation (auROC and auPRC) for deep NN models.
- Figure 5: Pollution source tracking (Customize ontology case 2)
- **More cases for customizing ontology.**

## Conclusion

- Problems:
	- Hard to define human-level performance, thus hard to estimate Bayes error rate and avoidable bias, prevent us to seeking further.
	- 

## Code Availability

## Acknowledgements

## References