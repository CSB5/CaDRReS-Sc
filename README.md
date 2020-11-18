# CaDRReS-Sc
---

## Introduction

CaDRReS-Sc is a framework that robustly combines single-cell RNA-sequencing (scRNA-seq) data with a recommender system to predict drug response heterogeneity in a tumor. It is extended from our an existing recommender system trained with cancer cell line data *(Suphavilai et al., 2018)*. A novel objective function enables the model to simultaneously classify sensitive/resistant cell types and predict half-maximal inhibitory concentration (IC50) values for sensitive cases. Comparison of predictive accuracy versus a naïve objective function (mean squared error for IC50) in [CaDRReS](https://github.com/CSB5/CaDRReS/) on unseen cell lines showed significant improvements, especially for drugs with a smaller proportion of sensitive cell lines. These new features improve robustness on diverse, unseen cell types, and the ability to combine cell-specific predictions into accurate tumor response values.

CaDRReS-Sc can predict drug response based on gene expression profiles of various types of samples, including single-cells, cell clusters, cell lines, and patient tumors. To obtain overall drug response of a given 'heterogenous' sample (such as tumor response given predictions of multiple cell clusters), CaDRReS-Sc relies on [existing single-cell clustering tools](https://github.com/theislab/scanpy) to identify transcriptomically distinct 'clones', as well as a clonal proportion within each heterogeneous sample. With this additional information, CaDRReS-Sc can estimate the overall response of individual drug as well as combinatorial drugs.

## Usage

```bash
git clone https://github.com/CSB5/CaDRReS-Sc.git
```

CaDRReS-Sc is based on Python 3.x

**Require packages**

- Pandas
- Numpy
- TensorFlow 1.14
- [Optional package for single-cell clustering] [Scanpy](https://github.com/theislab/scanpy)

**Example usages** can be found in Jupyter [notebook](https://github.com/CSB5/CaDRReS-Sc/tree/master/notebook).

## Notes

This repository is a new implementation of a recommender system for predicting moto- and combinatorial therapy response based on single-cell RNA-Seq data. The model relies on a novel objective function, resulting in a more robust prediction on unseen samples. CaDRReS-Sc is based on *TensorFlow v1.14*, allowing users to select optimizer, alternative objective functions, and reduce model training time. 

Our CaDRReS-Sc manuscript is under review.

## Contact information

For additional information, help and bug reports please send an email to Chayaporn Suphavilai (suphavilaic@gis.a-star.edu.sg)

