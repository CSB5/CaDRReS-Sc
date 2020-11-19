# Introduction

CaDRReS-Sc is a recommender system framework for drug response prediction based on single-cell RNA-sequencing (scRNA-seq) data. It takes into account intra-tumor transcriptomic heterogeneity to predict drug response heterogeneity within a tumor, as well as overall responses to monotherapy and combinatorial drugs. 

**Key features:**

- Train CaDRReS-Sc model using     a publicly available drug response dataset such as GDSC ([Tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebook/notebook_01_model_training.ipynb))
- Predict drug response of     unseen samples based on gene expression profile using CaDRReS-Sc model ([Tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebook/notebook_02_prediction.ipynb))
- Predict drug response in the     presence of intra-tumor transcriptomic heterogeneity based on scRNA-seq     data. In this [tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebook/notebook_03_monotherapy_and_combinatorial_drugs.ipynb), we explain how to apply the     pre-trained CaDRReS-Sc model to predict heterogeneous drug response within     each patient/tumor. Briefly, user can apply an [existing      single-cell clustering tool](https://github.com/theislab/scanpy) to identify transcriptomically     distinct *clusters* or *clones*, calculate cluster-specific gene     expression profile, and obtain cluster proportions within each *heterogeneous     sample* (patient/tumor). Based on this information, user apply     CaDRReS-Sc model to predict overall monotherapy and combinatorial drug     response for each patient.
- A novel objective function     enables CaDRReS-Sc to simultaneously classify sensitive/resistant cell     types and predict half-maximal inhibitory concentration (IC50) values for     sensitive cases. *Please refer to our manuscript for more detail about     the novel objective function.*
- CaDRReS-Sc relies on *kernel     features* which capture similarity of the gene expression profile,     rather than directly using gene expression values *(Suphavilai et al.,     2018)*. This feature allows the model to work across gene expression     platforms; for example, the model trained on microarray data can be used     for predicting drug response based on RNA-Seq data.

# Usage

git clone https://github.com/CSB5/CaDRReS-Sc.git

CaDRReS-Sc is based on Python 3.x

**Require packages**

- Pandas
- Numpy
- TensorFlow 1.14
- [Optional package for     single-cell clustering] [Scanpy](https://github.com/theislab/scanpy)

**Example usages** can be found in Jupyter [notebook](https://github.com/CSB5/CaDRReS-Sc/tree/master/notebook).

# Notes

This repository is a new implementation of a recommender system for predicting moto- and combinatorial therapy response based on scRNA-Seq data. The model relies on a novel objective function, resulting in a more robust prediction on unseen samples. CaDRReS-Sc is based on *TensorFlow v1.14*, allowing users to select optimizer, alternative objective functions, and reduce model training time. 

# Contact information

For additional information, help and bug reports please send an email to Chayaporn Suphavilai ([suphavilaic@gis.a-star.edu.sg](mailto:suphavilaic@gis.a-star.edu.sg))
