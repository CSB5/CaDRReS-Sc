# Introduction

CaDRReS-Sc is an AI/ML framework for robust cancer drug response prediction based on single-cell RNA-sequencing (scRNA-seq) data. It extends an existing recommender system model ([CaDRReS](https://github.com/CSB5/CaDRReS), [Suphavilai et al., 2018](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663)) with new features calibrated for diverse cell types, and accounting for tumor heterogeneity [Suphavilai et al., 2020](https://www.biorxiv.org/content/10.1101/2020.11.23.389676v1). In addition to monotherapy response, CaDRReS-Sc can also predict response to combinatorial therapy.

**Key features:**

- Predicts cancer drug response from transcriptomic profiles for (a) single-cells, (b) cell clusters (e.g. using [Scanpy](https://github.com/theislab/scanpy), see [tutorial notebook](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/predicting_monotherapy_combinatorial_drugs_scrna-seq.ipynb)), and (c) from bulk analysis.
- Includes a pre-trained model based on the [GDSC](https://www.cancerrxgene.org/celllines) database.  
- Reports half-maximal inhibitory concentration (IC50) values as well as predicted cell death percentage for a specific drug dosage (see [tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/prediction_pretrained_model.ipynb)).
- Robust to gene expression measurements across diverse platforms including microarrays, RNA-seq and scRNA-seq ([Suphavilai et al., 2018](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663), [Suphavilai et al., 2020](https://www.biorxiv.org/content/10.1101/2020.11.23.389676v1)).
- Interpretable model based on a latent "pharmacogenomic space" ([Suphavilai et al., 2018](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663), [Suphavilai et al., 2020](https://www.biorxiv.org/content/10.1101/2020.11.23.389676v1))
- Flexibility to train a new model based on other drug response datasets (see [model training tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/model_training.ipynb)).

# Usage

``git clone https://github.com/CSB5/CaDRReS-Sc.git``

CaDRReS-Sc is based on Python 3.x

**Required packages**

- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [TensorFlow 1.14](https://www.tensorflow.org/install/pip)

**Optional package**

- [Scanpy](https://github.com/theislab/scanpy) (for single-cell clustering)

**Usage examples** can be found in [notebooks](https://github.com/CSB5/CaDRReS-Sc/tree/master/notebooks).

# Citation

Suphavilai, C., et al. Predicting heterogeneity in clone-specific therapeutic vulnerabilities using single-cell transcriptomic signatures. [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.23.389676v1) 2020 Nov 

Suphavilai C, Bertrand D, Nagarajan N. Predicting cancer drug response using a recommender system. [Bioinformatics](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663) 2018 Nov 15;34(22):3907-14.

# Contact information

For additional information, help and bug reports please email Chayaporn Suphavilai ([suphavilaic@gis.a-star.edu.sg](mailto:suphavilaic@gis.a-star.edu.sg))
