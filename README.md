# Introduction

CaDRReS-Sc is an AI-environment system for robust cancer drug response prediction based on single-cell RNA-sequencing (scRNA-seq). It is a reimplementation of our existing recommender system ([CaDRReS](https://github.com/CSB5/CaDRReS), [Suphavilai et al., 2018](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663)) with new features calibrated for diverse cell types, accounting for intra-tumor transcriptomic heterogeneity (Suphavilai et al., 2020). In addition to monotherapy response, CaDRReS-Sc can also predict response to combinatorial drugs.

**Key features:**

- Predict monotherapy and combinatorial drugs response in the presence of transcriptomic heterogeneity based on scRNA-seq and predefined single-cell clustering from [Scanpy](https://github.com/theislab/scanpy). (See [tutorial notebook](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/predicting_monotherapy_combinatorial_drugs_scrna-seq.ipynb))
- Provide flexibility to train a new model from other drug response dataset. (See [a model training tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/model_training.ipynb))
- Predict half-maximal inhibitory concentration (IC50) based on gene expression profiles and maximum drug dosage using a pre-trained model, which is trained from a public cancer drug response dataset ([GDSC](https://www.cancerrxgene.org/celllines)). (See [a tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/prediction_pretrained_model.ipynb) on predicting drug response)
- Robust to diverse gene expression platforms (such as microarray and RNA-seq) and provide an interpretable latent pharmacogenomic space ([Suphavilai et al., 2018](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663), Suphavilai et al., 2020).

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

Suphavilai, C., et al. Predicting heterogeneity in clone-specific therapeutic vulnerabilities using single-cell transcriptomic signatures. 2020 November. (bioRxiv; Manuscript submitted)

Alternatively, you can cite our CaDRReS paper for the concepts of kernel features and latent pharmacogenomic space:

Suphavilai C, Bertrand D, Nagarajan N. Predicting cancer drug response using a recommender system. [Bioinformatics](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663). 2018 Nov 15;34(22):3907-14.

# Contact information

For additional information, help and bug reports please send an email to Chayaporn Suphavilai ([suphavilaic@gis.a-star.edu.sg](mailto:suphavilaic@gis.a-star.edu.sg))
