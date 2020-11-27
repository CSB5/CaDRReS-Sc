# Introduction

CaDRReS-Sc is an AI/ML framework for robust cancer drug response prediction based on single-cell RNA-sequencing (scRNA-seq) data. It extends an existing recommender system model ([CaDRReS](https://github.com/CSB5/CaDRReS), [Suphavilai et al., 2018](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663)) with new features calibrated for diverse cell types, and accounting for tumor heterogeneity ([Suphavilai et al., 2020](https://www.biorxiv.org/content/10.1101/2020.11.23.389676v1)). In addition to monotherapy response, CaDRReS-Sc can also predict response to combinatorial therapy targeting tumor sub-clones.

**Key features:**

- Predicts cancer drug response from transcriptomic profiles for (a) single-cells, (b) cell clusters (e.g. using [Scanpy](https://github.com/theislab/scanpy), see [tutorial notebook](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/predicting_monotherapy_combinatorial_drugs_scrna-seq.ipynb)), and (c) from bulk analysis.
- Includes a pre-trained model based on the [GDSC](https://www.cancerrxgene.org/celllines) database.  
- Reports half-maximal inhibitory concentrations (IC50) as well as predicted cell death percentages for a specific drug dosage (see [tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/prediction_pretrained_model.ipynb)).
- Robust to gene expression measurements across diverse platforms, including microarrays, RNA-seq and scRNA-seq ([Suphavilai et al., 2018](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663), [Suphavilai et al., 2020](https://www.biorxiv.org/content/10.1101/2020.11.23.389676v1)).
- Interpretable model based on a latent "pharmacogenomic space" ([Suphavilai et al., 2018](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663), [Suphavilai et al., 2020](https://www.biorxiv.org/content/10.1101/2020.11.23.389676v1)).
- Flexibility to train a new model based on other drug response datasets (e.g. [CTRP](https://portals.broadinstitute.org/ctrp/), see [model training tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/model_training.ipynb)).

# Usage

``git clone https://github.com/CSB5/CaDRReS-Sc.git``

CaDRReS-Sc is based on Python 3.x

**Required packages**

- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [TensorFlow 1.14](https://www.tensorflow.org/install/pip)

**Optional package**

- [Scanpy](https://github.com/theislab/scanpy) (for single-cell clustering)

## Usage examples

Please refer to our tutorial [notebooks](https://github.com/CSB5/CaDRReS-Sc/tree/master/notebooks). Below are snippets from the notebooks:

**Model training** ([tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/model_training.ipynb))

```python
model_dict, output_dict = model.train_model_logistic_weight(
    Y_train, X_train,                 # Y = drug response; X = kernel features
    Y_test, X_test, 
    sample_weights_logistic_x0_df,    # Sample weight w.r.t. maximum drug dosage
    indication_weight_df,             # High weight for specific tissue types
    10, 0.0, 100000, 0.01,            # Hyperparamters
    model_spec_name=model_spec_name,  # Select objective function
    save_interval=5000,
    output_dir=output_dir)
```

**Predicting monotherapy response based on a pre-trained model** ([tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/prediction_pretrained_model.ipynb))

```python
cadrres_model = model.load_model(model_file)
pred_df, P_df = model.predict_from_model(cadrres_model, X_test, model_spec_name)
```

```python
pred_df.head() # Predicted drug response (log2 IC50)
```

<table border="0">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D1</th>
      <th>D1001</th>
      <th>D1003</th>
      <th>D1004</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>906826</th>
      <td>3.86</td>
      <td>11.11</td>
      <td>-5.71</td>
      <td>-5.56</td>
      <td>...</td>
    </tr>
    <tr>
      <th>687983</th>
      <td>7.00</td>
      <td>11.52</td>
      <td>-4.12</td>
      <td>-4.48</td>
      <td>...</td>
    </tr>
    <tr>
      <th>910927</th>
      <td>1.74</td>
      <td>10.84</td>
      <td>-7.32</td>
      <td>-6.77</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1240138</th>
      <td>3.55</td>
      <td>11.42</td>
      <td>-4.79</td>
      <td>-4.82</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1240139</th>
      <td>2.79</td>
      <td>10.70</td>
      <td>-7.89</td>
      <td>-7.42</td>
      <td>...</td>
    </tr>
  </tbody>
</table>

```python
P_df.head() # A latent vector of each sample in the 10D pharmacogenomic space
```

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>906826</th>
      <td>0.38</td>
      <td>-1.39</td>
      <td>-1.26</td>
      <td>-0.15</td>
      <td>-0.37</td>
      <td>-1.35</td>
      <td>1.09</td>
      <td>0.04</td>
      <td>-0.78</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>687983</th>
      <td>0.26</td>
      <td>-0.68</td>
      <td>0.47</td>
      <td>1.31</td>
      <td>0.61</td>
      <td>0.93</td>
      <td>-0.09</td>
      <td>-0.77</td>
      <td>-2.20</td>
      <td>-0.42</td>
    </tr>
    <tr>
      <th>910927</th>
      <td>-0.52</td>
      <td>0.47</td>
      <td>0.12</td>
      <td>-0.10</td>
      <td>-1.56</td>
      <td>-2.99</td>
      <td>1.15</td>
      <td>-0.06</td>
      <td>-0.58</td>
      <td>-1.21</td>
    </tr>
    <tr>
      <th>1240138</th>
      <td>0.72</td>
      <td>0.51</td>
      <td>-1.16</td>
      <td>-0.34</td>
      <td>1.56</td>
      <td>-1.21</td>
      <td>1.06</td>
      <td>-0.59</td>
      <td>0.08</td>
      <td>-0.53</td>
    </tr>
    <tr>
      <th>1240139</th>
      <td>-0.08</td>
      <td>-0.45</td>
      <td>0.45</td>
      <td>-0.19</td>
      <td>-0.75</td>
      <td>-2.93</td>
      <td>0.50</td>
      <td>0.67</td>
      <td>-0.11</td>
      <td>0.27</td>
    </tr>
  </tbody>
</table>

**Predicting response to drug combinations at specific dosages** ([tutorial](https://github.com/CSB5/CaDRReS-Sc/blob/master/notebooks/predicting_combinatorial_drugs_scrna-seq.ipynb))

*Inputs:*

`freq_df` Proportion of each cell cluster in each sample

`cluster_gene_exp_df` Cluster-specific gene expression profiles

`drug_df` Drug dosage information

*Example outputs:*

```python
drug_combi_pred_df.head() # Predicted cell death percentage at specific dosage
```

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>patient</th>
      <th>drug_id_A</th>
      <th>drug_id_B</th>
      <th>cell_death_A</th>
      <th>cell_death_B</th>
      <th>cell_death_combi</th>
      <th>improve</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HN120</td>
      <td>D1007</td>
      <td>D133</td>
      <td>8.52</td>
      <td>75.63</td>
      <td>77.19</td>
      <td>1.56</td>
    </tr>
    <tr>
      <td>HN120</td>
      <td>D1007</td>
      <td>D201</td>
      <td>8.52</td>
      <td>60.39</td>
      <td>63.04</td>
      <td>2.65</td>
    </tr>
    <tr>
      <td>HN120</td>
      <td>D1007</td>
      <td>D1010</td>
      <td>8.52</td>
      <td>15.31</td>
      <td>22.39</td>
      <td>7.08</td>
    </tr>
    <tr>
      <td>HN120</td>
      <td>D1007</td>
      <td>D182</td>
      <td>8.52</td>
      <td>64.66</td>
      <td>67.22</td>
      <td>2.56</td>
    </tr>
    <tr>
      <td>HN120</td>
      <td>D1007</td>
      <td>D301</td>
      <td>8.52</td>
      <td>63.39</td>
      <td>66.39</td>
      <td>3.00</td>
    </tr>
  </tbody>
</table>

# Citation

Suphavilai C, Chia S, Sharma A et al. Predicting heterogeneity in clone-specific therapeutic vulnerabilities using single-cell transcriptomic signatures. [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.23.389676v1) 2020 Nov 

Suphavilai C, Bertrand D, Nagarajan N. Predicting cancer drug response using a recommender system. [Bioinformatics](https://academic.oup.com/bioinformatics/article/34/22/3907/5026663) 2018 Nov 15;34(22):3907-14.

# Contact information

For additional information, help and bug reports please email Chayaporn Suphavilai ([suphavilaic@gis.a-star.edu.sg](mailto:suphavilaic@gis.a-star.edu.sg))
