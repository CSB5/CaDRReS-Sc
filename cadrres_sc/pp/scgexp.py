"""
.. module:: scgexp
    :synopsis Preprocessing scRNA-seq gene expression data

.. moduleauthor:: Nok <suphavilaic@gis.a-star.edu.sg>

"""

import pandas as pd

# TODO: functions for preprocessing scRNA-seq data

def get_sample_cluster_percent(sample_cell_df, cutoff=0.05):
    """Calculate proportion of each cell cluster in each sample

    :param cell_cluster_df: Two columns for cell_id and cluster_id
    :type cell_cluster_df: DataFrame
    :param cutoff: If a cluster presented in less than the cutoff, then set its percentage to 0
    :type cutoff: float
    :returns:  DataFrame (cell_cluster_proportion_df) of proportion of cell clusters (row:sample, col:cluster)

    """
    # return pd.DataFrame([None])

def calculate_het_score(cell_cluster_proportion_df):
    """Calculate intra-sample heterogeneity score using entropy (with renormalization)

    :param cell_cluster_proportion_df: proportion of cell clusters (row:sample, col:cluster)
    :type cell_cluster_df: DataFrame
    :returns:  DataFrame of heterogeneity score

    """    

def calculate_cluster_prof(sc_log2_exp_df, sc_info_df):
    """Calculate gene expression profile for each cell cluster
    """
    # exp_with_info_df = pd.merge(sc_log2_exp_df.T, sc_info_df[['Cluster']], left_index=True, right_index=True)
    # return exp_with_info_df.groupby('Cluster').median().T

