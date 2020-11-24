"""
.. module:: gexp
    :synopsis Preprocessing gene expression data

.. moduleauthor:: Nok <suphavilaic@gis.a-star.edu.sg>

"""

import pandas as pd
import numpy as np
from scipy import stats
import time

def log2_exp(exp_df):
    """Calculate log2 gene expression
    """

    return np.log2(exp_df + 1)

# TODO: add pseudo count for RNA-seq data
def normalize_log2_mean_fc(log2_exp_df):
    """Calculate gene expression fold-change based on median of each genes. The sample size should be large enough (>10).
    """

    return (log2_exp_df.T - log2_exp_df.mean(axis=1)).T, pd.DataFrame(log2_exp_df.mean(axis=1), columns=['median'])

def normalize_log2_mean_fc_with_ref(log2_exp_df, log2_ref_exp_df):
    """Calculate gene expression fold-change based on median of each genes. 
    This should not be used if the data come from different experiments.
    """

    common_genes = set(log2_ref_exp_df.index).intersection(log2_exp_df.index)
    log2_exp_df = log2_exp_df.loc[common_genes]
    log2_ref_exp_df = log2_ref_exp_df.loc[common_genes]

    return (log2_exp_df.T - log2_ref_exp_df.mean(axis=1)).T, pd.DataFrame(log2_ref_exp_df.mean(axis=1), columns=['median'])

def normalize_L1000_suite():
    """
    """

# TODO: make this run in parallel
def calculate_kernel_feature(log2_median_fc_exp_df, ref_log2_median_fc_exp_df, gene_list):
    common_genes = [g for g in gene_list if (g in log2_median_fc_exp_df.index) and (g in ref_log2_median_fc_exp_df.index)]
    
    print ('Calculating kernel features based on', len(common_genes), 'common genes')

    print (log2_median_fc_exp_df.shape, ref_log2_median_fc_exp_df.shape)
    
    sample_list = list(log2_median_fc_exp_df.columns)
    ref_sample_list = list(ref_log2_median_fc_exp_df.columns)

    exp_mat = np.array(log2_median_fc_exp_df.loc[common_genes], dtype='float')
    ref_exp_mat = np.array(ref_log2_median_fc_exp_df.loc[common_genes], dtype='float')

    sim_mat = np.zeros((len(sample_list), len(ref_sample_list)))

    start = time.time()
    for i in range(len(sample_list)):
        if (i+1)%100 == 0:
            print ("{} of {} ({:.2f})s".format(i+1, len(sample_list), time.time()-start))
            start = time.time()
        for j in range(len(ref_sample_list)):
            p_cor, _ = stats.pearsonr(exp_mat[:,i], ref_exp_mat[:,j])
            sim_mat[i, j] = p_cor

    return pd.DataFrame(sim_mat, columns=ref_sample_list, index=sample_list)


