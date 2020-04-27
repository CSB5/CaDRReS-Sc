"""
.. module:: calculation
    :synopsis Calculation functions

.. moduleauthor:: Nok <suphavilaic@gis.a-star.edu.sg>

"""

import sys, os

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import xlogy

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns



def get_pca(data_df):

    X = np.array(data_df)
    X_embedded = PCA(n_components=2).fit_transform(X)
    
    return pd.DataFrame(X_embedded, index=data_df.index, columns=['PCA-1', 'PCA-2'])

def get_tsne(data_df, metric='euclidean'):

    np.random.seed(1)

    X = np.array(data_df)
    X_embedded = TSNE(n_components=2, metric=metric).fit_transform(X)
    
    return pd.DataFrame(X_embedded, index=data_df.index, columns=['tSNE-1', 'tSNE-2'])

def plot_tsne(data_df, x, y, hue, hue_order=None, style=None, markers=None, s=10, palette=None):
    
    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(data=data_df, x=x, y=y, hue=hue, hue_order=hue_order, style=style, markers=markers, s=s, alpha=0.75, linewidth=0, palette=palette)
    plt.xticks([], [])
    plt.yticks([], [])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_scatter(data_df, x, y, hue, hue_order=None, style=None, markers=None, s=10, palette=None):
    
    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(data=data_df, x=x, y=y, hue=hue, hue_order=hue_order, style=style, markers=markers, s=s, alpha=0.75, linewidth=0, palette=palette)
    plt.xticks([], [])
    plt.yticks([], [])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def get_gene_list(gene_list_fname):
    return list(pd.read_csv(gene_list_fname, header=None)[0].values)

def calculate_cluster_fraction(sample_cluster_info_df, sample_col_name, cell_cluster_col_name, min_fraction=0.05):
    sample_cluster_info_df.index.name = 'index'
    
    cnt_df = sample_cluster_info_df[[sample_col_name]].reset_index().groupby(sample_col_name).count()
    cluster_cnt_df = sample_cluster_info_df[[sample_col_name, cell_cluster_col_name]].reset_index().groupby([sample_col_name, cell_cluster_col_name]).count()
    
    cluster_frac_df = cluster_cnt_df.copy()

    for s, data in cnt_df.iterrows():
        s_cnt = data['index']
        s_cluster_list = list(cluster_cnt_df.loc[s].index)
        cluster_frac_df.loc[[(s, c) for c in s_cluster_list], 'index'] = (cluster_cnt_df.loc[s] / s_cnt).values
        
    cluster_frac_df = cluster_frac_df.reset_index().pivot(index=sample_col_name, columns=cell_cluster_col_name, values='index')
    cluster_frac_df = cluster_frac_df[cluster_frac_df > min_fraction]
    cluster_frac_df = cluster_frac_df.fillna(0)

    return cluster_frac_df

def calculate_sample_het_entropy(cluster_frac_df, sample_type_name='sample'):
    results = []
    for s, data in cluster_frac_df.iterrows():
        results += [[s, -xlogy(data.values, data.values).sum()]]
    
    return pd.DataFrame(results, columns=[sample_type_name, 'het_entropy']).set_index(sample_type_name)

############################################
##### For gene set enrichment analysis #####
############################################

def get_gs_dict(gs_fname):
    with open(gs_fname) as f:
        content = [l.strip().split('\t') for l in f.readlines()]

    gs_gene_dict = {}
    for gs in content:
        gs_gene_dict[gs[0]] = gs[2:]
        
    return gs_gene_dict

def calculate_pathway_activity(gs_gene_dict, log2_fc_exp_df):
    results = []

    input_sample_list = list(log2_fc_exp_df.columns)
    input_gene_list = set(log2_fc_exp_df.index)

    for gs_name, genes in gs_gene_dict.items():

        common_genes = list(input_gene_list.intersection(genes))
        results += [[gs_name] + list(log2_fc_exp_df.loc[common_genes].sum().values)]

    result_df = pd.DataFrame(results, columns = ['id'] + input_sample_list)
    pathway_activity_df = result_df.set_index('id').T

    return pathway_activity_df

def calculate_drug_pathway_assoc(pathway_activity_df, response_df):

    sample_list = [s for s in pathway_activity_df.index if s in response_df.index]
    gs_list = pathway_activity_df.columns
    drug_list = response_df.columns

    r_mat = np.array(response_df.loc[sample_list])
    a_mat = np.array(pathway_activity_df.loc[sample_list])

    assoc_mat = np.zeros((r_mat.shape[1], a_mat.shape[1]))
    for d, d_name in enumerate(drug_list):
        x = r_mat[:, d]
        for gs, gs_name in enumerate(gs_list):
            y = a_mat[:, gs]

            pcor, pval = stats.pearsonr(x, y)
            assoc_mat[d, gs] = pcor
    
    return pd.DataFrame(assoc_mat, index=drug_list, columns=gs_list)

def calculate_pathway_activity_gsea(log2_median_fc_exp_df, pathway_db_name='Biocarta'):
    """Calculate pathway activity
    """

    return pd.DataFrame([None])

#################################################################
##### Newton-like method for combining dose-response curves #####
#################################################################

def cal_y(x, a_list, b_list, p_list):

    """
    Calculate y (% cell death)
    """

    y = 0.0
    for a, b, p in zip(a_list, b_list, p_list):
        y += p*1/(1+2**((a-x)*b))

    return y

def cal_m(x, a_list, b_list, p_list):

    """
    Calculate m = the slope of the tangent line at the current (x, y)
    """

    m = 0.0
    for a, b, p in zip(a_list, b_list, p_list):
        m += p*(np.log(2)*2**(b*(a-x))*b)/((1+2**(b*(a-x)))**2)
    return m

def combine_curves(cl_d_df):

    """
    cl_d_df has columns = [frequency, ic50, slope]
    (See combine_curve_ic50.ipynb)
    Output: combined ic50 value
    """    

    n_clusters = cl_d_df.shape[0]
    p_list = cl_d_df['frequency'].values
    a_list = cl_d_df['ic50'].values
    b_list = cl_d_df['slope'].values
    
    ##### Initiate x #####
    # [Option 1] based on both positions and slopes
    # x = np.sum(np.multiply(p_list, np.multiply(a_list, b_list))) / np.sum(np.multiply(b_list, p_list))
    # [Option 2] based on positions
    x = np.sum(np.multiply(a_list, p_list))
    
    y = 0
    step = 0
    eps = 0.001
    while ~(np.abs(y - 0.05) > eps) :
        m = cal_m(x, a_list, b_list, p_list)
        y = cal_y(x, a_list, b_list, p_list)
        print ("Step {:d}: x={:.2f}, y={:.2f}, m={:.2f}".format(step, x, y, m))
        x1 = x + 1
        x2 = x - 1

        y1 = m * (x1 - x) + y
        y2 = m * (x2 - x) + y

        x = x + ((0.5-y)/m)
        step += 1
    
    combined_ic50 = x

    return combined_ic50

# def combine_IC50_no_quantity(cl_d_df):
#     """
#     cl_d_df has columns = [frequency, ic50, slope]
#     (See combine_curve_ic50.ipynb)
#     """
    
#     n_clusters = cl_d_df.shape[0]
#     p_list = np.array([1./n_clusters for i in range(n_clusters)])
#     a_list = cl_d_df['ic50'].values
#     b_list = cl_d_df['slope'].values
    
#     ##### Calculate initial x #####
#     # [1] based on both positions and slopes
#     # x = np.sum(np.multiply(p_list, np.multiply(a_list, b_list))) / np.sum(np.multiply(b_list, p_list))
#     # [2] based on positions
#     x = np.sum(np.multiply(a_list, p_list))
    
#     y = 0
#     step = 0
#     eps = 0.001
#     while ~(np.abs(y - 0.05) > eps) :
#         m = cal_m(x, a_list, b_list, p_list)
#         y = cal_y(x, a_list, b_list, p_list)
#         print ("Step {:d}: x={:.2f}, y={:.2f}, m={:.2f}".format(step, x, y, m))
#         x1 = x + 1
#         x2 = x - 1

#         y1 = m * (x1 - x) + y
#         y2 = m * (x2 - x) + y

#         x = x + ((0.5-y)/m)
#         step += 1
    
#     combined_ic50 = x

#     return combined_ic50

def create_input_for_calculate_combined_ic50(cluster_pred_df, cl_cluster_frac_df):

    results = []

    cluster_list = cl_cluster_frac_df.columns
    sample_list = cl_cluster_frac_df.index
    drug_list = cluster_pred_df.columns

    for s in sample_list:

        sel_cluster_name = cluster_list[(cl_cluster_frac_df.loc[s] > 0).values]
        sel_cluster_frac = cl_cluster_frac_df.loc[s][cl_cluster_frac_df.loc[s] > 0].values
        norm_sel_cluster_frac = sel_cluster_frac/np.sum(sel_cluster_frac)
        
        for c_name, frac in zip(sel_cluster_name, norm_sel_cluster_frac):
            for d in drug_list:
                ic50 = cluster_pred_df.loc[c_name, d]
                results += [[s, d, ic50, 1, c_name, frac]]

    cell_type_pred_df = pd.DataFrame(results, columns=['cell_line', 'drug', 'ic50', 'slope', 'cluster_id', 'frequency'])
                
    return cell_type_pred_df, sample_list, drug_list


def calculate_combined_ic50(cluster_pred_df, cl_cluster_frac_df):

    cell_type_pred_df, sample_list, drug_list = create_input_for_calculate_combined_ic50(cluster_pred_df, cl_cluster_frac_df)

    results = []
    for s in sample_list:
        for d in drug_list:
            cl_d_df = cell_type_pred_df[(cell_type_pred_df['cell_line'] == s) & (cell_type_pred_df['drug'] == d)]
            combined_ic50 = combine_curves(cl_d_df)
            results += [[s, d, combined_ic50]]

    combined_ic50_df = pd.DataFrame(results, columns=['cell_line', 'drug', 'combined_ic50'])

    return combined_ic50_df.pivot(index='cell_line', columns='drug', values='combined_ic50')