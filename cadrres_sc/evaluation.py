import pandas as pd
from scipy import stats

def calculate_spearman(obs_df, pred_df, sample_list, drug_list, prefix=''):

    obs_df = obs_df.loc[sample_list, drug_list]
    pred_df = pred_df.loc[sample_list, drug_list]
    
    results = []
    for s in sample_list:
        x = obs_df.loc[s].values
        y = pred_df.loc[s].values
        scor, pval = stats.spearmanr(x, y)
        results += [[s, scor, pval]]
    per_sample_df = pd.DataFrame(results, columns=['sample', '{}scor'.format(prefix), '{}pval'.format(prefix)]).set_index('sample')
    
    results = []
    for d in drug_list:
        x = obs_df[d].values
        y = pred_df[d].values
        scor, pval = stats.spearmanr(x, y)
        results += [[d, scor, pval]]
    per_drug_df = pd.DataFrame(results, columns=['drug', '{}scor'.format(prefix), '{}pval'.format(prefix)]).set_index('drug')
    
    return per_sample_df, per_drug_df

def calculate_spearman_multi_pred(obs_df, pred_df_dict, sample_list, drug_list):
    
    per_sample_df_list = []
    per_drug_df_list = []
    
    for (pred_name, pred_df) in pred_df_dict.items():
        per_sample_df, per_drug_df = calculate_spearman(obs_df, pred_df, sample_list, drug_list, prefix="{}_".format(pred_name))
        per_sample_df_list += [per_sample_df]
        per_drug_df_list += [per_drug_df]
        
    all_per_sample_df = pd.concat(per_sample_df_list, axis=1)
    all_per_drug_df = pd.concat(per_drug_df_list, axis=1)
    
    return all_per_sample_df, all_per_drug_df

# TODO: other evaluation scores
# def cal_ndcg(obs_df, pred_df):
# 	sample_ndcg_df = pd.DataFrame([None])
# 	return sample_ndcg_df