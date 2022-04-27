import os
import sys
import PATH
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import time
from models import reader
from models import train_val
from models.popen import Auto_popen
from models import max_activation_patch as MAP

import torch
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser(description='validate the effect by the independant FACS library')
parser.add_argument('--config_file', type=str, required=True, 
                    help='the `ini` config file of the model')

parser.add_argument('--library', type=str, required=False, 
                    default='/data/users/wergillius/UTR_VAE/Alan_dataset/JF3_processed.csv',
                    help='the validation sequence library')
parser.add_argument('--quant_column', type=str, required=False, default='log_count',
                    help='the column name recording `translation rate`')

parser.add_argument('--unique_activation', type=bool, required=False, default=True,
                    help='retain the most acitvated sequences to represent motifs')

parser.add_argument('--effect_metric', type=str, required=False, default="Pearsonr",
                    help='retain the most acitvated sequences to represent motifs')
parser.add_argument('--task', type=str, required=False, default='human',
                    help='retain the most acitvated sequences to represent motifs')
args = parser.parse_args()


JF3_sc = pd.read_csv(args.library)
label = args.quant_column #'log_count' 'pseudo_intensity' 'Bin_1_propotion', 'Bin1p-Bin3p'
assert (label in JF3_sc.columns)
unique = args.unique_activation

def load_MAP_and_pr_spr(config_path, n_patch=3000, kfold_index=1):
    config = Auto_popen(config_path)
    config.kfold_cv = 'train_val'
    mu_cv0 = MAP.Maxium_activation_patch(popen=config, which_layer=3, n_patch=n_patch, kfold_index=kfold_index)
    # featmap = mu_cv0.extract_feature_map(task='human',which_set=0)
    
    # spr_ls = []
    # pr_ls = []
    # for i in range(256):
    #     _ , (all_spr, all_pr) = mu_cv0.activation_density(i, to_print=False, to_plot=False)
    #     spr_ls.append(all_spr)
    #     pr_ls.append(all_pr)

    # pr_ay = np.stack(pr_ls)
    # spr_ay = np.stack(spr_ls)
    
    
    # RBP_df = pd.read_table(RBP_path).iloc[:-3]
    # RBP_df = RBP_df.sort_values(['Query_ID','q-value'], ascending=True).drop_duplicates(['Query_ID'], keep='first')
    # sig_RBP = np.array(RBP_df['q-value']  <= 0.05).astype(int)

    return mu_cv0 

def Channel_max_act_count_AlanDs(Alan_Df, feature_map_ay, patch_size=20):

    act_counts = pd.DataFrame([])
    # get the location of the most acti sequences
    posi = np.argpartition(feature_map_ay, -1*patch_size, axis=0)[-1*patch_size:]

    # take out the most acitvated sequences for each channel
    for i in range(256):
        act_count = Alan_Df.iloc[posi[:,i]][['seq',label]]
        activa_value = feature_map_ay[posi[:,i],i]
        # normalize activation so that they can compare across channel 
        Zscale = lambda x : (x - activa_value.mean()) / activa_value.std()
        act_count['activation'] = Zscale(activa_value) 
        act_count['channel_name'] = np.full((patch_size,),str(i))
        act_counts = act_counts.append(act_count)
        
    act_counts['channel'] = act_counts['channel_name'].astype(int)
    act_counts = act_counts.dropna(axis=0)
        
    return act_counts

def activation_filtering(act_counts, effect_df, unique, filtering_threshold, task):
    
    # merging channel wise information here
    
    temp_merge = act_counts.merge(effect_df, left_on=['channel'], right_on=['channel'])

    # apply filtering 
    act_count_df = temp_merge.query(f'`{task}` > @filtering_threshold | `{task}` < -1*@filtering_threshold')
    
    if unique:
        act_count_df = act_count_df.sort_values(['seq', 'activation'], ascending=False)
        act_count_df = act_count_df.drop_duplicates(subset=['seq'],keep='first' )

    print("n_seq : {}, threshold: {}".format(act_count_df.shape[0], filtering_threshold))

    # aggregation of counts of activated sequencs to channel level
    agg_of_ch = lambda x, fuc : fuc(act_count_df[act_count_df['channel']==x][label])

    act_count_df['mean_count'] = act_count_df['channel'].apply(
                agg_of_ch, args=(np.mean,))
    act_count_df['median_count'] = act_count_df['channel'].apply(
                agg_of_ch, args=(np.median,))
    act_counts.dropna()

    mean_V = JF3_sc[label].mean()
    median_V = JF3_sc[label].median()

    #  constrcut channel wise info with Random Forest Feature Importance
    channel_wise=act_count_df[['channel', 'mean_count', 'median_count']].drop_duplicates()
    channel_wise['above_average_mean'] = channel_wise['mean_count'].values - mean_V
    channel_wise['above_average_median'] = channel_wise['median_count'].values - median_V
    channel_wise = channel_wise.merge(effect_df,left_on='channel',right_on='channel')
    return act_count_df, channel_wise


def merge_by_channel(act_count_df,  channel_wise, task):
    """
    merge pearson_r, spearman_r, RF_info and unique activation into 1 df
    Input: 
    act_count_df : DataFrame : the output of `activation_filtering`  with columns pr spr channel mean/median_count
    RF_impfeat : The annotation file of RF transferred model, used to bring sequence consensus
    """    
    
    # channel_wise = channel_wise.query('`spearman_r` > @filtering_threshold | `spearman_r` < -1*@filtering_threshold')
    n_pos = channel_wise.query(f'`{task}` >=0 & `above_average_mean`>=0').shape[0]
    n_neg = channel_wise.query(f'`{task}` <0 & `above_average_mean`< 0').shape[0]
    pctg_pos = n_pos/channel_wise.query(f'`{task}` >=0').shape[0]
    pctg_neg = n_neg/channel_wise.query(f'`{task}` < 0').shape[0]
    print("pos: {} ({:.3f})".format(n_pos, pctg_pos))
    print("neg: {} ({:.3f})".format(n_neg, pctg_neg))
    
    return act_count_df, n_pos, pctg_pos, n_neg, pctg_neg

if __name__ == "__main__":
    
    # define MAP object
    config_path = os.path.join("/ssd/users/wergillius/Project/UTR_VAE/",args.config_file)
    M2R_MAP  = load_MAP_and_pr_spr(config_path)
    setting_name = M2R_MAP.popen.setting_name
    run_name = M2R_MAP.popen.run_name

    # load the estimatd effect of the model
    save_dir = (f'/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/channel_r/{setting_name}')
    est_effect_path = f'{save_dir}/{M2R_MAP.popen.run_name}_{args.effect_metric}.csv'
    effect_df = pd.read_csv(est_effect_path)
    effect_df.loc[:,'channel'] = effect_df.channel_name.apply(lambda x: int(x.split('_')[-1]))

    # load the FACS library
    ds_set = reader.MTL_dataset(JF3_sc,pad_to=105,seq_col='seq',
                            aux_columns=[label],other_input_columns=None,trunc_len=None)

    # convert to Dataloader and extract feature map
    JF3_loader =DataLoader(ds_set, batch_size=32, shuffle=False)
    jc3_featmap = M2R_MAP.extract_feature_map(task='human',
                         which_set=0,  extra_loader=JF3_loader)
    M2R_MAP.df = JF3_sc
    feature_map_ay = jc3_featmap.max(axis=-1)
    
    
    # the result dir
    date = time.strftime("%B%d")
    try:
        result_dir = args.library.split('/')[-1].split('.')[0]+['','_unique'][unique]+f"{setting_name}-{run_name}-{label}-{date}"
        result_dir = os.path.join('/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/FACS_unique_act_results',result_dir)
        os.mkdir(result_dir)
    except OSError as error:
        print(error) 
    
    FM_path = '{}/feature_map.npy'.format(result_dir)
    if not os.path.exists(FM_path):
        np.save(FM_path, feature_map_ay)
    # uniquely activation 
    patches_list = list(range(80,90,5)) 
    pos_n_list = []
    neg_n_list = []
    pos_pct_list = []
    neg_pct_list = []
    filter_list = []
    patch_size = []
    for n in patches_list:
        print("-- patch size = %d --"%n)
        act_counts = Channel_max_act_count_AlanDs(JF3_sc, feature_map_ay, patch_size=n)
        # triple mergenc by sequence, and by channels
        for f in [0]:# ,0.05, 0.1, 0.15]:# 0.2, 0.3]:
            unique_act_counts, channel_stats = activation_filtering(act_counts, effect_df, unique=unique, filtering_threshold=f, task=args.task)

            Tri_merge_seq, n_pos, pctg_pos, n_neg, pctg_neg = merge_by_channel(unique_act_counts, channel_stats, task=args.task)
            pos_n_list.append(n_pos)
            neg_n_list.append(n_neg)
            pos_pct_list.append(pctg_pos)
            neg_pct_list.append(pctg_neg)
            filter_list.append(f)
            patch_size.append(n)
        # save result
            channel_stats.to_csv('{}/N{}F{}_channel_info.csv'.format(result_dir, n, f), index=False)
            
        
    n_summery = pd.DataFrame({'n_patch':patch_size, 'threshold':filter_list, 
                                'n_pos':pos_n_list, 'pctg_pos':pos_pct_list,
                                'n_neg':neg_n_list, 'pctg_neg':neg_pct_list})
    n_summery.to_csv('{}/summery.csv'.format(result_dir,n), index=False) 