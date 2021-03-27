# To add a new cell, type ' '
# To add a new markdown cell, type '  [markdown]'
 
import os
os.chdir("/home/wergillius/Project/UTR_VAE/ribo_public")
import sys
import PATH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload
from matplotlib import cm
from sklearn.decomposition import PCA
import parse_ribomap as Pribo
from scipy import stats


from matplotlib.backends.backend_pdf import PdfPages
 
def pj(*args):
    path = os.path.join(*args)
    assert os.path.exists(path)
    return path

def read_quant_and_rename(path):
    SRR = path.split("/")[-2]
    DF = pd.read_table(path,sep='\t')[['Name','TPM','NumReads']]
    col_rename={"Name":"Transcipt_ID","TPM":"TPM_%s"%SRR,"NumReads":"TC_%s"%SRR}
    DF.rename(columns=col_rename,inplace=True)
    return DF

def scale_library_depth(df,col):
    df.loc[:,col.replace("TC","TCM")] = np.divide(df.loc[:,col],df.loc[:,col].sum())*1e6
    
def apply_filtering_by_threshold(DF,threshold=0):
    
    DF_thrs=DF[(DF.iloc[:,1:] > threshold).all(axis=1)]    
    data = DF_thrs.iloc[:,1:].values
    data_log = np.log2(DF_thrs.iloc[:,1:].values)
    
    return DF_thrs,data,data_log

def log_data_pair_plot(data_log,TCM_col,save_path):
    n_sample = len(TCM_col)
    fig,axs = plt.subplots(n_sample,n_sample,figsize=(6*n_sample,6*n_sample-2))

    for i in range(n_sample):
        for j in range(n_sample):
            axs[i,j].scatter(data_log[:,i],data_log[:,j],s=4,alpha=0.2)
            axs[i,j].set_xlabel("log "+TCM_col[i])
            axs[i,j].set_ylabel("log "+TCM_col[j])

    fig.savefig(save_path,format='png') 

public_ribo="/data/users/wergillius/UTR_VAE/public_ribo"
GSE65778_dir = "/data/users/wergillius/UTR_VAE/public_ribo/GSE65778"
GSE_ls = [dir for dir in os.listdir(public_ribo) if '.sh' not in dir]
print(GSE_ls)
new_GSE=GSE_ls[:-1]

# read processed GSE65778 datset as `full df`
full_GSE65778 = pd.read_csv(pj(GSE65778_dir,'Ribomap_salmon_out_annotated_full.csv'))

# ------- all salmon ouput -------------
all_salmon_sf_ls = []
for GSE in new_GSE:
    # for 3 new GSE dataset
    GSE_dir=pj(public_ribo,GSE,'salmon_out')
    SRR_ls = [dir for dir in os.listdir(GSE_dir) if 'SRR' in dir]

    for SRR in SRR_ls:
        # might have more than 1 RNA-seq 
        sf_path = pj(GSE_dir,SRR,'quant.sf')
        all_salmon_sf_ls.append(sf_path)
del sf_path

print("All salmon quant file are :")
all_salmon_sf_ls


# ------------ merge 3 new GSE -------------------
## initiate a df to merge
Salmon_df = read_quant_and_rename(all_salmon_sf_ls[0])

for quant in all_salmon_sf_ls[1:]:
    Df = read_quant_and_rename(quant)
    Salmon_df = Salmon_df.merge(Df,left_on=['Transcipt_ID'],right_on=['Transcipt_ID'])

# --------- merge with GSE65778 ------------    

TC_col_65778 = [col for col in full_GSE65778.columns if 'TC_S' in col]
GSE65778_TC = full_GSE65778[['Transcipt_ID']+TC_col_65778]
Salmon_df = Salmon_df.merge(GSE65778_TC,left_on=['Transcipt_ID'],right_on=['Transcipt_ID'],how='inner')


TC_col = [col for col in Salmon_df.columns if 'TC_SRR' in col]
TC_salmon_df = Salmon_df[['Transcipt_ID']+TC_col]

# ----------  calcualte TCM ------------
for col in TC_col:
    scale_library_depth(TC_salmon_df,col)
    
TCM_col = [col for col in TC_salmon_df.columns if 'TCM_SRR' in col]
TCM_salmon_df = TC_salmon_df[['Transcipt_ID']+TCM_col]

# -------- threshold != 0 -------------
TCM_thrs0_df,TCM_data_thrs0,TCM_data_log_thrs0 = apply_filtering_by_threshold(TCM_salmon_df,threshold=0)  


# ------------ pair plot -------------------
log_data_pair_plot(TCM_data_log_thrs0,TCM_col,
                   save_path="/home/wergillius/Project/UTR_VAE/ribo_public/data/9_SRR_TCM_threshold0_pairplot.png")


TCM_cor_M0 = Pribo.zwz_cor_m(TCM_data_thrs0,axis=1)
heatmap = Pribo.cor_M_heatmap(TCM_cor_M0)
heatmap.savefig("/home/wergillius/Project/UTR_VAE/ribo_public/data/9_SRR_thres0_heatmap.png",format='png')

# -------- threshold >1 -------------
TCM_thrs0_df,TCM_data_thrs1,TCM_data_log_thrs1 = apply_filtering_by_threshold(TCM_salmon_df,threshold=1)  


# ------------ pair plot -------------------
log_data_pair_plot(TCM_data_log_thrs1,TCM_col,
                   save_path="/home/wergillius/Project/UTR_VAE/ribo_public/data/9_SRR_TCM_threshold1_pairplot.png")


TCM_cor_M1 = Pribo.zwz_cor_m(TCM_data_log_thrs1,axis=1)
heatmap = Pribo.cor_M_heatmap(TCM_cor_M1)
heatmap.savefig("/home/wergillius/Project/UTR_VAE/ribo_public/data/9_SRR_thres1_heatmap.png",format='png')
