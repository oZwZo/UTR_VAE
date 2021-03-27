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

def Read_stats_and_rename(stats_path):
    """read stats file and label the SRR id on each column"""
    # read stats to DF first
    df = Pribo.read_ribomap_stats(stats_path)[['Transcipt_ID','Ribo_count','Transcript_Rabd','TE-score']]
    #  rename the column to label the SRR run it belongs to
    SRR = stats_path.split("/")[-1].split("_")[0]
    col_rename={"Transcipt_ID":"Transcipt_ID","Ribo_count":"RC_%s"%SRR,"Transcript_Rabd":"Tabd_%s"%SRR,	"TE-score":"TE_%s"%SRR}
    df.rename(columns=col_rename,inplace=True)
    return df



 
public_ribo="/data/users/wergillius/UTR_VAE/public_ribo"
GSE65778_dir = "/data/users/wergillius/UTR_VAE/public_ribo/GSE65778"
GSE_ls = [dir for dir in os.listdir(public_ribo) if '.sh' not in dir]
print(GSE_ls)
new_GSE=GSE_ls[:-1]


 
# find out all the abs path of ribo-map output file that in `stats` format
all_ribomap_stats_ls = []

for GSE in new_GSE:
    # for 3 new GSE dataset
    GSE_dir = pj(public_ribo,GSE,'ribomap_out')
    # include all `stats` file 
    stats_path = [pj(GSE_dir,file) for file in os.listdir(pj(GSE_dir)) if file.endswith(".stats")]
    all_ribomap_stats_ls += stats_path

print("All stast file are :")
all_ribomap_stats_ls


# ----------------- merge 3 new ribo-seq  with 4 from GSE65778 ------------------

# initiate a df as the base to merge with
Ribo_df = Read_stats_and_rename(all_ribomap_stats_ls[0])
for stats_path in all_ribomap_stats_ls[1:]:
    # read the 2nd and 3rd stats
    df = Read_stats_and_rename(stats_path)
    Ribo_df=df.merge(Ribo_df,left_on=['Transcipt_ID'],right_on=['Transcipt_ID'],how='inner')


# read processed GSE65778 datset as `full df`
full_GSE65778 = pd.read_csv(pj(GSE65778_dir,'Ribomap_salmon_out_annotated_full.csv'))

# take RC_SRRXXX columns of the `full DF`
RC_col_65778 = [col for col in full_GSE65778.columns if 'RC_S' in col]
GSE65778_RC = full_GSE65778[['Transcipt_ID']+RC_col_65778]

# merge these 4 processed ribo-seq samples with 3 
Ribo_df = Ribo_df.merge(GSE65778_RC,left_on=['Transcipt_ID'],right_on=['Transcipt_ID'],how='inner')


# ------------------------- compute Ribosome reads count per million  (RCM) --------------------
 
RC_col = [col for col in Ribo_df.columns if 'RC' in col]
for col in RC_col:
    Ribo_df.loc[:,col.replace("RC","RCM")] = np.divide(Ribo_df.loc[:,col],Ribo_df.loc[:,col].sum())*1e6


 
RCM_col = [col for col in Ribo_df.columns if 'RCM' in col]

# take the RCM part
RCM_ribo_df = Ribo_df[['Transcipt_ID']+RCM_col]

# save csv
RCM_ribo_df.to_csv("/home/wergillius/Project/UTR_VAE/ribo_public/data/20210326_ribo_7_SRR_RCM.csv",index=False)
RCM_ribo_df = pd.read_csv("/home/wergillius/Project/UTR_VAE/ribo_public/data/20210326_ribo_7_SRR_RCM.csv")

# ------------------------- filtterd     --------------------
# threshold = 0 
RCM_ribo_df=RCM_ribo_df[(RCM_ribo_df != 0).all(axis=1)]
# take values
RCM_data = RCM_ribo_df.iloc[:,1:].values
RCM_data_log = np.log2(RCM_ribo_df.iloc[:,1:].values)

corr_M = Pribo.zwz_cor_m(RCM_data_log,axis=1)

n_sample = 7
fig,axs = plt.subplots(7,7,figsize=(42,40))

for i in range(n_sample):
    for j in range(n_sample):
        axs[i,j].scatter(RCM_data_log[:,i],RCM_data_log[:,j],s=4,alpha=0.2)
        axs[i,j].set_xlabel("log "+RCM_col[i])
        axs[i,j].set_ylabel("log "+RCM_col[j])

fig.savefig("/home/wergillius/Project/UTR_VAE/ribo_public/data/20210326_ribo_7_SRR_RCM.png",format=
            'png')     

fig = plt.figure(figsize=(9,9))
ax = fig.gca()
heatmap=ax.imshow(corr_M,aspect='auto',cmap=cm.RdBu)
plt.colorbar(heatmap)

fig.savefig("/home/wergillius/Project/UTR_VAE/ribo_public/data/7_correlation_efficiency.png",format=
            'png') 



RCM_col = [col for col in Ribo_df.columns if 'RCM' in col]

# take the RCM part
RCM_ribo_df_ths1 = Ribo_df[['Transcipt_ID']+RCM_col]
# threshold = 0 
RCM_ribo_df_ths1=RCM_ribo_df_ths1[(RCM_ribo_df_ths1.iloc[:,1:] >= 1).all(axis=1)]
# take values
RCM_data_ths = RCM_ribo_df_ths1.iloc[:,1:].values
RCM_data_ths1_log = np.log2(RCM_ribo_df_ths1.iloc[:,1:].values)


n_sample = 7
fig,axs = plt.subplots(7,7,figsize=(42,40))

for i in range(n_sample):
    for j in range(n_sample):
        axs[i,j].scatter(RCM_data_ths1_log[:,i],RCM_data_ths1_log[:,j],s=4,alpha=0.2)
        axs[i,j].set_xlabel("log "+RCM_col[i])
        axs[i,j].set_ylabel("log "+RCM_col[j])

fig.savefig("/home/wergillius/Project/UTR_VAE/ribo_public/data/7_SRR_RCM_threshold1_pairplot.png",format=
            'png')   