import seaborn as sns
import pandas as pd
from sklearn import preprocessing
import os 

# read in the total csv
unmod_df = pd.read_csv(os.path.join('/data/users/wergillius/UTR_VAE/multi_task/scaled_unmod.csv'))

# separate them
train = unmod_df.sort_values('total_reads',axis=0,ascending=False).iloc[20000:,:]

val = unmod_df.sort_values('total_reads',axis=0,ascending=False).iloc[:20000,:]

train_scaled_rl = preprocessing.StandardScaler().fit_transform(train.loc[:,'rl'].values.reshape(-1,1))

val_scaled_rl = preprocessing.StandardScaler().fit_transform(val.loc[:,'rl'].values.reshape(-1,1))

train.loc[:,'scaled_rl'] = train_scaled_rl
val.loc[:,'scaled_rl'] = val_scaled_rl

train.to_csv('/data/users/wergillius/UTR_VAE/multi_task/train_split_like_paper.csv',index=False)
val.to_csv('/data/users/wergillius/UTR_VAE/multi_task/val_split_like_paper.csv',index=False)