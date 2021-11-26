import os
import sys
import PATH

from models.popen import Auto_popen
from models import train_val
from models import reader

import torch
import utils
import numpy as np
import pandas as pd
from scipy import stats

import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle 
import argparse

parser = argparse.Arparser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--k", type=int, default=None, required=True)
args = parser.parse_args()
layer = args.layer

"""
Time complexity : O( v * n log(n) ), where n is the number of samples and v is the number of variables/attributes
"""

date = time.now().strftime('%m_%d')
f = open(f'/home/wergillius/Project/UTR_VAE/ipynb/layer_map/NBT_explain/{layer}_k{args.k}_{args.data}_{date}.log', 'w')
model_save_path = f'layer_map/NBT_explain/{layer}_k{args.k}_{args.data}_Sep21.model'

pj = lambda x: os.path.join(utils.script_dir, x)

# model
# pretrain_popen = Auto_popen(pj("log/Backbone/RL_celline/schedule_MTL_log_te/schedule_MTL.ini"))
pretrain_popen = Auto_popen(pj("log/Backbone/RL_3_data/rl_train_val_10fold/schedule_lr.ini"))

cv_model = pretrain_popen.vae_pth_path.replace('.pth',f'_cv{args.k}.pth')
extractor = torch.load(cv_model, map_location='cpu')['state_dict']


# data loader
torch.multiprocessing.set_sharing_strategy('file_system')

data_config = {'unmod1':'log/Backbone/RL_gru/ds4rl_unmod1_new/schdule_cv.ini', 
                'human':'log/Backbone/RL_gru/ds4rl_human_reset/schdule_cv.ini',
                'vleng':'log/Backbone/RL_gru/ds4rl_vleng_new/schdule_cv.ini',
                'muscle': 'log/Backbone/RL_gru_3DS/muscle_single_task/log_te_cv.ini', 
                'pc3': 'log/Backbone/RL_gru_3DS/pc3_single_task/log_te_cv.ini',
                'Andrev2015': 'log/Backbone/RL_gru_3DS/Andrev_single_task/log_te_cv.ini'
                }[args.data]

human_popen = Auto_popen(pj(data_config))
human_popen.kfold_cv = False if args.k is None else human_popen.kfold_cv
human_popen.kfold_index = args.k  
human_popen.shuffle = False
human_popen.pad_to = 105 #if args.data=='vleng' else 57
train_loader, _, test_loader = reader.get_dataloader(human_popen)



Y_train = train_loader.dataset.df[human_popen.aux_task_columns].values.reshape(-1,)
Y_test = test_loader.dataset.df[human_popen.aux_task_columns].values.reshape(-1,)


def encode_seq_fea(loader, l):
    # loader :
    # l : int , the lth layer
    feature_map = []

    with torch.no_grad():
        for X,Y in loader:
            X = torch.transpose(X.float(), 1, 2)
            for layer in extractor.soft_share.encoder[:l]:
                out = layer(X)
                X = out
            feature_map.append(out)

    X = np.concatenate(feature_map, axis=0)
    X_flat = X#.reshape(len(X),-1)
    return X_flat


X_train = encode_seq_fea(train_loader, args.layer)
X_test = encode_seq_fea(test_loader, args.layer)

f.write(" Training size : {}\n".format(X_train.shape)) 
f.write(" Testing size : %s\n"%X_test.shape[0])

RF_model =RandomForestRegressor(n_estimators = 100 , n_jobs=10).fit(X_train, Y_train)
# with open("ipynb/"+model_save_path, 'rb') as f:
#     RF_model = pickle.load(f)

y_pred = RF_model.predict(X_test)

# true, pred
r2=r2_score(Y_test.flatten(), y_pred)
sp=stats.spearmanr(Y_test.flatten(), y_pred)[0]
f.write('\n r2 : %f'%r2)
f.write('\n sp : %f'%sp)

filehandler = open(model_save_path, 'wb') 
pickle.dump(RF_model, filehandler)

f.write("\n model saved to {}".format(model_save_path))
f.close()