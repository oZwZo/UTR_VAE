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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle 
import argparse

parser = argparse.Arparser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--k", type=int, default=None, required=True)
args = parser.parse_args()
layer = args.layer

"""
Time complexity : O( v * n log(n) ), where n is the number of records and v is the number of variables/attributes
"""

f = open(f'/home/wergillius/Project/UTR_VAE/ipynb/layer_map/{layer}_k{args.k}_RF_Sep15.log', 'w')
model_save_path = f'layer_map/{layer}_k{args.k}_RF_Sep15.model'

pj = lambda x: os.path.join(utils.script_dir, x)

# model
pretrain_popen = Auto_popen(pj("log/Backbone/RL_gru/mixing_task/big_RL_gru.ini"))
extractor = torch.load(pretrain_popen.vae_pth_path, map_location='cpu')['state_dict']


# data loader
torch.multiprocessing.set_sharing_strategy('file_system')
human_popen = Auto_popen(pj('log/Backbone/RL_gru/ds4rl_human_reset/schdule_cv.ini'))
human_popen.kfold_cv = False if args.k is None else 'train_val'
human_popen.kfold_index = args.k  
human_popen.shuffle = False
human_popen.pad_to = 57
train_loader, _, test_loader = reader.get_dataloader(human_popen)



Y_train = train_loader.dataset.df.rl.values.reshape(-1,)
Y_test = test_loader.dataset.df.rl.values.reshape(-1,)


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


y_pred = RF_model.predict(X_test)

# true, pred
r2=r2_score(Y_test.flatten(), y_pred)
f.write('\n r2 : %f'%r2)

filehandler = open(model_save_path, 'wb') 
pickle.dump(RF_model, filehandler)

f.write("\n model saved to {}".format(model_save_path))
f.close()