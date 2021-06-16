import os
import sys
import PATH
import numpy as np
import pandas as pd
import utils

import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from importlib import reload
import analysis_for_MTL as A_MTL
import ribo_public.parse_ribosome as Pribo
from sklearn import preprocessing

from models import Backbone
from models import reader
from models import train_val
from models import Cross_stitch
from models.popen import Auto_popen
from models.loss import Dynamic_Task_Priority,Dynamic_Weight_Averaging

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset,DataLoader,random_split

DF_to_use = sys.argv[1]
assert DF_to_use in ["unmod1_df","design_df","vleng_df"]

NBT_dir = "/data/users/wergillius/UTR_VAE/Ex_data"
# read in data
data_dir = "/data/users/wergillius/UTR_VAE/Ex_data"
df_dict = {}
df_dict["unmod1_df"] = pd.read_csv(os.path.join(data_dir,"GSM3130435_egfp_unmod_1.csv"))
df_dict["design_df"] = pd.read_csv(os.path.join(data_dir,"GSM3130443_designed_library.csv"))
df_dict["vleng_df"]  = pd.read_csv(os.path.join(data_dir,"GSM4084997_varying_length_25to100.csv"))

DF = df_dict[DF_to_use]

# the multi task model. trained with unmod1
abb5_P = "/home/wergillius/Project/UTR_VAE/log/CrossStitch/CrossStitch_Model_MTL/a5b5/a5b5_g2_11.ini"
logger = utils.setup_logs('test.log')
abb5_dict= A_MTL.read_main_(config_file=abb5_P,logger=logger,cuda=None)

# cross stitch model
CS_model = abb5_dict['model']
CS_popen = abb5_dict['popen']

CS_popen.cuda_id = 3
CS_model.eval()
unmod1_recons_latent = []

unmod1_loader = reader.loader_from_df(DF,POPEN=CS_popen)

with torch.no_grad():
    for X,Y in unmod1_loader:
        X = X.float().cuda(3)
        out = CS_model(X)
        mu,sigma = out['Recons'][1:]
        code = CS_model.backbone['Recons'].reparameterize(mu,sigma)
        
        unmod1_recons_latent.append(code.cpu().numpy())
        
unmod1_latent_ay = np.concatenate(unmod1_recons_latent,axis=0)
        
np.save(os.path.join(NBT_dir,'%s_recons_latent.npy'%DF_to_use[:-3]),unmod1_latent_ay)