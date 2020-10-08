import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import torch
import numpy as np 
from torch import functional as F
from model import DL_models
from model import reader

dataset = reader.UTR_dataset(cell_line='A549')

train_loader,val_loader,test_loader = reader.get_splited_dataloader(dataset,ratio=[0.7,0.1,0.2],batch_size=20,num_workers=4)

# ===== Conv =====

model = DL_models.LSTM_VAE(input_size=4, 
                           hidden_size_enc=16, 
                           hidden_size_dec=8, 
                           num_layers=2, 
                           latent_dim=8, 
                           seq_in_dim=1,
                           decode_type='seq')


