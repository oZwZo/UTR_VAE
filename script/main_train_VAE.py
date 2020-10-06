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

conv_sizes = [4,32,128,32]

# encoder = DL_models.Conv_backbond(conv_sizes,stride=2,padding=1)
# decoder = DL_models.Conv_backbond(conv_sizes[::-1],stride=2,padding=0,direction='decode')
# model = DL_models.VAE(encoder=encoder,decoder=decoder,latent_dim=16,hidden_dim=128)
hidden_size=8
L_encoder = DL_models.LSTM_backbond(4,hidden_size=8,num_layers=3,bidirectional=True)
L_decoder = DL_models.LSTM_backbond(2*hidden_size,hidden_size=4,num_layers=3,bidirectional=False)
# initiate VAE
LSTM_out_dim = 2*hidden_size*100
model = DL_models.VAE(encoder=L_encoder,decoder=L_decoder,latent_dim=16,out_dim=LSTM_out_dim)

for idx,data in enumerate(train_loader):
    (X,y) = data
    X = X.float()
    # X = X.float().transpose(1,2)
    y = y.long()
    
    
    Z,(hidden_state,cell_state) = L_encoder(X)
    out,_ = L_decoder(Z)
    
    # Z,_ = encoder(X)
    # out,_ = decoder(Z)
    print(Z.shape)
