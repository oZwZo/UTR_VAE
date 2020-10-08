import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import time
import logging
import numpy as np 

import torch
from torch import nn, optim
from torch import functional as F

from model import DL_models
from model import reader
from model import train_val
from model.optim import ScheduledOptim

# Run name



# read data
dataset = reader.UTR_dataset(cell_line='A549')
train_loader,val_loader,test_loader = reader.get_splited_dataloader(dataset,ratio=[0.7,0.1,0.2],batch_size=20,num_workers=4)

# start model
model = DL_models.LSTM_VAE(input_size=4, 
                           hidden_size_enc=16, 
                           hidden_size_dec=8, 
                           num_layers=2, 
                           latent_dim=8, 
                           seq_in_dim=1,
                           decode_type='seq')

# set optimizer
optimizer = ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                      betas=(0.9, 0.98), 
                                      eps=1e-09, 
                                      weight_decay=1e-4, 
                                      amsgrad=True),
                           n_warmup_steps=20)


for epoch in range(max_epoch):
    
    train_val.train(dataloader=train_loader,model=model,optimizer=optimizer)
    
    # validate model for every 20 epoch
    if epoch % 20 == 0:
        validate_values = train_val.validate(val_loader,model)
    
    # TODO : verbose
    
    # TODO : compare the result 
    
    # TODO : save model 
