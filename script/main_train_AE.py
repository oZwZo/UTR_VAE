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
from model.popen import Auto_popen

device = 'cuda' if torch.cuda.is_available() else 'cpu'
popen = Auto_popen(os.path.join(utils.log_dir,'LSTM_AE','setting1_AE','setting1_AE.json'))


# TODO :Run name
if popen.run_name is None:
    run_name = popen.model_type + time.strftime("__%Y_%m_%d_%H:%M")
else:
    run_name = popen.run_name
    
# TODO : built model dir
utils.check_experiment(run_name)

# TODO : log dir

logger = utils.setup_logs(save_dir = os.path.join(utils.log_dir,model_type,run_name),
                          run_name=run_name)


# read data
dataset = reader.UTR_dataset(cell_line='A549')
train_loader,val_loader,test_loader = reader.get_splited_dataloader(dataset,ratio=[0.7,0.1,0.2],batch_size=20,num_workers=4)

# start model
model = DL_models.LSTM_AE(input_size=4, 
                           hidden_size_enc=16,  
                           hidden_size_dec=32,
                           num_layers=2, 
                           latent_dim=8, 
                           seq_in_dim=4).cuda()


# set optimizer
optimizer = ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                      betas=(0.9, 0.98), 
                                      eps=1e-09, 
                                      weight_decay=1e-4, 
                                      amsgrad=True),
                           n_warmup_steps=20)



for epoch in range(max_epoch):
    
    # train
    logger.info("\n===============================|    epoch {}   |===============================\n".format(epoch))
    train_val.train(dataloader=train_loader,model=model,optimizer=optimizer,popen=popen)
    
    # validate 
    # for every 20 epoch
    if epoch % 20 == 0:
        val_total_loss,val_avg_acc = train_val.validate(val_loader,model,popen=popen)
    
    # TODO : compare the result 
    
    # TODO : save model 
    save_path = utils.pth_dir
    utils.snapshot(save_path,run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_avg_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_total_loss,
                'optimizer': optimizer.state_dict(),
            })