import os
import sys
import time
import numpy as np
import pandas as pd
import logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import torch
from torch import optim
from models.ScheduleOptimizer import ScheduledOptim , find_lr
from models.loss import Dynamic_Task_Priority as DTP


def train(dataloader,model,optimizer,popen,epoch,lr=None):

    logger = logging.getLogger("VAE")
    loader_len = len(dataloader)       # number of iteration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.cuda(popen.cuda_id)
    # model.teacher_forcing = popen.teacher_forcing     # turn on teacher_forcing
    model.train()
    
    for idx,data in enumerate(dataloader):
        X,y = data       
        X = X.float().cuda(popen.cuda_id)
        Y = y.float().cuda(popen.cuda_id)
        X.required_grad = True  # check !!!
        # Y = Y if X.shape == Y.shape else None  # for mask data
        
        optimizer.zero_grad()
        
        out = model(X)
        # TODO : debug here
        loss_dict = model.compute_loss(out,X,Y,popen)
        loss = loss_dict['Total']
        acc_dict = model.compute_acc(out,X,Y,popen)
        loss_dict.update(acc_dict) # adding the acc ditc into loss dict
         
        loss.backward()        
        optimizer.step()
        
        # ====== update lr =======
        if (lr is None) & (type(optimizer) == ScheduledOptim):
            lr = optimizer.update_learning_rate()      # see model.optim.py
            loss_dict["lr"]=lr
        elif popen.optimizer == 'Adam':
            lr = optimizer.param_groups[0]['lr']
        if popen.loss_schema != 'constant':
            popen.chimerla_weight = popen.loss_schedualer._update(loss_dict)
            for t in popen.tasks:
                loss_dict[popen.loss_schema+"_wt_"+t] = popen.chimerla_weight[t]
        
       
        # ======== verbose ========
        # record result 5 times for a epoch
        if idx % int(loader_len/5) == 0:
            
            
            loss_dict_keys = list(loss_dict.keys())
        
            train_verbose = "{:5d} / {:5d} ({:.1f}%):"
            verbose_args = [idx,loader_len,idx/loader_len*100]
            for key in loss_dict_keys:
                train_verbose += "\t %s:{:.7f}"%key
                verbose_args.append(loss_dict[key])
                
            train_verbose = train_verbose.format(*verbose_args)                         
        
            logger.info(train_verbose)
        with torch.cuda.device(popen.cuda_id):
            torch.cuda.empty_cache()

def validate(dataloader,model,popen,epoch):

    logger = logging.getLogger("VAE")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.cuda(popen.cuda_id)
    model.teacher_forcing = False    # turn off teacher_forcing
    
    # ====== set up empty =====
    # model.loss_dict_keys = ['RL_loss', 'Recons_loss', 'Motif_loss', 'Total', 'RL_Acc', 'Recons_Acc', 'Motif_Acc']
    verbose_list=[]
    
    # ======== evaluate =======
    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(dataloader):
            X,y = data       
            X = X.float().cuda(popen.cuda_id)
            Y = y.float().cuda(popen.cuda_id)
            X.required_grad = True  # check !!!
            # Y = Y if X.shape == Y.shape else None  # for mask data
            
            out = model(X)
            # TODO : debug here
            loss_dict = model.compute_loss(out,X,Y,popen)
            loss = loss_dict['Total']
            acc_dict = model.compute_acc(out,X,Y,popen)
            loss_dict.update(acc_dict)
            
            loss_dict = utils.clean_value_dict(loss_dict)  # convert possible torch to single item
            verbose_list.append(loss_dict)
               
            with torch.cuda.device(popen.cuda_id):  
                torch.cuda.empty_cache()
            
          # # average among batch
    
    # ======== verbose ========
    
    verbose_df = pd.json_normalize(verbose_list)
    
    logger.info("\n===============================| start validation |===============================\n")
        
    val_verbose = ""
    verbose_args = []
    for key in verbose_df.columns:
        val_verbose += "\t %s:{:.7f}"%key
        verbose_args.append(verbose_df[key].mean())
        
    val_verbose = val_verbose.format(*verbose_args)                         

    logger.info(val_verbose)
    
    # what avg acc return : mean of  RL_Acc , Recons_Acc, Motif_Acc
    acc_col = list(acc_dict.keys())
    Avg_acc = np.mean(verbose_df.loc[:,acc_col].mean(axis=0))  
    
    # return these to save current performance
    return verbose_df['Total'].mean(),Avg_acc
            
            
            
            