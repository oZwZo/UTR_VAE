import os
import sys
import time
import numpy as np
import logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
from models import DL_models
from models import reader
import torch
from torch import optim
from models.ScheduleOptimizer import ScheduledOptim , find_lr


def train(dataloader,model,optimizer,popen,epoch,lr=None):

    logger = logging.getLogger("VAE")
    loader_len = len(dataloader)       # number of iteration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.teacher_forcing = popen.teacher_forcing     # turn on teacher_forcing
    model.train()
    
    for idx,data in enumerate(dataloader):
        X,y = data       
        X = X.float().to(device)
        Y = y.float().to(device)
        X.required_grad = True  # check !!!
        Y = Y if X.shape == Y.shape else None  # for mask data
        
        optimizer.zero_grad()
        
        if 'VAE' in popen.model_type:
            X_reconstruct, mu , sigma  = model(X,epoch)
            loss_dict = model.loss_function(X_reconstruct,X,Y,mu,sigma,kld_weight=popen.kld_weight)
            loss = loss_dict['loss']
            Avg_acc = model.compute_acc(X_reconstruct,X,Y)
            
        elif popen.model_type.split("_")[1] == "AE":
            out_seq = model(X=X,epoch=epoch,Y=Y)
            loss = model.loss_function(out_seq,X,Y)
        
        # ======== grd clip and step ========
        
        loss.backward()
        
        if "LSTM" in popen.model_type:
            _ = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=5)
        
        optimizer.step()
        
        # ====== update lr =======
        if (lr is None) & (type(optimizer) == ScheduledOptim):
            lr = optimizer.update_learning_rate()      # see model.optim.py
        elif popen.optimizer == 'Adam':
            lr = optimizer.param_groups[0]['lr']
       
        # ======== verbose ========
        # record result 5 times for a epoch
        if idx % int(loader_len/5) == 0:
            if 'VAE' in popen.model_type:
                train_verbose = "{:5d} / {:5d} ({:.1f}%): \t TOTAL:{:.9f} \t KLD:{:.9f} \t MSE:{:.9f} \t M_N:{:3d} \t lr: {:.9f} \t Avg_ACC: {}".format(idx,loader_len,idx/loader_len*100,
                                                                                                    loss_dict['loss'].item(),
                                                                                                    loss_dict['KLD'].item(),
                                                                                                    loss_dict['MSE'].item(),
                                                                                                    model.kld_weight,
                                                                                                    lr,
                                                                                                    Avg_acc)
            else:
                train_verbose = "{:5d} / {:5d} ({:.1f}%): \t LOSS:{:.9f} \t lr: {:.9f} \t teaching_rate: {:.9f} ".format(idx,loader_len,idx/loader_len*100,
                                                                                                    loss.item(),
                                                                                                    lr,
                                                                                                    model.teaching_rate(epoch))
            logger.info(train_verbose)
        
        torch.cuda.empty_cache()

def validate(dataloader,model,popen,epoch):

    logger = logging.getLogger("VAE")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.teacher_forcing = False    # turn off teacher_forcing
    
    # ====== set up empty =====
    Total_loss = 0
    KLD_loss = 0
    MSE_loss = 0
    acc_ls = []
    std_ls = []
    
    # ======== evaluate =======
    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(dataloader):
            X,y = data       
            X = X.float().to(device)
            Y = y.float().to(device)
            X.required_grad = True  # check !!!
            Y = Y if X.shape == Y.shape else None  # for mask data
            
            if "VAE" in popen.model_type:
                # forward and predict
                X_reconstruct, mu , sigma  = model(X,epoch)                    
                acc_ls.append(model.compute_acc(X_reconstruct,X,Y))
                            
                # evaluate loss
                loss = model.loss_function(X_reconstruct,X,Y,mu,sigma,kld_weight=popen.kld_weight)
                Total_loss += loss['loss'].item()
                KLD_loss += loss['KLD'].item()
                MSE_loss += loss['MSE'].item()
                # std_ls.append(sigma)
            elif "AE" in popen.model_type:
                out_seq = model(X,epoch)
                loss = model.loss_function(out_seq,X,Y)
            
                Total_loss += loss.item()
                # average within batch
                acc_ls.append(model.compute_acc(out_seq,X,Y))  # the product of one-hot seq give identity  
                
            torch.cuda.empty_cache()
            
    avg_acc = np.mean(acc_ls)  # # average among batch
    
    # ======== verbose ========
    logger.info("\n===============================| start validation |===============================\n")
    if "VAE" in popen.model_type:
        val_verbose = "\t  Total:{:.7f} \n\t KLD:{:.7f} \n\t MSE:{:.7f} \n\t M_N:{} \n\t Avg_ACC: {}".format(Total_loss,
                                                                                                       KLD_loss,
                                                                                                       MSE_loss,
                                                                                                       model.kld_weight,
                                                                                                       avg_acc)
    else:
        val_verbose = "\t LOSS:{:.7f}  Avg_ACC: {}".format(loss,avg_acc)
    logger.info(val_verbose)
    # return these to save current performance
    return Total_loss,avg_acc
            
            
            
            