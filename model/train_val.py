import os
import sys
import time
import logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
from model import DL_models
from model import reader
import torch
from torch import optim
from model.optim import ScheduledOptim , find_lr


def train(dataloader,model,optimizer,popen,lr=None):

    logger = logging.getLogger("VAE")
    loader_len = len(dataloader)       # number of iteration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    model.train()
    
    for idx,data in enumerate(dataloader):
        X,y = data       
        X = X.float().to(device)
        y = y.long().to(device)
        X.required_grad = True  # check !!!
        
        optimizer.zero_grad()
        
        if 'VAE' in popen.model_type:
            X_reconstruct, mu , sigma  = model(X)
            
            loss_dict = model.loss_function(X_reconstruct,X,mu,sigma,M_N=1)
            loss = loss_dict['loss']
        
        elif popen.model_type.split("_")[1] == "AE":
            out_seq = model(X)
            loss = model.loss_function(out_seq,X)
        
        # ======== grd clip and step ========
        
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=10)
        
        optimizer.step()
        
        # ====== update lr =======
        if lr is None:
            lr = optimizer.update_learning_rate()      # see model.optim.py
        
        # ======== verbose ========
        # record result 5 times for a epoch
        if idx % int(loader_len/5) == 0:
            if 'VAE' in popen.model_type:
                train_verbose = "{} / {}({:.1f}%): \t TOTAL:{} \t KLD:{} \t MSE:{} \t M_N:{} \t lr: {}".format(idx,loader_len,idx/loader_len*100,
                                                                                                    loss_dict['loss'].item(),
                                                                                                    loss_dict['KLD'].item(),
                                                                                                    loss_dict['MSE'].item(),
                                                                                                    model.kld_weight,
                                                                                                    lr)
            else:
                train_verbose = "{} / {}({:.1f}%): \t LOSS:{} \t lr: {}".format(idx,loader_len,idx/loader_len*100,
                                                                                                    loss.item(),
                                                                                                    lr)
            logger.info(train_verbose)
        
        torch.cuda.empty_cache()

def validate(dataloader,model,popen):

    logger = logging.getLogger("VAE")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    # ====== set up empty =====
    Total_loss = 0
    KLD_loss = 0
    MSE_loss = 0
    avg_acc = 0
    std_ls = []
    
    # ======== evaluate =======
    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(dataloader):
            X,y = data
            X = X.float().to(device)
            y = y.long().to(device)
            X.required_grad = True  # check !!!
            
            if "VAE" in popen.model_type:
                # forward and predict
                X_reconstruct, mu , sigma  = model(X)                    
                seq = model.reconstruct_seq(X)
                            
                # evaluate loss
                loss = model.loss_function(X_reconstruct,X,mu,sigma,M_N=1)
                Total_loss += loss['loss'].item()
                KLD_loss += loss['KLD'].item()
                MSE_loss += loss['MSE'].item()
                # std_ls.append(sigma)
            elif "AE" in popen.model_type:
                out_seq   = model(X)
                loss = model.loss_function(out_seq,X)
                seq = model.reconstruct_seq(out_seq,X)
                
                # average within batch
                avg_acc += torch.mean(X.mul(seq).sum(dim=2).sum(dim=1))  # the product of one-hot seq give identity    
            
            
            torch.cuda.empty_cache()
            
    avg_acc /= (idx+1)  # # average among batch
    
    # ======== verbose ========
    logger.info("\n===============================| start validation |===============================\n")
    if "VAE" in popen.model_type:
        val_verbose = "\t TOTAL:{:.7f} \n\t KLD:{:.7f} \n\t MSE:{:.7f} \n\t M_N:{} \n\t Avg_ACC: {}".format(Total_loss,
                                                                                                       KLD_loss,
                                                                                                       MSE_loss,
                                                                                                       model.kld_weight,
                                                                                                       avg_acc)
    else:
        val_verbose = "\t LOSS:{:.7f}  Avg_ACC: {}".format(loss,avg_acc)
    logger.info(val_verbose)
    # return these to save current performance
    return Total_loss,avg_acc
            
            
            
            