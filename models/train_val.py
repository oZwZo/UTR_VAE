import os
import sys
import time
import numpy as np
import logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import torch
from torch import optim
from models.ScheduleOptimizer import ScheduledOptim , find_lr


def train(dataloader,model,optimizer,popen,epoch,lr=None):

    logger = logging.getLogger("VAE")
    loader_len = len(dataloader)       # number of iteration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.cuda(popen.cuda_id)
    model.teacher_forcing = popen.teacher_forcing     # turn on teacher_forcing
    model.train()
    
    for idx,data in enumerate(dataloader):
        X,y = data       
        X = X.float().cuda(popen.cuda_id)
        Y = y.float().cuda(popen.cuda_id)
        X.required_grad = True  # check !!!
        # Y = Y if X.shape == Y.shape else None  # for mask data
        
        optimizer.zero_grad()
        
        if 'VAE' in popen.model_type:
            X_reconstruct, mu , sigma  = model(X,epoch)
            loss_dict = model.loss_function(X_reconstruct,X,Y,mu,sigma,kld_weight=popen.kld_weight)
            loss = loss_dict['loss']
            Avg_acc = model.compute_acc(X_reconstruct,X,Y)
            
        elif 'AE' in popen.model_type:
            out_seq = model(X=X,epoch=epoch,Y=Y)
            loss = model.loss_function(out_seq,X,Y)
        
        elif popen.dataset == 'MTL':
            out = model(X)
            loss_dict = model.chimela_loss(out,Y,popen.chimerla_weight)
            acc = model.compute_acc(out,Y)
            loss = loss_dict['Total']
            if type(popen.te_net_l2) == int:
                loss += popen.te_net_l2*torch.sum(next(model.predictor.parameters())**2)
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
                train_verbose = "{:5d} / {:5d} ({:.1f}%): \t TOTAL:{:.9f} \t KLD:{:.9f} \t MSE:{:.9f} \t M_N:{} \t lr: {:.9f} \t Avg_ACC: {}".format(idx,loader_len,idx/loader_len*100,
                                                                                                    loss_dict['loss'].item(),
                                                                                                    loss_dict['KLD'].item(),
                                                                                                    loss_dict['MSE'].item(),
                                                                                                    model.kld_weight,
                                                                                                    lr,
                                                                                                    Avg_acc)
            elif popen.dataset == 'MTL':
                loss_dict_keys = list(loss_dict.keys())

                train_verbose = "{:5d} / {:5d} ({:.1f}%): \t Avg_ACC: {:.7f} \t lr: {:.9f}"
                verbose_args = [idx,loader_len,idx/loader_len*100,acc,lr]
                for key in loss_dict_keys:
                    train_verbose += "\t %s:{:.9f}"%key
                    verbose_args.append(loss_dict[key])
                    
                train_verbose = train_verbose.format(*verbose_args)                
                
            
            else:
                train_verbose = "{:5d} / {:5d} ({:.1f}%): \t LOSS:{:.9f} \t lr: {:.9f} \t teaching_rate: {:.9f} ".format(idx,loader_len,idx/loader_len*100,
                                                                                                    loss.item(),
                                                                                                    lr,
                                                                                                    model.teaching_rate(epoch))
            logger.info(train_verbose)
        with torch.cuda.device(popen.cuda_id):
            torch.cuda.empty_cache()

def validate(dataloader,model,popen,epoch):

    logger = logging.getLogger("VAE")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.cuda(popen.cuda_id)
    model.teacher_forcing = False    # turn off teacher_forcing
    
    # ====== set up empty =====
    if model.loss_dict_keys is not None:
        loss_verbose = {key:0 for key in model.loss_dict_keys}
    
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
            X = X.float().cuda(popen.cuda_id)
            Y = y.float().cuda(popen.cuda_id)
            X.required_grad = True  # check !!!
            # Y = Y if X.shape == Y.shape else None  # for mask data
            
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
                
            elif popen.dataset == 'MTL':
                Y = Y.squeeze().long()
                out = model(X)
                loss_dict = model.chimela_loss(out,Y,popen.chimerla_weight)
                
                
                for key in model.loss_dict_keys:
                    loss_verbose[key]  += loss_dict[key].item()
                
                # TODO : compute acc of classification
                
                acc_ls.append(model.compute_acc(out,Y))
                
            with torch.cuda.device(popen.cuda_id):  
                torch.cuda.empty_cache()
            
    avg_acc = np.mean(acc_ls)  # # average among batch
    
    # ======== verbose ========
    logger.info("\n===============================| start validation |===============================\n")
    if "VAE" in popen.model_type:
        val_verbose = "\t  TOTAL:{:.7f} \t KLD:{:.7f} \t MSE:{:.7f} \t M_N:{} \t Avg_ACC: {}".format(Total_loss,
                                                                                                       KLD_loss,
                                                                                                       MSE_loss,
                                                                                                       model.kld_weight,
                                                                                                       avg_acc)
    elif popen.dataset == 'MTL':
        
        # this part try to scale verbose to whatever it takes in `loss_dict_keys`
        
        val_verbose = "\t chimerla_weight: {} \t Avg_ACC: {:.7f}"
        if type(popen.chimerla_weight) == list:
            chimela_weight = popen.chimerla_weight[0]/popen.chimerla_weight[1] if popen.chimerla_weight[1] != 0 else popen.chimerla_weight[0]
        elif type(popen.chimerla_weight) == float:
            chimela_weight = popen.chimerla_weight
        verbose_args = [chimela_weight,avg_acc]
        
        for key in model.loss_dict_keys:
            val_verbose += " \t  "  + key + ":{:.7f}" 
            verbose_args.append(loss_verbose[key]/idx)
        
        val_verbose = val_verbose.format(*verbose_args)
        
    else:
        val_verbose = "\t LOSS:{:.7f}  Avg_ACC: {}".format(loss,avg_acc)
    logger.info(val_verbose)
    # return these to save current performance
    return Total_loss,avg_acc
            
            
            
            