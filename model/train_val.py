import os
import sys
import time
sys.path.append(os.dirname(os.dirname(__file__)))
import utils
from model import DL_models
from model import reader
import torch
from torch import optim
from model.optim import ScheduledOptim , find_lr


def train(dataloader,model,optimizer,lr=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    model.train()
    
    for idx,data in enumerate(dataloader):
        X,y = data       
        X = X.float().to(device)
        y = y.long().to(device)
        X.required_grad = True  # check !!!
        
        optimizer.zero_grad()
        
        X_reconstruct, mu , sigma  = model(X)
        
        loss = model.loss_function(X_reconstruct,X,mu,sigma,K_N=1)['loss']
        
        loss.backward()
        
        # gradient clipping
        _ = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=10)
        
        optimizer.step()
        
        if lr is None:
                lr = optimizer.update_learning_rate()      # see model.optim.py
        
        torch.cuda.empty_cache()

def validate(dataloader,model):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    # some of the metric
    Total_loss = 0
    KLD_loss = 0
    MSE_loss = 0
    avg_acc = 0
    std_ls = []
    
    model.eval()
    with torch.no_grad():
        
        for idx,data in enumerate(dataloader):
            X,y = data
            X = X.float().to(device)
            y = y.long().to(device)
            X.required_grad = True  # check !!!
                        
            X_reconstruct, mu , sigma  = model(X)                    
            seq = model.reconstruct_seq(X)
                        
            # evaluate loss
            loss = model.loss_function(X_reconstruct,X,mu,sigma,K_N=1)
            Total_loss += loss['loss'].item()
            KLD_loss += loss['KLD'].item()
            MSE_loss += loss['MSE'].item()
            
            # average within batch
            avg_acc += torch.mean(X.mul(seq).sum(dim=2).sum(dim=1))  # the product of one-hot seq give identity
            
            std_ls.append(sigma)
            
            torch.cuda.empty_cache()
            
    avg_acc /= (idx+1)  # # average among batch
    
    return Total_loss, KLD_loss, MSE_loss, avg_acc, std_ls
            
            
            
            
            