import os
import sys
import torch
from torch import nn
import numpy as np
from .CNN_models import Conv_AE,Conv_VAE,cal_conv_shape
from .Self_attention import self_attention

class Baseline(nn.Module):
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label):
        super(Baseline,self).__init__()
        self.channel_ls = channel_ls
        self.padding_ls = padding_ls
        self.diliat_ls =  diliat_ls
        
        # the basic element block of CNN
        self.Conv_block = lambda inChan,outChan,padding,diliation: nn.Sequential(
                    nn.Conv1d(inChan,outChan,kernel_size,stride=1,padding=padding,dilation=diliation),
                    nn.BatchNorm1d(outChan),
                    nn.ReLU())

        self.encoder = nn.ModuleList(
            [self.Conv_block(channel_ls[i],channel_ls[i+1],padding_ls[i],diliat_ls[i]) for i in range(len(channel_ls)-1)]
        )        
        
        self.fc_out = nn.Sequential(
            ## TODO : what ???
            nn.Linear(3,40),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(40,num_label)
        )
        
        self.regression_hinge = MyHingeLoss()
        
        self.apply(self.weight_initialize)
        
    def weight_initialize(self, model):
        if type(model) in [nn.Linear]:
        	nn.init.xavier_uniform_(model.weight)
        	nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
        elif isinstance(model, nn.Conv1d):
            nn.init.kaiming_normal_(model.weight, nonlinearity='leaky_relu',)
        elif isinstance(model, nn.BatchNorm1d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)

    def encode(self,X):
        if X.shape[1] == 100:
            X = X.transpose(1,2)  # to B*4*100
        Z = X
        for model in self.encoder:
            Z = model(Z)
        return Z
    
    def forward(self,X):
        
        batch_size = X.shape[0]
        
        z = self.encode(X)
        
        z_flat = z.view(batch_size,-1)
        
        out = self.fc_out(z_flat)
        
        return out
        
        
    def compute_acc(self,out,Y):
        """
        for this regression task, accuracy  is the percentage that prediction error < epsilon (Lambda) 
        """
        
        batch_size = Y.shape[0]
        with torch.no_grad():
            loss = self.chimela_loss(out,Y,self.Lambda)
            n_inrange = torch.sum(loss == 0).item()
        
        return n_inrange / batch_size
       
        
    def chimela_loss(self,out,Y,Lamba):
        
        self.Lambda = Lambda
        
        loss = self.regression_hinge(out,Y,Lamba)
        
        return {"Total":loss}
    
class MyHingeLoss(torch.nn.Module):

    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, out, Y,epsilon):
        """
        linear version hinge regression loss 
        """
        hinge_loss = torch.abs(out-Y) - epsilon
        hinge_loss[hinge_loss < 0] = 0
        
        return hinge_loss