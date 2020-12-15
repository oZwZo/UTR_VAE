import os
import sys
import torch
from torch import nn
import numpy as np
from .CNN_models import Conv_AE,Conv_VAE,cal_conv_shape
from .Self_attention import self_attention

class Baseline(nn.Module):
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label,loss_fn):
        """
        Conv - Dense Framework DL Regressor to predict ribosome load (rl) from 5' UTR sequence
        Arguments:
        ...channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label : parameter to define the Conv layers
        ...loss_fn : regression loss function `MSELoss` or `MyHingeLoss`
        """
        super(Baseline,self).__init__()
        #         ==<<|  properties  |>>==
        self.kernel_size = kernel_size
        self.loss_fn = loss_fn
        self.channel_ls = channel_ls
        self.padding_ls = padding_ls
        self.diliat_ls =  diliat_ls
        self.loss_dict_keys = ['Total','MAE','RMSE']
        # the basic element block of CNN
        
        #         ==<<|  Conv layers  |>>==
        # 3 layers in the 5'UTR paper
        self.encoder = nn.ModuleList(
            [self.Conv_block(channel_ls[i],channel_ls[i+1],padding_ls[i],diliat_ls[i]) for i in range(len(channel_ls)-1)]
        )        
        
        # compute the output shape of conv layer
        self.out_len = int(self.compute_out_dim(kernel_size))
        self.out_dim = self.out_len * channel_ls[-1]
        
        #         ==<<|  Dense layers  |>>==
        self.fc_out = nn.Sequential( 
            nn.Linear(self.out_dim,40),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(40,num_label)
        )
        
        self.define_loss()  
        self.acc_hinge = MyHingeLoss(None)
        
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
    
    def Conv_block(self,inChan,outChan,padding,diliation,stride=1): 
        """
        Building Block of stack conv
        """
        net = nn.Sequential(
                    nn.Conv1d(inChan,outChan,self.kernel_size,stride=1,padding=padding,dilation=diliation),
                    # nn.BatchNorm1d(outChan),
                    nn.ReLU())
        return net
    
    def define_loss(self):
        if self.loss_fn in ['mse','MSE']:
            self.regression_loss = nn.MSELoss(reduction='mean')
        else:
            self.regression_loss = MyHingeLoss(reduction='mean')
        
    
    def encode(self,X):
        if X.shape[1] == 100:
            X = X.transpose(1,2)  # to B*4*100
        Z = X
        for model in self.encoder:
            Z = model(Z)
        return Z
    
    def forward(self,X):
        """
        Conv -> flatten -> Dense
        """
        batch_size = X.shape[0]
        
        z = self.encode(X)
        z_flat = z.view(batch_size,-1)
        out = self.fc_out(z_flat)       # no activation for the last layer
        
        return out
        
        
    def compute_acc(self,out,Y):
        """
        for this regression task, accuracy  is the percentage that prediction error < epsilon (Lambda) 
        """
        
        batch_size = Y.shape[0]
        with torch.no_grad():
            loss = self.acc_hinge(out,Y,self.Lambda)
            n_inrange = (loss==0).squeeze().sum().item()
        
        return n_inrange / batch_size
       
        
    def chimela_loss(self,out,Y,Lambda):
        """
        it's termed `chimela_loss` to keep compatability with MTL MODELS (the same dataset was used)
        `chiemela_loss` requires three input : 
        ...out:
        ...Y:
        ...Lambda : which is the epsilon of hinge loss if possible
        """
        self.Lambda = Lambda
        batch_size = Y.shape[0]
        
        #  Hinge or MSE
        loss = self.regression_loss(out,Y,Lambda).squeeze()
        with torch.no_grad():
            MAE = torch.abs(out-Y).sum().item()                            # Mean Absolute Error
            RMSE = torch.sqrt( torch.sum((out-Y)**2) / batch_size).item()  # Root Mean Square Error
        return {"Total":loss,"MAE":MAE,"RMSE":RMSE}
    
    def compute_out_dim(self,kernel_size,L_in = 100):
        """
        manually compute the final length of convolved sequence
        """
        L_in = 100
        for i in range(len(self.channel_ls)-1):
            L_out = cal_conv_shape(L_in,kernel_size,stride=1,padding=self.padding_ls[i],diliation=self.diliat_ls[i])
            L_in = L_out
        return L_out
    
class MyHingeLoss(torch.nn.Module):

    def __init__(self,reduction='mean'):
        self.reduction = reduction
        super(MyHingeLoss, self).__init__()

    def forward(self, out, Y,epsilon):
        """
        linear version hinge regression loss 
        """
        hinge_loss = torch.abs(out-Y) - epsilon
        hinge_loss[hinge_loss < 0] = 0
        if self.reduction  == 'mean':
            hinge_loss = torch.mean(hinge_loss)
        return hinge_loss