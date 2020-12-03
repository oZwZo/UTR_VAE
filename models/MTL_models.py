import os
import sys
import torch
from torch import nn
from .CNN_models import Conv_AE,Conv_VAE,cal_conv_shape

class TO_SEQ_TE(Conv_AE):
    # TODO: figure the `args` : kernel_sizef
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label):
        super(TO_SEQ_TE,self).__init__(channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size)
        
        de_diliat_ls = self.diliat_ls[::-1]
        de_channel_ls = [chann*2 for chann in self.channel_ls[::-1]]
        de_channel_ls[-1] = self.channel_ls[0]
        de_padding_ls = self.padding_ls[::-1]
        
        self.out_len = int(self.compute_out_dim(kernel_size))
        self.out_dim = self.out_len * channel_ls[-1]
        
        
        # shared function 
        self.fc_hard_share = nn.Sequential(
            nn.Linear(self.out_dim,1024),
            nn.Dropout(0.2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ) 
        
        # two linear transform to two task
        self.fc_to_dec = nn.Linear(512,self.out_dim*2)
        self.fc_to_pre = nn.Linear(512,256)
        
        self.decoder = nn.ModuleList(
            [self.Deconv_block(de_channel_ls[i],de_channel_ls[i+1],de_padding_ls[i],de_diliat_ls[i]) for i in range(len(channel_ls)-1)]
        )
        
        self.mse_fn = nn.MSELoss()
        self.cn_fn = nn.CrossEntropyLoss()
        
        # num_label
        self.predictor =nn.Sequential(
            nn.Conv1d(1,16,kernel_size=4,stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Conv1d(16,4,kernel_size=3,stride=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            
            nn.Conv1d(4,1,kernel_size=3,stride=2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            
            nn.Linear(31,64),
            nn.ReLU(),
            
            nn.Linear(64,num_label),
            nn.ReLU()
        )
        
    def forward(self,X):
        batch_size= X.shape[0]
        
        Z = self.encode(X)
         
        # reshape 
        Z_flat = Z.view(batch_size,self.out_dim)
        
        # transform Z throgh the hard share fc 
        Z_trans = self.fc_hard_share(Z_flat)
        
        # linear transform to sub-task
        Z_to_dec = self.fc_to_dec(Z_trans)
        Z_to_pred = self.fc_to_pre(Z_trans).unsqueeze(1)
        
        # reconstruction task
        Z_to_dec = Z_to_dec.view(batch_size,-1,self.out_len) 
        X_reconst = self.decode(Z_to_dec)
        
        # prediction taskz
        TE_pred = self.predictor(Z_to_pred).squeeze(1)
        return X_reconst, TE_pred  
    
    def chimela_loss(self,X_reconst,X_true,TE_pred,TE_true,Lambda):
        """
        Total Loss =  lambda_0 * MSE_Loss + lambda_1 * CrossEntropy_Loss
        """
        if X_true.shape[1] != 4:
            X_true = X_true.transpose(1,2)
        mse_loss = self.mse_fn(X_reconst,X_true)
        ce_loss = self.cn_fn(TE_pred,TE_true)
        
        total_loss = Lambda[0]*mse_loss + Lambda[1]*ce_loss
        
        return {"Total":total_loss,"MSE":mse_loss,"CrossEntropy":ce_loss}
    
    def compute_acc(self,TE_pred,TE_true):
        """
        compute the accuracy of TE range class prediction
        """
        batch_size = TE_true.shape[0]
        
        with torch.no_grad():
            pred = torch.sum(torch.argmax(TE_pred,dim=1) == TE_true).item()
            
        return pred / batch_size
    
    def compute_out_dim(self,kernel_size):
        """
        manually compute the final length of convolved sequence
        """
        L_in = 100
        for i in range(len(self.channel_ls)-1):
            L_out = cal_conv_shape(L_in,kernel_size,stride=2,padding=self.padding_ls[i],diliation=self.diliat_ls[i])
            L_in = L_out
        return L_out
        
        
    