import os
import sys
import torch
from torch import nn
import numpy as np
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
        
        
class TO_TE_nSS(nn.Module):
    def __init__(self,VAE_latent_dim,Linear_chann_ls,num_label,TE_chann_ls=None,SS_chann_ls=None,dropout_rate=0.2):
        """
        the downstream model that hands over the VAE encoder and predict the TE-score, secondary structure loop number
        Arguments:
        ...VAE_latent_dim:
        ...Linear_chann_ls:
        ...num_label  :
        ...nSS_num_label :
        """
        
        #TODO: args : pre-train ini ? , VAE_latent_dim 
        
        self.VAE_latent_dim = VAE_latent_dim
        self.Linear_chann_ls = [VAE_latent_dim] + Linear_chann_ls
        self.num_label = num_label
        self.dropout_rate = dropout_rate
        
        self.Norm_warmup = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(VAE_latent_dim)  # i.e. 64
        )
        
        # a stacked- deep dose network 
        self.hard_share_dense = nn.ModuleList(
            [self.linear_block(in_dim,out_dim) for in_dim,out_dim in zip(self.Linear_chann_ls[:-1],self.Linear_chann_ls[1:])]
            )                                            #   B*64 -> B*hs_out 
        
        # Attention : 
        # each header take charge of each aux task
        self.hs_out = Linear_chann_ls[-1]
        
        self.TE_at = nn.ModuleDict({"Wq":nn.Linear(self.host_out,64),
                                    "Wk":nn.Linear(self.host_out,64)})  # have transform of value !!
        
        self.SS_at = nn.ModuleDict({"Wq":nn.Linear(self.host_out,64),
                                    "Wk":nn.Linear(self.host_out,64)})
        
        
        # the network for TE range prediction
        if TE_chann_ls == None:
            TE_chann_ls = [128,64,32,16]
        TE_chann_ls = [self.hs_out] + TE_chann_ls
        self.TE_dense = nn.ModuleList([self.linear_block(in_dim,out_dim) for in_dim,out_dim in zip(TE_chann_ls[:-1],TE_chann_ls[1:])])
        self.TE_out_fc = nn.Linear(TE_chann_ls[-1],num_label[0])
        
        if SS_chann_ls == None:
            SS_chann_ls = [128,32]
        SS_chann_ls = [self.hs_out] + SS_chann_ls
        self.SS_dense = nn.ModuleList([self.linear_block(in_dim,out_dim,0.3) for in_dim,out_dim in zip(SS_chann_ls[:-1],SS_chann_ls[1:])])
        self.SS_out_fc = nn.Linear(SS_chann_ls[-1],num_label[1])
            
        
    def linear_block(self,in_Chan,out_Chan,dropout_rate=self.dropout_rate):
        """
        building block func to define dose network
        """
        block = nn.Sequential(
            nn.Linear(in_Chan,out_Chan),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.BatchNorm1d(out_Chan)
        )
        return block
    
    def attention_forward(self,W_dict,X):
        """
        compute the attention for each task
        """
        # some dimension 
        dk = next(W_dict['Wq'].parameters()).shape[0]
        dv = next(W_dict['Wv'].parameters()).shape[0]
        dk_sqrt = int(np.sqrt(dk))
        if len(X.shape)==2: 
            X = X.unsqueeze(2)      # make it 3 dimentional
        
        # computation part         X: B*X_dim*1
        query = W_dict['Wq'](X)     # B*X_dim*dk
        key = W_dict['Wk'](X)       # B*X_dim*dk
        #
        
        # attetnion
        sim_M = torch.bmm(query,key.transpose(1,2))/8  # B* X_dim * X_dim
        attention = torch.softmax(sim_M,dim=-1)
        # result
        result = torch.bmm(attention,X).squeeze(2)    # B*X_dim*1 -> B*X_dim
        
        return result,attention
        
    def forward(self,X):
        
        # forward the share part
        share_output = X
        for layer in self.hard_share_dense:
            share_output = layer(share_output)
        
        # aux task 1 : TE score rank prediction 0-4, catagorical
        TE_input = self.attention_forward(self.TE_at,share_output)
        for layer in self.TE_dense:
            TE_input = layer(TE_input)
        TE_out = self.TE_out_fc(TE_input)
        
        
        # aux task 2 : SS num prediction , contineous
        SS_input = self.attention_forward(self.SS_at,share_output)
        for layer in self.SS_dense:
            SS_input = layer(SS_input)
        SS_out = self.SS_out_fc(SS_input)
        
        return TE_out,SS_out
        
        
        
        

class self_attention(nn.Module):
    
    def __init__(self,X_dim,n_head,d_k,d_v):
        self.W_q = nn.Linear(self.host_out,n_head*d_k)    # makes \sqrt_{d_k} easier 
        self.W_k = nn.Linear(self.host_out,n_head*d_k)
        self.W_v = nn.Linear(self.host_out,n_head*d_v)     # the values dimension can be higher
        
        self.SS_at = nn.ModuleDict({"Wq":nn.Linear(self.host_out,64),
                                    "Wk":nn.Linear(self.host_out,64)})
    
    def self_at_forward(self,W_dict):
        
        # some dimension 
        dk = next(W_dict['Wq'].parameters()).shape[0]
        dv = next(W_dict['Wv'].parameters()).shape[0]
        dk_sqrt = int(np.sqrt(dk))
        
        query = W_dict['Wq'](X)     # B*X_dim*dk
        key = W_dict['Wk'](X)       # B* host_out -> B*64
        values = W_dict['Wv'](X)    # B* host_out -> B*128
        
        sim_M = torch.bmm(querys,keys.transpose(1,2))/8  # B* X_dim * X_dim
        attention = torch.softmax(sim_M,dim=-1)
        # result
        result = torch.bmm(attention,X).squeeze(2)    # B*X_dim*1 -> B*X_dim
        
        return result