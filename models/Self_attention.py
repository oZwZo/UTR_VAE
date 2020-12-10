import os
import sys
import torch
from torch import nn
import numpy as np

class self_attention(nn.Module):
    
    def __init__(self,in_channel,n_head,d_k,d_v): 
        super(self_attention,self).__init__()
        self.n_head=n_head
        self.d_k = d_k
        self.d_v = d_v
        self.W_dict = nn.ModuleDict({"Wq" : nn.Linear(in_channel,n_head*d_k),
                                     "Wk" : nn.Linear(in_channel,n_head*d_k),
                                     "Wv" : nn.Linear(in_channel,n_head*d_v)})

    def forward(self,X):
        
        # some dimension 
        dk_sqrt = int(np.sqrt(self.d_k))
        
        querys = self.W_dict['Wq'](X)     # B*X_dim*dk
        keys = self.W_dict['Wk'](X)       # B* hs_out -> B*64
        values = self.W_dict['Wv'](X)    # B* hs_out -> B*128
        
        sim_M = torch.bmm(querys,keys.transpose(1,2))/8  # B* X_dim * X_dim
        attention = torch.softmax(sim_M,dim=-1)
        # result
        result = torch.bmm(attention,values).squeeze(2)    # B*X_dim*1 -> B*X_dim
        
        return result
     