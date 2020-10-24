import torch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .DL_models import AE,VAE

global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Conv_AE(AE):
    def __init__(self,channel_ls,padding_ls,latent_dim,seq_in_dim):
        """
        Conv1D backbone Auto-encoder that encode 100bp sequence data and reconstruct them.
        Symmentric design of `Encoder` and `Decoder`
        Fixed kernel size=4 for each layer
        Arguments:
        ...channel_ls : a list of channel for Conv1D, the longer the channel, the deeper the network
        ...padding_ls : list of padding for each Conv layer to ensure we reconstruct extactly 100 bp back 
        ...latent_dim : for VAE, of no use now. For the compatibility.
        ...seq_in_dim : for VAE, of no use now. For the compatibility.
        """
        self.channel_ls = channel_ls
        self.padding_ls = padding_ls
        de_channel_ls = channel_ls[::-1]
        de_padding_ls = padding_ls[::-1]
        
        # the basic element block of CNN
        self.Conv_block = lambda inChan,outChan,padding: nn.Sequential(
                    nn.Conv1d(inChan,outChan,4,stride=2,padding=padding),
                    nn.BatchNorm1d(outChan),
                    nn.LeakyReLU())
        self.Deconv_block = lambda inChan,outChan,padding: nn.Sequential(
                    nn.ConvTranspose1d(inChan,outChan,4,stride=2,padding=padding),
                    nn.BatchNorm1d(outChan),
                    nn.LeakyReLU())
        
        Encoder = nn.ModuleList(
            [self.Conv_block(channel_ls[i],channel_ls[i+1],padding_ls[i]) for i in range(len(channel_ls)-1)]
        )
        
        Decoder = nn.ModuleList(
            [self.Deconv_block(de_channel_ls[i],de_channel_ls[i+1],de_padding_ls[i]) for i in range(len(channel_ls)-1)]
        )

        super(Conv_AE,self).__init__(Encoder,Decoder)
        self.teaching_rate = lambda x : 0   # just for compatibility
    def encode(self,X):
        if X.shape[1] == 100:
            X = X.tranpose(1,2)  # to B*4*100
        Z = X
        for model in self.encoder:
            Z = model(Z)
        return Z
    
    def decode(self,Z):
        out = Z
        for model in self.decoder:
            out = model(out)
        return out
            
    def forward(self, X,epoch=None,Y=None):
        Z = self.encode(X)
        out = self.decode(Z)
        return out
    
    def loss_function(self,out,X,Y=None):
        """
        compute the MSE loss of reconstrcuted X and origin X
        """
        if Y is None:
            Y = X

        loss_fn = nn.MSELoss(reduction='mean')
        return loss_fn(out,Y)
    
    def compute_acc(self,out,X,Y=None):
        """
        compute the reconstruction accuracy
        """
        if Y is None:
            Y = X
        batch_size = X.shape[0]       # B*100*4
        
        true_max=torch.argmax(Y,dim=2)
        recon_max=torch.argmax(out,dim=2)
        return torch.sum(true_max == recon_max).item() /batch_size

        
    def reconstruct_seq(self,out_seq,X):
        seq = torch.zeros_like(X)
        position = torch.argmax(out_seq,dim=2)     # X_reconst : b*100*4
        
        for batch_idx in range(X.shape[0]):
            for i,j in enumerate(position[batch_idx]):
                seq[batch_idx,i,j.item()] = 1     
        return seq
        