import torch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .DL_models import AE,VAE
from utils import cal_convTrans_shape , cal_conv_shape

global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Conv_AE(AE):
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim):
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
        self.diliat_ls =  diliat_ls
        de_diliat_ls = diliat_ls[::-1]
        de_channel_ls = channel_ls[::-1]
        de_padding_ls = padding_ls[::-1]
        
        # the basic element block of CNN
        self.Conv_block = lambda inChan,outChan,padding,diliation: nn.Sequential(
                    nn.Conv1d(inChan,outChan,4,stride=2,padding=padding,dilation=diliation),
                    nn.BatchNorm1d(outChan),
                    nn.ReLU())
        self.Deconv_block = lambda inChan,outChan,padding,diliation: nn.Sequential(
                    nn.ConvTranspose1d(inChan,outChan,4,stride=2,padding=padding,dilation=diliation),
                    nn.BatchNorm1d(outChan),
                    nn.ReLU())
        
        Encoder = nn.ModuleList(
            [self.Conv_block(channel_ls[i],channel_ls[i+1],padding_ls[i],diliat_ls[i]) for i in range(len(channel_ls)-1)]
        )        
        Decoder = nn.ModuleList(
            [self.Deconv_block(de_channel_ls[i],de_channel_ls[i+1],de_padding_ls[i],de_diliat_ls[i]) for i in range(len(channel_ls)-1)]
        )

        super(Conv_AE,self).__init__(Encoder,Decoder)
        self.teaching_rate = lambda x : 0   # just for compatibility
    def encode(self,X):
        if X.shape[1] == 100:
            X = X.transpose(1,2)  # to B*4*100
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
        return out.transpose(1,2)
    
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


class Conv_VAE(Conv_AE):
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim):
        # set up attr
        self.channel_ls = channel_ls
        self.padding_ls = padding_ls
        self.diliat_ls = diliat_ls
        self.latent_dim = latent_dim
        
        # compute the out dim
        self.out_length = int(self.compute_out_dim())
        self.out_dim = int(self.out_length * self.channel_ls[-1])
        # 
        super(Conv_VAE,self).__init__(channel_ls,padding_ls,diliat_ls,latent_dim)
        self.fc_mu = nn.Linear(self.out_dim,self.latent_dim)
        self.fc_sigma = nn.Linear(self.out_dim,self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim,self.out_dim)
        
    def compute_out_dim(self):
        """
        manually compute the final length of convolved sequence
        """
        L_in = 100
        for i in range(len(self.channel_ls)-1):
            L_out = cal_conv_shape(L_in,4,stride=2,padding=self.padding_ls[i],diliation=self.diliat_ls[i])
            L_in = L_out
        return L_out

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.Tensor) [B x D]
        """
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self,X,epoch=None,Y=None):
        batch_size = X.shape[0]
        # encode and flatten
        Z = self.encode(X)
        Z = Z.view(batch_size,self.out_dim)
        
        # project to N(µ, ∑)
        mu = self.fc_mu(Z)
        sigma = self.fc_sigma(Z)
        code = self.reparameterize(mu,sigma)
        
        # reshape decode
        re_code = self.fc_decode(code)
        re_code = re_code.view(batch_size,self.channel_ls[-1],self.out_length)
        out = self.decode(re_code)
        return out.transpose(1,2), mu , sigma
    
    def loss_function(self,recons,X,Y,mu,sigma,kld_weight):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        input = X if Y is None else Y
        log_var = sigma

        # Account for the minibatch samples from the dataset
        self.kld_weight = kld_weight
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)  # why is it negative ???
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'MSE':recons_loss, 'KLD':kld_loss}