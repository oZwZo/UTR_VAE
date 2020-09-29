import torch
import numpy as np
import pandas as pd
from torch import nn


class Conv_backbond(nn.Module):
    def __init__(self,conv_size,kernel_size=3):
       super(Conv_backbond,self).__init__()

       self.channels = [4] + conv_size
       
       self.block = lambda in_channel , out_channel: nn.Sequential(
           nn.Conv1d(in_channel,out_channel,kernel_size=kernel_size,stride=1),
        #    nn.MaxPool1d(),
           nn.BatchNorm1d(out_channel),
           nn.LeakyReLU()
        )
       
       self.encoder = nn.ModuleList([self.block(i,j) for i,j in zip(self.channels[:-1],self.channels[1:])])

       self.weight_initialize()
    
    def _code(self,X):
        for block in self.encoder:
            X = block(X)
        return X
    
    def forward(self,X):
        X = self._code(X)
        return X

class LSTM_backbond(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,**lstmargs):
       super(LSTM_backbond,self).__init__()
       
    self.encoder = nn.LSTM(input_size,hidden_size,num_layers=2,batch_first=True,bidirectional=True,**lstmargs)
    
    def _code(self,X):
        X = self.encoder(X)
        return X
    
    def forward(self,X):
        X = self._code(X)
        return X

class AE(nn.Module):
    def __init__(self,encoder,decoder):
        super(AE_framework,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def weight_initialize(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu',)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self,X):
        Z = self.encoder(X)
        X_prime = self.decoder(Z)
        return X_prime, Z

class VAE(nn.Module):
    def __init__(self,encoder,decoder,latent_dim,hidden_dim):
        super(AE_framework,self).__init__()
        
        # encoder part
        self.encoder = encoder
        self.fc_mu = nn.Linear(out_dim,latent_dim)
        self.fc_sigma = nn.Linear(out_dim,latent_dim)

        # decoder part 
        self.decoder = decoder

        # param init
        self.apply(self._weight_initialize)

    def _weight_initialize(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu',)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def encode(self,X) -> tuple[Tensor]:
        Z = self.encoder(X)
        Z = torch.flatten(Z,start_dim=1)
        mu = self.fc_mu(Z)
        sigma = self.fc_sigma(Z)
        return (mu,sigma)

    def decode(self,X) -> Tensor:
        Z = self.decoder(X)
        return result

    def reparameterize(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self,X):
        mu,sigma = self.encoder(X)
        Z = self.reparameterize(mu,sigma)
        X_reconst = self.decoder(Z)
        return X_reconst, mu,sigma
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
