import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F

class Conv_backbond(nn.Module):
    def __init__(self,conv_size,kernel_size=3,direction='encode',**kwargs):
        super(Conv_backbond,self).__init__()

        self.channels =  conv_size
        self.conv = nn.Conv1d if direction == 'encode' else nn.ConvTranspose1d
       
        self.block = lambda in_channel , out_channel: nn.Sequential(
           self.conv(in_channel,out_channel,kernel_size=kernel_size,**kwargs),
        #    nn.MaxPool1d(),
           nn.BatchNorm1d(out_channel),
           nn.LeakyReLU()
        )
       
        self.encoder = nn.ModuleList([self.block(i,j) for i,j in zip(self.channels[:-1],self.channels[1:])])

    #    self.weight_initialize()
    
    def _code(self,X):
        
        for block in self.encoder:
            X = block(X)
        return X
    
    def forward(self,X):
        X = self._code(X)
        return X
    
class AE(nn.Module):
    def __init__(self,encoder,decoder):
        super(AE,self).__init__()
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
        elif isinstance(model, nn.Conv1d):
            nn.init.kaiming_normal_(model.weight, nonlinearity='leaky_relu',)
        elif isinstance(model, nn.BatchNorm1d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)

    def forward(self,X):
        Z = self.encoder(X)
        X_prime = self.decoder(Z)
        return X_prime, Z

class VAE(nn.Module):
    def __init__(self,encoder,decoder,latent_dim,out_dim):
        super(VAE,self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        # encoder part
        self.Encoder = encoder
        self.out_dim = out_dim
        self.fc_mu = nn.Linear(out_dim,latent_dim)
        self.fc_sigma = nn.Linear(out_dim,latent_dim)

        # decoder part 
        self.Decoder = decoder

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
        elif isinstance(model, nn.Conv1d):
            nn.init.kaiming_normal_(model.weight, nonlinearity='leaky_relu',)
        elif isinstance(model, nn.BatchNorm1d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)

    def encode(self,X):
        """
        the encode part of the VAE, two fc layer to compute \mu and \sigma
        """
        Z = self.Encoder(X)
        Z = torch.flatten(Z,start_dim=1)
        mu = self.fc_mu(Z)
        sigma = self.fc_sigma(Z)   # the log std of variational mixture distribution
        return (mu,sigma)

    def decode(self,X):
        Z = self.Decoder(X)
        return Z

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

    def forward(self,X):
        mu,sigma = self.encode(X)
        Z = self.reparameterize(mu,sigma)
        X_reconst = self.decode(Z)
        return X_reconst, mu,sigma
    
    def reconstruct_seq(self,X):
        """
        for an original one-hot seq, return the reconstructed one-hot seq
        argmax was used to rebuilt
        """
        X_reconst,mu,sigma = self.forward(X)
        # make zero
        seq = torch.zeros_like(X)
        
        position = torch.argmax(X_reconst,dim=2)     # X_reconst : b*100*4
        
        for batch_idx in range(X.shape[0]):
            for i,j in enumerate(position[batch_idx]):
                seq[batch_idx,i,j.item()] = 1
                
        return seq
        
    
    def loss_function(self,
                      *args,
                      **kwargs):
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
        self.kld_weight = kld_weight
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'MSE':recons_loss, 'KLD':-kld_loss}



class LSTM_VAE(VAE):
    def __init__(self,input_size,hidden_size_enc,hidden_size_dec,num_layers,latent_dim,seq_in_dim,decode_type):
        
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.input_size = input_size
        self.num_layers = num_layers
        self.seq_in_dim = seq_in_dim
        self.decode_type = decode_type
        
        # the out_dim  will  flatten the cell_state of LSTM encoder output
        out_dim = num_layers*2*hidden_size_enc 
        latent_dim = latent_dim
        """
        Encoder hidden_state :  [num_layers*num_directions, batch, hidden_size]
        """
        Encoder = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size_enc,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        """
        Decoder take reparameterize code as hidden state, and start decoding
        hidden required : [3,batch,4]
        The input :  [batch, 1 , ???]
        The output : [batch, 100, 4]
        """
        Decoder = nn.LSTM(input_size=seq_in_dim,
                          hidden_size=hidden_size_dec,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=False)
        
        
       
        super(LSTM_VAE,self).__init__(encoder=Encoder,decoder=Decoder,out_dim=out_dim,latent_dim=latent_dim)
        
        self.fc_code2hs = nn.Linear(latent_dim,num_layers*hidden_size_dec)
        self.fc_code2seq = nn.Linear(latent_dim,seq_in_dim*100)    # seq_in_dim : int*100
        self.fc_out2word = nn.Linear(self.hidden_size_dec,input_size)   # hidden_size 
        
    def encode(self,X):
        """
        Cover the encode , for LSTM backbone , use the cell state as latent variables
        """
        Z,(hidden_state,cell_state) = self.Encoder(X)        # cell_state : [num_layers*num_directions, batch, hidden_size]
        flat_cell = torch.flatten(cell_state.transpose(0,1),start_dim=1) # -> [batch, num_layers*num_directions*hidden_size]
    
        # compute
        mu = self.fc_mu(flat_cell)
        sigma = self.fc_sigma(flat_cell)
        return mu, sigma
    
    def LSTM_seq_in(self,code,warmup=True):
        """
        take code as input seq
        """
        # transform reparameterized code to input seq
        batch_size = code.shape[0]
        code_to_seq = self.fc_code2seq(code)                  # [batch,latent_size] -> [batch,seq_in]
        in_seq = code_to_seq.view(batch_size,100,-1)       # -> [batch,100,seq_in/100]
        
        # warm up with the first 10 step
        hidden_state = None
        if warmup:
            _,hidden_state = self.Decoder(in_seq[:,:10,:])
        
        # LSTM reconstruct with warm up hidden state
        out,_ = self.Decoder(in_seq,hidden_state)           # out: [batch,100,hidden_size]
        
        # transform to 4 feature
        logit = self.fc_out2word(out)                       #logit:[batch,100,4]
        return logit

    def LSTM_cell_in(self,code,warmup=True):
        """
        take code as cell state
        """
        logit = []
        batch_size = code.shape[0]
        # initiate the seq generation chain
        start_don = torch.zeros([batch_size,1,self.latent_size])
        hidden_state = torch.zeros([self.num_layers,batch_size,self.hidden_size])
        if warmup:
            _ = start_don
            _,hidden_state = self.Decoder(_,hidden_state)
        
        out = start_don
        for i in range(100):
            out,hidden_state = self.Decoder(out,hidden_state)
            out = self.fc_out2word(out)
            logit.append(out)
        
        return logit
           
    def decode(self,code):
        if self.decode_type == 'seq':
            logit = self.LSTM_seq_in(code)
        else:        
            logit = self.LSTM_cell_in(code)
        return logit