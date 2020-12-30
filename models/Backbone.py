import torch 
from torch import nn
import torch.nn.functional as F

class Conv1d_block(nn.Module):
    """
    the Convolution backbone define by a list of convolution block
    """
    def __init__(self,Channel_ls,kernel_size,stride,padding_ls=None,diliation_ls=None):
        """
        Argument
            Channel_ls : list, [int] , channel for each conv layer
            kernel_size : int
            stride :  list , [int]
            padding_ls :   list , [int]
            diliation_ls : list , [int]
        """
        super(Conv1d_block,self).__init__()
        ### property
        self.Channel_ls = Channel_ls
        self.kernel_size = kernel_size
        self.stride = stride
        if padding_ls is None:
            self.padding_ls = [0] * (len(Channel_ls) - 1)
        else:
            assert len(padding_ls) == len(Channel_ls) - 1
            self.padding_ls = padding_ls
        if diliation_ls is None:
            self.diliation_ls = [1] * (len(Channel_ls) - 1)
        else:
            assert len(diliation_ls) == len(Channel_ls) - 1
            self.diliation_ls = diliation_ls
        
        self.encoder = nn.ModuleList(
            #                   in_C         out_C           padding            diliation
            [self.Conv_block(Channel_ls[i],Channel_ls[i+1],self.padding_ls[i],self.diliation_ls[i],self.stride[i]) for i in range(len(self.padding_ls))]
        )
        
    def Conv_block(self,in_Chan,out_Chan,padding,dilation,stride): 
        
        block = nn.Sequential(
                nn.Conv1d(in_Chan,out_Chan,self.kernel_size,stride,padding,dilation),
                nn.BatchNorm1d(out_Chan),
                nn.ReLU())
        
        return block
    
    def forward(self,x):
        out = x.transpose(1,2)
        for block in self.encoder:
            out = block(out)
        return out
    
    def forward_stage(self,x,stage):
        """
        return the activation of each stage for exchanging information
        """
        assert stage < len(self.encodre)
        
        out = self.encoder[stage](x)
        return out

    def cal_out_shape(self,L_in=100,padding=0,diliation=1,stride=2):
        """
        For convolution 1D encoding , compute the final length 
        """
        L_out = 1+ (L_in + 2*padding -diliation*(self.kernel_size-1) -1)/stride
        return L_out
    
    def last_out_len(self,L_in=100):
        for i in range(len(self.padding_ls)):
            padding = self.padding_ls[i]
            diliation = self.diliation_ls[i]
            stride = self.stride[i]
            L_in = self.cal_out_shape(L_in,padding,diliation,stride)
        assert int(L_in) == L_in , "convolution out shape is not int"
        return int(L_in)
    
class ConvTranspose1d_block(Conv1d_block):
    """
    the Convolution transpose backbone define by a list of convolution block
    """
    def __init__(self,Channel_ls,kernel_size,stride,padding_ls=None,diliation_ls=None):
        Channel_ls = Channel_ls[::-1]
        stride = stride[::-1]
        padding_ls =  padding_ls[::-1] if padding_ls  is not None else  [0] * (len(Channel_ls) - 1)
        diliation_ls =  diliation_ls[::-1] if diliation_ls  is not None else  [1] * (len(Channel_ls) - 1)
        super(ConvTranspose1d_block,self).__init__(Channel_ls,kernel_size,stride,padding_ls,diliation_ls)
        
    def Conv_block(self,in_Chan,out_Chan,padding,dilation,stride): 
        """
        replace `Conv1d` with `ConvTranspose1d`
        """
        block = nn.Sequential(
                nn.ConvTranspose1d(in_Chan,out_Chan,self.kernel_size,stride,padding,dilation),
                nn.BatchNorm1d(out_Chan),
                nn.ReLU())
        
        return block
    
    def cal_out_shape(self,L_in,padding=0,diliation=1,stride=1,out_padding=0):
        #                  L_in=100,padding=0,diliation=1,stride=2
        """
        For convolution Transpose 1D decoding , compute the final length
        """
        L_out = (L_in -1 )*stride + diliation*(self.kernel_size -1 )+1-2*padding + out_padding 
        return L_out


class linear_block(nn.Module):
    def __init__(self,in_Chan,out_Chan,dropout_rate=0.2):
        """
        building block func to define dose network
        """
        super(linear_block,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_Chan,out_Chan),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.BatchNorm1d(out_Chan)
        )
    def forward(self,x):
        return self.block(x)
    
class backbone_model(nn.Module):
    def __init__(self,conv_args):
        """
        the most bottle model which define a soft-sharing convolution block some forward method 
        """
        super(backbone_model,self).__init__()
        Channel_ls,kernel_size,stride,padding_ls,diliation_ls = conv_args
        L_in = 100 if kernel_size %2 == 0 else 101
        # model
        self.soft_share = Conv1d_block(Channel_ls,kernel_size,stride,padding_ls,diliation_ls)
        # property
        self.stage = list(range(len(Channel_ls)-1))
        self.out_length = self.soft_share.last_out_len(L_in)
        self.out_dim = self.soft_share.last_out_len(L_in)*Channel_ls[-1]
    
    def forward_stage(self,X,stage):
        return self.soft_share.forward_stage(X,stage)
    
    def forward_tower(self,Z):
        """
        Each new backbone model should re-write the `forward_tower` method
        """
        return Z
    
    def forward(self,X):
        Z = self.soft_share(X)
        out = self.forward_tower(Z)
        return out
    
class RL_regressor(backbone_model):
    
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2):
        """
        backbone for RL regressor task  ,the same soft share should be used among task
        Arguments:
            conv_args: (Channel_ls,kernel_size,stride,padding_ls,diliation_ls)
        """
        super(RL_regressor,self).__init__(conv_args)
        
        #  ------- architecture -------
        self.tower = linear_block(in_Chan=self.out_dim,out_Chan=tower_width,dropout_rate=dropout_rate)
        self.fc_out = nn.Linear(tower_width,1)
        
        #     ----- task specific -----
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.task_name = 'RL_regression'
        self.loss_dict_keys = ['Total']
        
    def forward_tower(self,Z):
        # flatten
        batch_size = Z.shape[0]
        Z_flat = Z.view(batch_size,-1)
        # tower part
        Z_to_out = self.tower(Z_flat)
        out = self.fc_out(Z_to_out)
        return out
    
    def squeeze_out_Y(self,out,Y):
        # ------ squeeze ------
        if len(Y.shape) == 2:
            Y = Y.squeeze(1)
        if len(out.shape) == 2:
            out = out.squeeze(1)
        
        assert Y.shape == out.shape
        return out,Y
    
    def compute_acc(self,out,Y,epsilon=0.3):
        out,Y = self.squeeze_out_Y(out,Y)
        # error smaller than epsilon
        with torch.no_grad():
            acc = torch.sum(torch.abs(Y-out) < epsilon).item() / Y.shape[0]
        return acc
    
    def compute_loss(self,out,Y,popen):
        out,Y = self.squeeze_out_Y(out,Y)
        loss = self.loss_fn(out,Y)
        return {"Total":loss}
    
class Reconstruction(backbone_model):
    def __init__(self,conv_args,VAE=False,latent_dim=80):
        """
        the sequence reconstruction backbone
        """
        self.VAE = VAE
        self.latent_dim = latent_dim
        super(Reconstruction,self).__ini__(conv_args)
        
        #  ------- architecture -------
        self.tower = ConvTranspose1d_block(*conv_args)
        
        #  ---- VAE only ----
        if self.VAE == True:
            self.fc_mu = nn.Linear(self.out_dim,self.latent_dim)
            self.fc_sigma = nn.Linear(self.out_dim,self.latent_dim)
            self.fc_decode = nn.Linear(self.latent_dim,self.out_dim)
        
        #  ------- task specific -------
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.task_name = 'Reconstruction'
        self.loss_dict_keys = ['Total', 'MSE', 'KLD'] if self.VAE else ['Total']
        
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
    
    def forward_tower(self,Z):
        batch_size = Z.shape[0]
        re_code = Z
        if self.VAE:
             # project to N(µ, ∑)
            Z_flat = Z.view(batch_size,-1)
            mu = self.fc_mu(Z_flat)
            sigma = self.fc_sigma
            code = self.reparameterize(mu,sigma)
            
            re_code = self.fc_decode(code)
            re_code = re_code.view(batch_size,self.channel_ls[-1],self.out_length)    
        # decode
        recon_X = self.tower(re_code)
        
        return recon_X,mu,sigma
    
    def forward(self,X):
        Z = self.soft_share(X)
        out = self.forward_tower(Z)
        recons_loss =self.loss_fn(out[0], X)
        return out,recons_loss
    
    def compute_loss(self,out,Y,popen):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        (recons,mu,sigma),recons_loss = out
        log_var = sigma

        # Account for the minibatch samples from the dataset
        
        loss =recons_loss
        loss_dict = {'Total': loss}
        
        if self.VAE:
            self.kld_weight = popen.kld_weight
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)  # why is it negative ???
            loss = recons_loss + popen.kld_weight * kld_loss
            loss_dict = {'Total': loss, 'MSE':recons_loss, 'KLD':kld_loss}
        return loss_dict
    
    def compute_acc(self,out,X,Y=None):
        """
        compute the reconstruction accuracy
        """
        (recons,mu,sigma),recons_loss = out
        if Y is None:
            Y = X
        batch_size = X.shape[0]       # B*4*100
        true_max=torch.argmax(Y,dim=1)
        recon_max=torch.argmax(recons,dim=1)
        return torch.sum(true_max == recon_max).item() /batch_size

class Motif_detection(backbone_model):
    def __init__(self,conv_args,motifs:list,tower_width=40):
        """
        can detect different motif
        """
        super(Motif_detection,self).__init__(conv_args)
        self.num_labels = len(motifs)
        self.task_name = [seq + "_detection" for seq in motifs]
        
        # architecture
        self.tower_to_out = linear_block(in_Chan=self.out_dim,out_Chan=tower_width)
        self.fc_out = nn.Sequential(
            nn.Linear(tower_width,self.num_labels),
            nn.Sigmoid()
        )
        
        # task specific
        self.loss_fn = nn.BCELoss(reduction='mean')
        self.loss_dict_keys = ['Total']
        
    def forward_tower(self,Z):
        """
        predicting several motifs at the same times
        """
        batch_size = Z.shape[0]
        Z_flat = Z.view(batch_size,-1)
        
        inter = self.tower_to_out(Z_flat)
        out = self.fc_out(inter)         # B * num_labels
        return out
    
    def compute_loss(self,X,Y,popen):
        loss = 0
        
        for i in range(X.shape[1]):
            x = X[:,i]
            y = Y[:,i].long()
            loss += self.loss_fn(x,y)
        return {"Total":loss}
    
    def compute_acc(self,X,Y,threshold=0.5):
        
        decision = X > threshold
        decision = decision.long()
        Y = Y.long()
        
        acc = torch.sum(decision == Y) / (X.shape[0]*X.shape[1])
        return acc