import os
import sys
import torch
from torch import nn
import numpy as np
from .CNN_models import Conv_AE,Conv_VAE,cal_conv_shape
from .Self_attention import self_attention

class TO_SEQ_TE(Conv_AE):
    # TODO: figure the `args` : kernel_sizef
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label):
        super(TO_SEQ_TE,self).__init__(channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size)
        self.num_label = num_label
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
            # nn.Dropout(0.5),
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
        if num_label == 1:
            self.cn_fn = nn.MSELoss(reduction='mean')
        else:
            self.cn_fn = nn.CrossEntropyLoss()
        
        # num_label
        self.predictor =nn.Sequential(
            nn.Conv1d(1,16,kernel_size=4,stride=2),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Conv1d(16,4,kernel_size=3,stride=2),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            
            nn.Conv1d(4,1,kernel_size=3,stride=2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            
            nn.Linear(31,64),
            # nn.Dropout(0.5),
            nn.ReLU(),
            
            nn.Linear(64,num_label)#,             if model report error pls look back here
            # nn.ReLU()
        )
        
        self.loss_dict_keys = ["Total","MSE","CrossEntropy"]
        
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
        
        if X.shape[1] != 4:
            X = X.transpose(1,2)
        mse_loss = self.mse_fn(X_reconst,X)     # compute mse loss here so that we don;t need X in the chimela loss
        return mse_loss,X_reconst, TE_pred  
    
    def compute_loss(self,out,X,Y,popen):
        """
        ALL chimela loss should only take 3 arguments : out , Y and lambda 
        Total Loss =  lambda_0 * MSE_Loss + lambda_1 * CrossEntropy_Loss
        """
        Lambda = popen.chimerla_weight

        mse_loss ,X_reconst, TE_pred = out        
        if self.num_label == 1:
            TE_true = Y
        else:
            TE_true = Y.squeeze().long()
        
        
        ce_loss = self.cn_fn(TE_pred,TE_true)
        
        total_loss = Lambda[0]*mse_loss + Lambda[1]*ce_loss
        
        return {"Total":total_loss,"MSE":mse_loss,"CrossEntropy":ce_loss}
    
    def compute_acc(self,out,X,Y,popen):
        """
        compute the accuracy of TE range class prediction
        """
        mse_loss ,X_reconst, TE_pred = out 
        TE_true = Y
        batch_size = TE_true.shape[0]
        epsilon = 0.3
        with torch.no_grad():
            if self.num_label == 1:
                pred = torch.sum(torch.abs(TE_pred - TE_true) < epsilon).item()
            else:
                pred = torch.sum(torch.argmax(TE_pred,dim=1) == TE_true).item()
            
        return pred / batch_size
    
    def compute_out_dim(self,kernel_size,L_in = 100):
        """
        manually compute the final length of convolved sequence
        """
        L_in = 100
        for i in range(len(self.channel_ls)-1):
            L_out = cal_conv_shape(L_in,kernel_size,stride=2,padding=self.padding_ls[i],diliation=self.diliat_ls[i])
            L_in = L_out
        return L_out
        
class TRANSFORMER_SEQ_TE(TO_SEQ_TE):
    # TODO: figure the `args` : kernel_sizef
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label):
        super(TRANSFORMER_SEQ_TE,self).__init__(channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label)
        
        de_channel_ls = [chann*2 for chann in self.channel_ls[::-1]]
        # shared function           # out of conv : B * Chan * out_len
        
        d_k = 64
        n_head = 1
        d_v = [channel_ls[-1]] + [64,128,128]    # list
        
        self.fc_hard_share = nn.Sequential(
            self_attention(d_v[0],n_head,d_k,d_v[1]),
            nn.ReLU(),
            # nn.BatchNorm1d(d_v[1]),
            
            self_attention(d_v[1]*n_head,n_head,d_k,d_v[2]), 
            nn.ReLU(),
            # nn.BatchNorm1d(d_v[2]),
            
            self_attention(d_v[2]*n_head,n_head,d_k,d_v[3]), 
            nn.ReLU(),
            # nn.BatchNorm1d(d_v[3])
        ) 
        
        # two linear transform to two task
        self.fc_to_dec = nn.Linear(n_head*d_v[3],de_channel_ls[0])
        self.fc_to_pre = nn.Linear(n_head*d_v[3],48)   # i.e.  B* len * 128 *n_head -> B* len * 32
        
        # num_label
        # input : B* len * 32
        # TODO : predictor list 
        self.predictor =nn.Sequential(
            nn.Conv1d(48,32,kernel_size=4,stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32,16,kernel_size=4,stride=1),
            # nn.Dropout(0.3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Conv1d(16,4,kernel_size=4,stride=1),      # out : len : 15.0
            nn.BatchNorm1d(4),
            nn.ReLU())
        
        self.predictor_fc =nn.Sequential(
            nn.Linear(15*4,128),
            # nn.Dropout(0.4),
            nn.ReLU(),
            
            nn.Linear(128,num_label),
            nn.ReLU()
        )
        
        self.loss_dict_keys = ["Total","MSE","CrossEntropy"]
        
    def forward(self,X):
        batch_size= X.shape[0]
        
        Z = self.encode(X)
         
        # reshape 
        Z_flat = Z.view(batch_size,self.out_len,-1)   # B * out_len * channel_ls[-1]
        
        # transform Z throgh the hard share fc 
        Z_trans = self.fc_hard_share(Z_flat)          # B * out_len * n_head * dv_ls [-1] 
        
        # head as task
        # Z_to_dec = Z_trans[:,:,:,0]
        # Z_to_pred = Z_trans[:,:,:,1]   
        
        # linear transform to sub-task 
        Z_to_dec = self.fc_to_dec(Z_trans).transpose(1,2)
        Z_to_pred = self.fc_to_pre(Z_trans).transpose(1,2)
        
        # reconstruction task
        Z_to_dec = Z_to_dec
        X_reconst = self.decode(Z_to_dec)
        
        # prediction taskz
        TE_conv_out = self.predictor(Z_to_pred).view(batch_size,-1)
        TE_pred = self.predictor_fc(TE_conv_out)
        
        if X.shape[1] != 4:
            X = X.transpose(1,2)
        mse_loss = self.mse_fn(X_reconst,X)     # compute mse loss here so that we don;t need X in the chimela loss
        return mse_loss,X_reconst, TE_pred  

class TRANSFORMER_SEQ_RL(TRANSFORMER_SEQ_TE):
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label):
        super(TRANSFORMER_SEQ_RL,self).__init__(channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label)
        self.predictor_fc =nn.Sequential(
            nn.Linear(12*4,128),
            # nn.Dropout(0.4),
            nn.ReLU(),
            
            nn.Linear(128,num_label),
        )
        self.regression_loss = nn.MSELoss(reduction='mean')
        self.loss_dict_keys = ["Total","MSE","RegLoss"]
        
    def chimela_loss(self,out,Y,Lambda):
        """
        ALL chimela loss should only take 3 arguments : out , Y and lambda 
        Total Loss =  lambda_0 * MSE_Loss + lambda_1 * CrossEntropy_Loss
        """
        mse_loss ,X_reconst, TE_pred = out        
        TE_true = Y
        
        ce_loss = self.regression_loss(TE_pred,TE_true)
        
        total_loss = Lambda[0]*mse_loss + Lambda[1]*ce_loss
        
        return {"Total":total_loss,"MSE":mse_loss,"RegLoss":ce_loss}
    
    def compute_acc(self,out,Y):
        """
        compute the accuracy of TE range class prediction
        """
        
        mse_loss ,X_reconst, TE_pred = out 
        TE_true = Y
        batch_size = TE_true.shape[0]
        
        
        
        with torch.no_grad():
            # number that are
            pred = torch.sum(torch.abs(TE_pred - TE_true) < epsilon).item()
            
        return pred / batch_size
    


class TWO_TASK_AT(nn.Module):
    def __init__(self,VAE_latent_dim,Linear_chann_ls,num_label,TE_chann_ls=None,SS_chann_ls=None,dropout_rate=0.2):
        """
        the downstream model that hands over the VAE encoder and predict the TE-score, secondary structure loop number
        Arguments:
        ...VAE_latent_dim:
        ...Linear_chann_ls:
        ...num_label  :
        ...nSS_num_label :
        """
        super(TWO_TASK_AT,self).__init__()
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
        
        self.TE_at = nn.ModuleDict({"Wq":nn.Linear(1,64),
                                    "Wk":nn.Linear(1,64)})  # have transform of value !!
        
        self.SS_at = nn.ModuleDict({"Wq":nn.Linear(1,64),
                                    "Wk":nn.Linear(1,64)})
        
        
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
        
        self.Cn_loss = nn.CrossEntropyLoss(reduction = 'mean')   #
        self.MSE_loss = nn.MSELoss(reduction = 'mean')           #            
        
        self.loss_dict_keys = ["Total","TE", "Loop" , "Match"]
        
    def linear_block(self,in_Chan,out_Chan,dropout_rate=None):
        """
        building block func to define dose network
        """
        if dropout_rate is None:
            dropout_rate = self.dropout_rate
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
        # dv = next(W_dict['Wv'].parameters()).shape[0]
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
        TE_input,attention = self.attention_forward(self.TE_at,share_output)
        for layer in self.TE_dense:
            TE_input = layer(TE_input)
        TE_out = self.TE_out_fc(TE_input)
        
        
        # aux task 2 : SS num prediction , contineous
        SS_input,attention2 = self.attention_forward(self.SS_at,share_output)
        for layer in self.SS_dense:
            SS_input = layer(SS_input)
        SS_out = self.SS_out_fc(SS_input)
        
        return TE_out,SS_out
    
    def chimela_loss(self,X_pred,Y,chimerla_weight):
        """
        loss function of multi-task
        """
        
        assert len(chimerla_weight)==3
         
        Lamb1,Lamb2,alpha = chimerla_weight
        assert (alpha >0 ) & (alpha<1)
        # model out : (TE_out : B*5, SS_out : B*5)
        TE_pred = X_pred[0]
        loop_pred = X_pred[1][:,:-1]
        match_pred = X_pred[1][:,-1]
        
        # True values  Y: B*6
        TE_true = Y[:,0].long()
        loop_true = Y[:,1:-1].float()
        match_true = Y[:,-1].float()
        
        # 3 type of loss
        TE_loss = self.Cn_loss(TE_pred,TE_true)
        loop_loss = self.MSE_loss(loop_pred,loop_true)
        match_loss = self.MSE_loss(match_pred,match_true)
        
        # LOSS = 
        Total_loss = Lamb1*TE_loss + Lamb2*((1-alpha)*loop_loss + alpha*match_loss)
        
        return {"Total":Total_loss,"TE": TE_loss, "Loop":loop_loss , "Match":match_loss}
    
    
    def compute_acc(self,out,Y):
        """
        all compute_acc takes 2 arguments out,Y
        This function only compute the accuracy of TE prediction
        """
        TE_pred = out[0]
        TE_true = Y[:,0].long()
        
        batch_size = Y.shape[0]
        
        with torch.no_grad():
            pred = torch.sum(torch.argmax(TE_pred,dim=1) == TE_true).item()
            
        return pred / batch_size

        
        
class Enc_n_Down(nn.Module):
    
    def __init__(self,pretrain_enc,downstream):
        super(Enc_n_Down,self).__init__()
        self.pretrain_model = pretrain_enc
        self.MTL_downstream = downstream
        try:
            self.loss_dict_keys = self.MTL_downstream.loss_dict_keys
        except:
            self.loss_dict_keys = None
        for param in self.pretrain_model.parameters():
                param.requires_grad = False
        
    def forward(self,X):
        code = self.pretrain_model.embed(X)
        out = self.MTL_downstream(code)
        
        return out
    
    def chimela_loss(self,X,Y,chimerla_weight):    
        return self.MTL_downstream.chimela_loss(X,Y,chimerla_weight)
    
    def compute_acc(self,out,Y):
        return self.MTL_downstream.compute_acc(out,Y)
        
