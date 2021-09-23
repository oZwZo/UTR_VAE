import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import copy
import torch
import train_val
import reader
import logomaker
from torch import nn
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from importlib import reload
from matplotlib import pyplot as plt

class Maxium_activation_patch(object):
    def __init__(self, popen, which_layer, n_patch=9, kfold_index=None, device_string='cpu'):
        self.popen = popen
        self.layer = which_layer
        self.popen.cuda_id = torch.device(device_string)
        self.n_patch = n_patch
        self.kfold_index = kfold_index
        self.total_stride = np.product(popen.stride[:self.layer])
        self.compute_reception_field()
        self.compute_virtual_pad()
    
    def load_indexed_dataloader(self,task):
        #        True or 'train_val'
        assert (self.popen.kfold_cv!=False) == (self.kfold_index != None), \
                "kfold CV should match with kfold index"
        
        self.popen.kfold_index = self.kfold_index
        self.popen.shuffle = False
        self.popen.pad_to = 57 if self.popen.cycle_set==None else 105
        
        if self.popen.cycle_set!=None:
            base_path = copy.copy(self.popen.split_like_paper)
            base_csv = copy.copy(self.popen.csv_path)
            
            if base_path is not None:
                assert task!=None , "task is not defined !"
                self.popen.split_like_paper = [path.replace('cycle', task) for path in base_path]
            else:
                self.popen.csv_path = base_csv.replace('cycle', task)
                
        return reader.get_dataloader(self.popen)
    
    def load_model(self):
        if self.kfold_index is not None:
            pth = self.popen.vae_pth_path.replace(".pth","_cv%s.pth"%self.kfold_index)
            self.popen.vae_pth_path = pth
        
        return utils.load_model(self.popen, None)
    
    def loading(self,task, which_set):
        model = self.load_model().to(self.popen.cuda_id)
        dataloader = self.load_indexed_dataloader(task)[which_set]
        self.df = dataloader.dataset.df
        return model, dataloader
    
    def extract_feature_map(self, task=None,which_set=0):
        """
        load trained model and unshuffled dataloader, make model forwarded
        Arg:
            task : str
            which_set : int, 0 : training set, 1 : val set, 2 : test set
        """
        model, dataloader= self.loading(task, which_set)
        
        feature_map = []
        X_ls = []
        Y_ls = []
        
        model.eval()
        with torch.no_grad():
            for Data in tqdm(dataloader):
                # iter each batch
                x,y = train_val.put_data_to_cuda(Data,self.popen,False)
                x = torch.transpose(x, 1, 2)
#                 X_ls.append(x.numpy())
                Y_ls.append(y.detach().cpu().numpy())
                
                for layer in model.soft_share.encoder[:self.layer]:
                    out = layer(x)
                    x = out
                feature_map.append(out.detach().cpu().numpy())
                
                torch.cuda.empty_cache()
        del model
        
        feature_map_l = np.concatenate( feature_map, axis=0)
        
#         self.X_ls = np.concatenate(X_ls, axis=0)
        self.Y_ls = np.concatenate(Y_ls, axis=0)

        print("activation map of layer |%d|"%self.layer,feature_map_l.shape)
#         print(self.X_ls.shape)
        print("Y : ",self.Y_ls.shape)
        self.feature_map = feature_map_l
        return feature_map_l
    
    def compute_reception_field(self):
        r = 1
        strides = self.popen.stride[:self.layer][::-1]

        for i,s in enumerate(strides):
            r = s*(r-1) + self.popen.kernel_size
        
        self.r = r
        self.strides = strides
        print('the reception filed is ', r)
    
    def compute_virtual_pad(self):
        v = []
        pd = self.popen.padding_ls[:self.layer]

        for i,p in enumerate(pd):
            v.append(p*np.product(self.popen.stride[:len(pd)-i-1]))
        
        self.virtual_pad = np.sum(v)
        print('the virtual pad is', self.virtual_pad)
    
    def retrieve_input_site(self, high_layer_site):
        """pad ?"""
        
        virtual_start = high_layer_site*self.total_stride - self.virtual_pad
        start = max(0, virtual_start)
        
        end = virtual_start + self.r
        return max(0,int(start)), int(end)

    def locate_MA_seq(self, channel, feature_map):
        """
        
        """
        if feature_map is None:
            feature_map = self.feature_map
            
        channel_feature = feature_map[:, channel,:]
        F0_ay = channel_feature.max(axis=-1)

        max_9_index = np.argpartition(F0_ay, -1*self.n_patch, axis=0)[-1*self.n_patch:]


        # a patch of sequences
        max_patch = self.df.utr.values[max_9_index]

        # find the location of maximally acitvated
        Conv4_sites = np.argmax(channel_feature[max_9_index],axis=1)
        
        mapped_input_sites = [self.retrieve_input_site(site) for site in Conv4_sites]
        
        # the mapped region of the sequences
        max_act_region = []
        for utr, (start, end) in zip(max_patch, mapped_input_sites):
            pad_gap = self.popen.pad_to - len(utr)
            field = utr[max(0, start-pad_gap): end-pad_gap]
            max_act_region.append(field)
            
#         print(mapped_input_sites)
        return max_act_region, max_9_index, max_patch
    
    def sequence_to_matrix(self, max_act_region):
        max_len = max([len(seq) for seq in max_act_region])
        
        M = np.zeros((max_len,4))
        
        for seq in max_act_region:
            oh_M = reader.one_hot(seq)
            M += np.concatenate([np.zeros((max_len - len(seq),4)), oh_M],axis=0)
            
        return M/self.n_patch
        
    def plot_sequence_logo(self, max_act_region, save_fig_path=None):
        """
        input : max_act_region ; list of str
        """
        assert np.all([seq != "" for seq in max_act_region])
        
        seq_logo_M = self.sequence_to_matrix(max_act_region)
        seq_logo_df = pd.DataFrame(seq_logo_M, columns=['A', 'C', 'G', 'T'])
    
        # plot
        MA_C = logomaker.Logo(seq_logo_df);
        MA_C.fig.gca().axis("off")
        
        # save
        if save_fig_path is not None:
            MA_C.fig.savefig(save_fig_path,transparent=True,dpi=600)
            save_dir = os.path.dirname(save_fig_path)
            
            try:
                self.save_dir
            except:
                # which is the first time we save
                self.save_dir = save_dir
                print('fig saved to',self.save_dir)
            plt.close(MA_C.fig)
    
    def fast_logo(self, channel, feature_map=None, save_fig_path=None):
        
        max_act_region, _, _ = self.locate_MA_seq(channel, feature_map)
        self.plot_sequence_logo(max_act_region, save_fig_path)
    
    def activation_density(self, channel, feature_map=None):
        if feature_map is None:
            feature_map = self.feature_map
            
        channel_feature = feature_map[:, channel,:]
        F0_ay = channel_feature.max(axis=-1)
        sns.kdeplot(F0_ay);
        print("num of max acti: %s"%np.sum(F0_ay==F0_ay.max()))
        