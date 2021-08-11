import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas a pd
import logomaker

class Maxium_activation_patch(object):
    def __init__(self, popen,n_patch=9):
        self.popen = popen
        self.n_patch = n_patch
        
        self.total_stride = np.product(popen.stride)
        self.compute_reception_field()
        self.compute_virtual_pad()
        
    def compute_reception_field(self):
        r = 1
        strides = self.popen.stride[::-1]

        for i,s in enumerate(strides):
            r = s*(r-1) + k
        
        self.r = r
        print('the reception filed is ', r)
    
    def compute_virtual_pad(self):
        v = []
        pd = self.popen.padding_ls

        for i,p in enumerate(pd):
            v.append(p*np.product(self.popen.stride[:len(pd)-i-1]))
        
        print('the virtual pad is', virtual_pad)
        self.virtual_pad = np.sum(v)

    def retrieve_input_site(self, high_layer_site,pad=7):

        virtual_start = high_layer_site*self.total_stride - self.virtual_pad
        start = max(0, virtual_start)

        end = virtual_start + self.r
        return max(0,int(start)-pad), int(end)-pad

    def locate_MA_seq(self, channel, train_feature_map,df):
        
        channel_feature = train_feature_map[:, channel,:]
        F0_ay = channel_feature.max(axis=-1)

        max_9_index = np.argpartition(F0_ay, -1*self.n_patch, axis=0)[-1*self.n_patch:]


        # a patch of sequences
        max_patch = df.utr[max_9_index]

        # find the location of maximally acitvated
        Conv4_sites = np.argmax(channel_feature[max_9_index],axis=1)
        
        mapped_input_sites = [self.retrieve_input_site(site) for site in Conv4_sites]
        
        # the mapped region of the sequences
        max_act_region = [utr[slice(*start_end)] for utr, start_end in zip(max_patch, mapped_input_sites)]

        print(mapped_input_sites)
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