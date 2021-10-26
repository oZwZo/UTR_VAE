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
from scipy import stats
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.manifold import TSNE

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
    
    def get_filter_param(self, model):
        Conv_layer = model.soft_share.encoder[self.layer-1]
        
        return next(Conv_layer[0][0].parameters()).detach().cpu().numpy()
    
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
            
        
        
        feature_map_l = np.concatenate( feature_map, axis=0)
        
#         self.X_ls = np.concatenate(X_ls, axis=0)
        self.Y_ls = np.concatenate(Y_ls, axis=0)

        print("activation map of layer |%d|"%self.layer,feature_map_l.shape)
#         print(self.X_ls.shape)
        print("Y : ",self.Y_ls.shape)
        self.feature_map = feature_map_l
        
        self.filters = self.get_filter_param(model)
        
        del model
        
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

    def locate_MA_seq(self, channel, feature_map=None):
        """
        
        """
        if feature_map is None:
            feature_map = self.feature_map
            
        channel_feature = feature_map[:, channel,:]
        F0_ay = channel_feature.max(axis=-1)

        max_n_index = np.argpartition(F0_ay, -1*self.n_patch, axis=0)[-1*self.n_patch:]


        # a patch of sequences
        max_patch = self.df.utr.values[max_n_index]

        # find the location of maximally acitvated
        Conv4_sites = np.argmax(channel_feature[max_n_index],axis=1)
        
        mapped_input_sites = [self.retrieve_input_site(site) for site in Conv4_sites]
        
        # the mapped region of the sequences
        max_act_region = []
        for utr, (start, end) in zip(max_patch, mapped_input_sites):
            

            pad_gap = self.popen.pad_to - len(utr)
            field = utr[max(0, start-pad_gap): end-pad_gap]
            
            max_act_region.append(field)
            
#         print(mapped_input_sites)
        
        full_field = [len(field)==self.r for field in max_act_region]
        
        return np.array(max_act_region)[full_field], max_n_index[full_field], max_patch[full_field]
    
    def sequence_to_matrix(self, max_act_region, weight=None, transformation='counts'):
        
        assert np.all([seq != "" for seq in max_act_region])
        
        max_len = max([len(seq) for seq in max_act_region])
        
        M = np.zeros((max_len,4))
        if weight is None:
            weight = np.ones((self.n_patch,))
        for seq, w in zip(max_act_region, weight):
            oh_M = reader.one_hot(seq)*w
            M += np.concatenate([np.zeros((max_len - len(seq),4)), oh_M],axis=0)
            

        seq_logo_df = pd.DataFrame(M, columns=['A', 'C', 'G', 'T'])
        if transformation!='counts':
            seq_logo_df = logomaker.transform_matrix(seq_logo_df, from_type='counts', to_type=transformation)
            
        return seq_logo_df
        
    def plot_sequence_logo(self, seq_logo_M,  save_fig_path=None):
        """
        input : max_act_region ; list of str
        """
        
        
        
        # plot
        plt.figure(dpi=300)
        MA_C = logomaker.Logo(seq_logo_df);
        ax = MA_C.fig.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
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
    
    def fast_logo(self, channel, feature_map=None, n_patch=None, transformation='information', save_fig_path=None):
        
        if n_patch is not None:
            self.n_patch = n_patch
        max_act_region, _, _ = self.locate_MA_seq(channel, feature_map)
        M = self.sequence_to_matrix(max_act_region)
        F0_ay, (spr,pr) = self.activation_density(channel, False, False)
        self.plot_sequence_logo(M,  transformation=transformation, save_fig_path=save_fig_path)
        ax=plt.gca()
        ax.set_title("Fileter {} : $r =$ {}".format(channel, round(spr[0], 3)), fontsize=35)
         
    
    def activation_density(self, channel, to_print=True, to_plot=True, feature_map=None, **kwargs):
        if feature_map is None:
            feature_map = self.feature_map
            
        channel_feature = feature_map[:, channel,:]
        F0_ay = channel_feature.max(axis=-1)
        if to_plot:
            sns.kdeplot(F0_ay, **kwargs);
        if to_print:
            print("num of max acti: %s"%np.sum(F0_ay==F0_ay.max()))
        
        spr = stats.spearmanr(F0_ay, self.Y_ls.flatten())
        pr = stats.pearsonr(F0_ay, self.Y_ls.flatten())
        return F0_ay, (spr,pr)
    
    def within_patch_clustering(self, channel, n_clusters, n_patch=None , to_plot=True,**kwargs):
        """
        return:
            subclusters : list, list of patch (alignments)
        """
        self.n_patch = n_patch
        act_patch, _, _ = self.locate_MA_seq(channel)
        flatten_seq = np.stack([reader.one_hot(seq).flatten() for seq in act_patch])
        print("the shape of sequence matrix {}".format(flatten_seq.shape))
        
        # clsutering
        cluster_index = cluster.KMeans(n_clusters=n_clusters).fit_predict(flatten_seq)
        # down
        if to_plot:
            tsne = TSNE(metric='cosine', square_distances=True).fit(flatten_seq)
            self.tsne=tsne
            plt.figure(figsize=(6,5),dpi=150)
    #         sns.set_theme(style='ticks', palette='viridis')
            scatter_args = {"palette":'viridis'}
            scatter_args.update(kwargs)
            sns.scatterplot(x=tsne.embedding_[:,0], y=tsne.embedding_[:,1], 
                            hue=cluster_index, **scatter_args);
        
        return act_patch, flatten_seq, cluster_index
    
    def activation_overview(self, **kwargs):
        channel_spearman = []
       
        fig, ax = plt.subplots(1,1, figsize=(10,5), dpi=300)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        n_channel = self.popen.channel_ls[self.layer]
        for i in tqdm(range(n_channel)):
            act, (spr, pr)  = self.activation_density(i, to_print=False, feature_map=None, **kwargs);
            channel_spearman.append(spr[0])
        return fig, ax, np.array(channel_spearman)
    
    def matrix_to_seq(self, matrix):
        """
        return the sequence with maximum weight at each site
        """
        max_act_seq = ''
        assert matrix.shape[1] == 4
        for i in matrix.argmax(axis=1):
            max_act_seq += ['A', 'C', 'G', 'T'][i]
            
        return max_act_seq
    
    def save_as_meme_format(self,channels:list, save_path, filer_prefix='filter', transformation='probability'):
        """
        Save the position weight matrix as the meme-suite acceptale minimal motif format
        """
        
        with open(save_path, 'w') as f:
            f.write("MEME version 5.4.1\n\n")
            f.write("ALPHABET= ACGT\n\n")
            f.write("strands: + -\n\n")
            f.write("Background letter frequencies\n")
            f.write("A 0.25 C 0.25 G 0.25 T 0.25\n")

            for cc in channels:
                
                try:
                    region, index, patches = self.locate_MA_seq(channel=cc)
                    M = self.sequence_to_matrix(region, transformation=transformation);
                    f.write('\n')
                    f.write(f"MOTIF {filer_prefix}_{cc}\n")
                    seq_len = len(region[0])
                    f.write(f"letter-probability matrix: alength= 4 w= {seq_len} \n")
                    for line in M.values:
                        f.write(" "+line.__str__()[1:-1]+'\n')
                except ValueError:
                    continue
                    
            f.close()
            print('writed')
        return None
    
    def get_input_grad(test_X,index,model):
    
        # process X

        sampled_X = test_X[index].unsqueeze(0)
        sampled_X.requires_grad = True
        # forward
        model.train()
        sampled_out = model(sampled_X)

        # auto grad part
        external_grad = torch.ones_like(sampled_out)
        sampled_out.backward(gradient=external_grad,retain_graph=True) # define \frac{d out}{ }
        print(sampled_out)
        return sampled_X,sampled_X.grad

def extract_max_seq_pattern(condition, n_clusters=6, n_patch=3000):
    pattern = []
    channel_source = []
    for channel in np.where(condition)[0]:

        try:
            act_patch, flatten_seq, cluster_index = self.within_patch_clustering(channel, n_clusters=n_clusters, to_plot=False,n_patch=n_patch)
        except ValueError:
            continue

        for i in range(n_clusters):
            sub_cluster = act_patch[cluster_index==i]
            if len(sub_cluster) > 0:
                matrix = self.sequence_to_matrix(sub_cluster)
                pattern.append(self.matrix_to_seq(matrix))
                channel_source.append(channel)
        return pattern, channel_source

def sum_occurrance(df, pattern):
    pattern_occurance = []
    for p in pattern:
        pattern_occurance.append(np.sum([(p in utr) for utr in df.seq.values]))
    return np.array(pattern_occurance)

def generate_scramble_index(size  , N_1):
    scramble_index=np.zeros((size,))

    while scramble_index.sum() < N_1:    
        num_ = int(N_1 - scramble_index.sum())
        randindex = np.random.randint(0, size, size=(num_,))

        for i in randindex:
            scramble_index[i] = 1
    return scramble_index    

