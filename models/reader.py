import torch
from torch.utils.data import  DataLoader, Dataset ,random_split
from sklearn.model_selection import KFold,train_test_split
import numpy as np
import pandas as pd
import copy
import os
import sys
import json
from torch import nn
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import Seq_one_hot,read_UTR_csv,read_label

global script_dir
global data_dir

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),"machine_configure.json"),'r') as f:
    config = json.load(f)

script_dir = config['script_dir']
data_dir = config['data_dir']

def one_hot(seq,complementary=False):
    """
    one_hot encoding on sequence
    complementary: encode nucleatide into complementary one
    """
    # setting
    seq = list(seq.replace("U","T"))
    seq_len = len(seq)
    complementary = -1 if complementary else 1
    # compose dict
    keys = ['A', 'C', 'G', 'T'][::complementary]
    oh_dict = {keys[i]:i for i in range(4)}
    # array
    oh_array = np.zeros((seq_len,4))
    for i,C in enumerate(seq):
        try:
            oh_array[i,oh_dict[C]]=1
        except:
            continue      # for nucleotide that are not in A C G T   
    return oh_array 

class MTL_enc_dataset(Dataset):
    def __init__(self,DF,pad_to=100,aux_columns=None,input_col=None):
        """
        the dataset for Multi-task learning, will return sequence in one-hot encoded version, together with some auxilary task label
        arguments:
        ...csv_path: abs path of csv file, column `utr` should be in the csv
        ...pad_to : maximum length of the sequences
        ...columns : list  contains what  axuslary task label will be 
        """
        self.pad_to = pad_to
        self.df = DF     # read Df
        self.seqs = self.df.utr.values       # take out all the sequence in DF
        self.columns = aux_columns
        self.input_col = input_col
                
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,index):
        seq = self.seqs[index]        # sequence: str of len 25~100
        
        seq_oh = one_hot(seq)         # seq_oh : np array, one hot encoding sequence 
        X = self.pad_zeros(seq_oh)    # X  : torch.tensor , zero padded to 100
        
        if self.columns == None:
            # which means no auxilary label is needed
            item = X,X
        elif (type(self.columns) == list) & (len(self.columns)!=0):
            # return what's in columns
            aux_labels = self.df.loc[:,self.columns].values[index]
            item = X ,aux_labels

            if self.input_col is not None:
                input = [X]
                for col in self.input_col:
                    input.append(self.df.loc[:,col].values[index]) 
                item = input,aux_labels

        return item
        
    def pad_zeros(self,X):
        """
        zero padding at the right end of the sequence
        """
        gap = self.pad_to - X.shape[0]
        pad_fn = nn.ZeroPad2d([0,0,0,gap])  #  (padding_left , padding_right , padding_top , padding_bottom )
        # gap_array = np.zeros()
        X_padded = pad_fn(torch.tensor(X))
        return X_padded

        
class UTR_dataset(Dataset):
    def __init__(self,cell_line:str,script_dir = script_dir,data_dir = data_dir):
        # read csv first
        self.cell_line = cell_line 
        self.df = read_UTR_csv(cell_line=cell_line)

        # raw data
        self.oh_x = Seq_one_hot().d_transform(self.df,flattern=False) # (3970, 100, 4)
        self.y = self.df.TEaverage.values 

    def __len__(self):
        return self.oh_x.shape[0]
    
    def __getitem__(self,index):
        return self.oh_x[index],self.y[index]


class mask_reader(Dataset):
    def __init__(self,npy_path):
        """
        read the mask A549 sequence and with real sequence
        """
        self.data_set = np.load(npy_path)
        self.X = self.data_set[:,0]
        self.Y = self.data_set[:,1]
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        return (self.X[index,:,:],self.Y[index,:,:])
    

# def get_splited_dataloader(dataset,ratio:list,batch_size,num_workers):
#     """
#     split the total dataset into train val test, and return in a DataLoader (train_loader,val_loader,test_loader) 
#     dataset : the defined <UTR_dataset>
#     ratio : the ratio of train : val : test
#     batch_size : int
#     """

#     # make ratio to length
#     total_len = len(dataset)
#     lengths = [int(total_len*ratio[0]),int(len(dataset)*ratio[1])]
#     lengths.append(total_len-sum(lengths))         # make sure the sum of length is the total len

#     set_ls = random_split(dataset,lengths,generator=torch.Generator().manual_seed(42))         # split dataset 
    

    # return [DataLoader(subset, batch_size=batch_size, shuffle=True,num_workers=num_workers) for subset in set_ls]

def get_mask_dataloader(batch_size,num_workers):
    """
    (train_loader,val_loader,test_loader) 
    This is a splited data-loader with seed `42` on cell line `A549`
    each sequence have `5` nucleotide masked
    """
    set_ls = ['train','val','test']
    dataset_ls = [mask_reader(os.path.join(data_dir,'mask_data',set+"_mask.npy")) for set in set_ls]
    
    return [DataLoader(subset, batch_size=batch_size, shuffle=True,num_workers=num_workers) for subset in dataset_ls]

def get_mix_dataloader(batch_size,num_workers,shuffle=True):
    """
    (train_loader,val_loader,test_loader) 
    This is a splited data-loader with seed `42` on cell line `A549` and "human-library" and "snv-library"
    each sequence have `5` nucleotide
    """
    set_ls = ['train','val','test']
    dataset_ls = [mask_reader(os.path.join(data_dir,'mix_data',"mix_%s.npy"%set)) for set in set_ls]
    
    return [DataLoader(subset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers) for subset in dataset_ls]

def read_bpseq(test_bpseq_path):
    """
    read bpseq file, extract sequence and ptable
    """
    with open(test_bpseq_path,'r') as f:
        test_bpseq = f.readlines()
        f.close()

    for i in range(8):
        if test_bpseq[i].startswith('1'):
            start_index = i 

    bp_seq = test_bpseq[start_index:]

    seq = ''.join([line.strip().split(" ")[1] for line in bp_seq])
    ptable = [int(line.strip().split(" ")[2]) for line in bp_seq]

    assert len(ptable) == int(bp_seq[-1].split(" ")[0])

    return seq,ptable

def pad_zeros(X,pad_to):
    """
    zero padding at the right end of the sequence
    """
    gap = pad_to - X.shape[0]

    #  here we change to padding ahead  , previously  nn.ZeroPad2d([0,0,0,gap])
    pad_fn = nn.ZeroPad2d([0,0,gap,0])  #  (padding_left , padding_right , padding_top , padding_bottom )
    # gap_array = np.zeros()

    X_padded = pad_fn(X)
    return X_padded

class GSE65778_dataset(Dataset):
    
    def __init__(self,DF,pad_to,trunc_len=50,seq_col='utr',value_col='TE_count',other_input_columns=None):
        """
        Dataset to trancate sequence and return in one-hot encoding way
        `dataset(DF,pad_to,trunc_len=50,seq_col='utr')`
        ...DF: the dataframe contain sequence and its meta-info
        ...pad_to: final size of the output tensor
        ...trunc_len: maximum sequence to retain. number of nt preceding AUG
        ...seq_col : which col of the DF contain sequence to convert
        """
        self.df = DF
        self.pad_to = pad_to
        self.trunc_len =trunc_len
        
        # X and Y
        self.seqs = self.df.loc[:,seq_col].values
        self.other_input_columns = other_input_columns
        self.Y = self.df.loc[:,value_col].values
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,i):
        seq = self.seqs[i]
        x_padded = self.seq_chunk_N_oh(seq)
        input = x_padded
        if self.other_input_columns is not None:
                input = [x_padded]
                for col in self.other_input_columns:
                    input.append(self.df.loc[:,col].values[i]) 
        y = self.Y[i]
        return input,y
    
    def seq_chunk_N_oh(self,seq):
        """
        truncate the sequence and encode in one hot
        """
        if len(seq) >  self.trunc_len:
            seq = seq[-1* self.trunc_len:]
        
        X = one_hot(seq)
        X = torch.tensor(X)

        X_padded = pad_zeros(X, self.pad_to)
        
        return X_padded.float()


def get_splited_dataloader(dataset_func,df_ls,ratio:list,batch_size,num_workers):
    """
    split the total dataset into train val test, and return in a DataLoader (train_loader,val_loader,test_loader) 
    dataset : the defined <UTR_dataset>
    ratio : the ratio of train : val : test
    batch_size : int
    """

    #  determined set-ls
    if len(df_ls) == 1:
        dataset = dataset_func(df_ls[0])
        
        total_len = len(dataset)
        lengths = [int(total_len*sub_ratio) for sub_ratio in ratio[:-1]]
        lengths.append(total_len-sum(lengths))         # make sure the sum of length is the total len

        set_ls = random_split(dataset,lengths,generator=torch.Generator().manual_seed(43))         # split dataset 
    
    else:
        set_ls = [dataset_func(df) for df in df_ls]
    
    #  wrap dataset to dataloader
    loader_ls = [
                 DataLoader(subset,batch_size=batch_size,
                            shuffle=True,num_workers=num_workers,
                            generator=torch.Generator().manual_seed(43)) for subset in set_ls
                 ]
    if len(ratio) == 2:
        # a complement of empty test set
        loader_ls.append(None) 
    return loader_ls



def get_dataloader(POPEN):
    """
    wrapper
    """
    if POPEN.dataset == 'mix':
        loader_ls  = get_mix_dataloader(batch_size=POPEN.batch_size,num_workers=4)
    elif POPEN.dataset == "mask":
        loader_ls = get_mask_dataloader(batch_size=POPEN.batch_size,num_workers=4)
        
    elif POPEN.dataset == "ribo":
        full_df = pd.read_csv(POPEN.csv_path)
        
        if POPEN.kfold_cv is not None:
            # K-fold CV : 8:1:1 for each partition
            df_ls = KFold_df_split(full_df,POPEN.kfold_index)
        else:
            # POPEN.ratio will determine train :val :test ratio
            df_ls = [full_df] 
        
        dataset_func = lambda x :GSE65778_dataset(x,pad_to=POPEN.pad_to,trunc_len=POPEN.trunc_len,
                                                    seq_col=POPEN.seq_col,value_col=POPEN.aux_task_columns,
                                                    other_input_columns=POPEN.other_input_columns)
        
            
        loader_ls = get_splited_dataloader(dataset_func,df_ls,ratio=POPEN.train_test_ratio,
                                            batch_size=POPEN.batch_size,num_workers=4) # new function
    elif POPEN.dataset == "MTL":
        full_df = pd.read_csv(POPEN.csv_path)        
        
        if type(POPEN.split_like_paper) == list:
            # two csv path : 260,000 train set & 20,000 test set
            df_ls = [pd.read_csv(path) for path in POPEN.split_like_paper]
        elif POPEN.kfold_cv is not None:
            # K-fold CV : 8:1:1 for each partition
            df_ls = KFold_df_split(full_df,POPEN.kfold_index)
        else:
            df_ls = [full_df]
        
        dataset_func = lambda x : MTL_enc_dataset(DF=x,pad_to=POPEN.pad_to,
                                        aux_columns=POPEN.aux_task_columns,input_col=POPEN.other_input_columns)
        
        loader_ls = get_splited_dataloader(dataset_func,df_ls,ratio=POPEN.train_test_ratio,
                                            batch_size=POPEN.batch_size,num_workers=4) # new function
        
    else:
        # YK 's dataset
        dataset = UTR_dataset(cell_line=POPEN.cell_line)
        loader_ls = get_splited_dataloader(dataset,ratio=[0.7,0.1,0.2],
                                            batch_size=POPEN.batch_size,num_workers=4)
         # train,  val , test
    return loader_ls 

def KFold_df_split(df,K,**kfoldargs):
    """
    split the dataset DF in a ratio of 8:1:1 , train:val:test in the framework of  K-fold CV 
    set random seed = 43
    arguments:
    df : the `pd.DataFrame` object containing all data info
    K : [0,4] , the index of subfold 
    """
    
    # K-fold partition : n_splits=5
    fold_index = list(KFold(5,shuffle=True,random_state=43).split(df))
    train_index, val_test_index = fold_index[K]  
    # the first 4/5 part of it is train set
     
    # index the df
    train_df = df.iloc[train_index]
    val_test_df = df.iloc[val_test_index]
    
    # the remaining 1/5 data will further break into val and test
    val_df,test_df = train_test_split(val_test_df,test_size=0.5,random_state=43)
    
    return [train_df,val_df,test_df]