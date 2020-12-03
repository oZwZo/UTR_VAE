import torch
from torch.utils.data import  DataLoader, Dataset ,random_split
import numpy as np
import os
import sys
import json
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


class UTR_dataset(Dataset):
    def __init__(self,cell_line:str,script_dir = script_dir,data_dir = data_dir):
        # read csv first
        self.cell_line = cell_line 
        self.csv = read_UTR_csv(cell_line=cell_line)

        # raw data
        self.oh_x = Seq_one_hot().d_transform(self.csv,flattern=False) # (3970, 100, 4)
        self.y = self.csv.TEaverage.values 

    def __len__(self):
        return self.oh_x.shape[0]
    
    def __getitem__(self,index):
        return self.oh_x[index],self.y[index]


def get_splited_dataloader(dataset,ratio:list,batch_size,num_workers):
    """
    split the total dataset into train val test, and return in a DataLoader (train_loader,val_loader,test_loader) 
    dataset : the defined <UTR_dataset>
    ratio : the ratio of train : val : test
    batch_size : int
    """

    # make ratio to length
    total_len = len(dataset)
    lengths = [int(total_len*ratio[0]),int(len(dataset)*ratio[1])]
    lengths.append(total_len-sum(lengths))         # make sure the sum of length is the total len

    set_ls = random_split(dataset,lengths,generator=torch.Generator().manual_seed(42))         # split dataset 
    

    return [DataLoader(subset, batch_size=batch_size, shuffle=True,num_workers=num_workers) for subset in set_ls]


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
    

def get_splited_dataloader(dataset,ratio:list,batch_size,num_workers):
    """
    split the total dataset into train val test, and return in a DataLoader (train_loader,val_loader,test_loader) 
    dataset : the defined <UTR_dataset>
    ratio : the ratio of train : val : test
    batch_size : int
    """

    # make ratio to length
    total_len = len(dataset)
    lengths = [int(total_len*ratio[0]),int(len(dataset)*ratio[1])]
    lengths.append(total_len-sum(lengths))         # make sure the sum of length is the total len

    set_ls = random_split(dataset,lengths,generator=torch.Generator().manual_seed(42))         # split dataset 
    

    return [DataLoader(subset, batch_size=batch_size, shuffle=True,num_workers=num_workers) for subset in set_ls]

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
