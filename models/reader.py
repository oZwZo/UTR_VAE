import torch
from torch.utils.data import  DataLoader, Dataset ,random_split
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


