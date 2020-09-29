import torch
from torch.utils.data import  DataLoader, Dataset ,random_split
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import Seq_one_hot,read_UTR_csv,read_label

global script_dir
global data_dir

with open("machine_configure.json",'r') as f:
    config = json.load(f)

script_dir = config['script_dir']
data_dir = config['data_dir']


class UTR_dataset(Dataset):
    def __init__(selfm,cell_line:str,script_dir = script_dir,data_dir = data_dir):
        # read csv first
        self.cell_line = cell_line 
        self.csv = read_UTR_csv(cell_line)

        # raw data
        self.oh_x = Seq_one_hot().d_transform(self.csv,flattern=False) # (3970, 100, 4)
        self.y = read_label(self.csv)

    def __len__(self):
        return self.oh_x.shape[0]
    
    def __getitem__(self,index):
        return self.oh_x[index],self.y[index]


def get_splited_dataloader(dataset,ratio:list,batch_size,num_worker) -> list[DataLoader]:
    """
    split the total dataset into train val test, and return in a DataLoader
    dataset : the defined <UTR_dataset>
    ratio : the ratio of train : val : test
    batch_size : int
    """

    # make ratio to length
    total_len = len(dataset)
    lengths = [int(total_len*ratio[0]),int(len(dataset)*ratio[1])]
    lengths.append(total_len-sum(lengths))         # make sure the sum of length is the total len

    set_ls = random_split(dataset,lengths)         # split dataset 

    return [DataLoader(subset, batch_size=batch_size, shuffle=True,num_worker=num_worker) for subset in set_ls]


