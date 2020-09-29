from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection  import train_test_split
import numpy as np
import pandas as pd
import os
import re


# ====================|   some path   |=======================
global script_dir
global data_dir
global cell_lines

with open("machine_configure.json",'r') as f:
    config = json.load(f)

script_dir = config['script_dir']
data_dir = config['data_dir']

match_celline = lambda x: re.match(r"rankedTE_(.*)\.csv",x).group(1)
data_fn = list(filter(lambda x : '.csv' in x,os.listdir(data_dir)))
cell_lines = [match_celline(fn) for fn in data_fn]


# =====================|   read data   |=======================

def read_UTR_csv(data_dir=data_dir,cell_line='ALL'):
    """
    read the UTR csv file give the path and cell line name
    argument:
    ...data_dir : dir where csv stores
    ...cell_line : determines the csv to read , can be 'ALL' , single cell_line or list of cell_line
    """
    if cell_line == 'ALL':
        cell_line = cell_lines
    
    # specify the cell lines
    for ii_cl in cell_line:
        assert ii_cl in cell_lines ,"%s is not a valid cell line name"%iicl

    # locate the index from cell_lines and get the file_name
    df_ls = [pd.read_csv(
                    os.path.join(data_dir,
                                 data_fn[cell_lines.index(ii_cl)])
                            ) for ii_cl in cell_line]

    if len(cell_line) ==1:
        return df_ls[0]
    else:
        return df_ls 
    
def read_label(df_ls):
    """
    return the TE score given the dataframe list piped from read_UTR_csv
    """  
    return [df.TEaverage.values for df in df_ls]

# =====================| one hot encode |=======================

class Seq_one_hot(object):
    def __init__(self,seq_type='nn'):
        """
        initiate the sequence one hot encoder
        """
        self.seq_len=100
        self.seq_type =seq_type
        self.enable_encoder()
        
    def enable_encoder(self):
        if self.seq_type == 'nn':
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder.drop_idx_ = None
            self.encoder.categories_ = [np.array(['A', 'C', 'G', 'T'], dtype='<U1')]*self.seq_len

    def discretize_seq(self,data):
        """
        discretize sequence into character
        argument:
        ...data: can be dataframe with UTR columns , or can be single string
        """
        if type(data) is pd.DataFrame:
            return np.stack(data.UTR.apply(lambda x: list(x)))
        elif type(data) is str:
            return np.array(list(data))
    
    def transform(self,data,flattern=True):
        """
        One hot encode
        argument:
        data : is a 2D array
        flattern : True
        """
        X = self.encoder.transform(data)                             # 400 for each seq
        X_M = np.stack([seq.reshape(self.seq_len,4) for seq in X])   # i.e 100*4
        return X if flattern else X_M
    
    def d_transform(self,data,flattern=True):
        """
        discretize data and put into transform
        """
        X = self.discretize_seq(data)
        return self.transform(X,flattern)