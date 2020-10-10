from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection  import train_test_split
import logging
import numpy as np
import pandas as pd
import os
import json
import re

print(os.path.dirname(__file__))

# ====================|   some path   |=======================
global script_dir
global data_dir
global log_dir
global pth_dir
global cell_lines

with open(os.path.join(os.path.dirname(__file__),"machine_configure.json"),'r') as f:
    config = json.load(f)

script_dir = config['script_dir']
data_dir = config['data_dir']
log_dir = config['log_dir']
pth_dir = config['pth_dir']

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
    elif type(cell_line) == str:
        cell_line = [cell_line]
    
    # specify the cell lines
    for ii_cl in cell_line:
        assert ii_cl in cell_lines ,"%s is not a valid cell line name"%ii_cl

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

# =====================|calculate Conv shape|==================
    
def cal_convTrans_shape(L_in,padding,diliation,kernel_size,stride,out_padding=0):
    """
    For convolution Transpose 1D decoding , compute the final length
    """
    L_out = (L_in -1 )*stride + diliation*(kernel_size -1 )+1-2*padding + out_padding 
    return L_out

def cal_conv_shape(L_in,padding,diliation,kernel_size,stride):
    """
    For convolution 1D encoding , compute the final length 
    """
    L_out = 1+ (L_in + 2*padding -diliation*(kernel_size-1) -1)/stride
    return L_out

# =====================|   logger       |=======================

def setup_logs(save_dir, run_name):
    """

    :param save_dir:  the directory to set up logs
    :param type:  'model' for saving logs in 'logs/cpc'; 'imp' for saving logs in 'logs/imp'
    :param run_name:
    :return:logger
    """
    # initialize logger
    logger = logging.getLogger("VAE")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    log_file = os.path.join(save_dir, run_name + ".log")
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger



def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_best.pth')
    # torch.save can save any object
    # dict type object in our cases
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))

    