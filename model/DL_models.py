import torch
import numpy as np
import pandas as pd
from torch import nn

class Conv_encoder(nn.Module):
    def __init__(self,conv_size):
       
       super(Conv_encoder,self).__init__()
       
       self.block = nn.Sequential()
        