import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import json
from model import DL_models


class Auto_popen(object):
    def __init__(self,config_file):
        """
        read the config_fiel
        """
        with open(config_file) as f:
            config_dict = json.load(f)
        self.config_dict = config_dict
        self.max_epoch = config_dict['epoch']
        self.run_name = config_dict['run_name']
        self.setting_name = config_dict['setting_name']
        self.model_type = config_dict['model_type']
        
        # data
        self.cell_line = config_dict['cell_line']
        self.batch_size = config_dict['batch_size']
        
        # generate self.model_args
        self.get_model_config()
        
    def get_model_config(self):
        """
        assert we type in the correct model type and group them into model_args
        """
        assert self.model_type in dir(DL_models), "model type not correct"
        
        self.Model_Class = eval("DL_models.{}".format(self.model_type))
        
        if "LSTM" in self.model_type:
            self.model_args=[self.config_dict["input_size"],
                             self.config_dict["hidden_size_enc"],
                             self.config_dict["hidden_size_dec"],
                             self.config_dict["num_layers"],
                             self.config_dict["latent_dim"],
                             self.config_dict["seq_in_dim"],
                             self.config_dict["decode_type"]]
            
        if "Conv" in self.model_type:
            self.model_args=[self.config_dict["enc_Conv_size"],
                             self.config_dict['dec_Conv_size'],
                             self.config_dict["latent_dim"],
                             self.config_dict["seq_in_dim"]]
    
    def check_record(self):
        """
        check any unfinished experiment ?
        """
        # TODO: check the current setting , if exitsed, then resume and continue the exp
