import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import torch
import utils
import json
from model import DL_models
import configparser
import logging

class Auto_popen(object):
    def __init__(self,config_file,check_record):
        """
        read the config_fiel
        """
        # machine config path
        self.script_dir = utils.script_dir
        self.data_dir = utils.data_dir
        self.log_dir = utils.log_dir
        self.pth_dir = utils.pth_dir
        
        
        # transform to dict and convert to  specific data type
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.config_file = config_file
        self.config_dict = {item[0]: eval(item[1]) for item in config.items('DEFAULT')}
        
        self.max_epoch = self.config_dict['max_epoch']
        self.run_name = self.config_dict['run_name']
        self.setting_name = self.config_dict['setting_name']
        self.model_type = self.config_dict['model_type']
        # the saving direction
        self.vae_log_path = os.path.join(self.log_dir,self.model_type,self.setting_name,self.run_name +'.log')
        self.vae_pth_path = os.path.join(self.pth_dir,self.model_type,self.setting_name,self.run_name + '-model_best.pth')
        self.Resumable = False
        
        # data
        self.cell_line = self.config_dict['cell_line']
        self.batch_size = self.config_dict['batch_size']
        
        # generate self.model_args
        self.get_model_config()
        
    def get_model_config(self):
        """
        assert we type in the correct model type and group them into model_args
        """
        assert self.model_type in dir(DL_models), "model type not correct"
        
        self.Model_Class = eval("DL_models.{}".format(self.model_type))
        
        # teacher foring
        if self.config_dict['teacher_forcing'] is True:
            # default setting
            popen.teacher_forcing = [0.032188758248682,0.032188758248682]  # k = b , k = log5 / 50
        
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
    
    def check_experiment(self,logger):
        """
        check any unfinished experiment ?
        """
        
        # pth exist ? log exist ?
        if not os.path.exists(self.vae_log_path):
            os.makedirs(self.vae_log_path)
        if not os.path.exists(self.vae_pth_path):
            os.makedirs(self.vae_pth_path)
            
        if os.path.exists(self.vae_log_path) & os.path.exists(self.vae_pth_path):
            self.Resumable = True
            logger.info(' \t \t ==============<<<  Experiment detected  >>>============== \t \t \n')
            
    def update_ini_file(self,E,logger):
        """
        E  is the dict contain the things to update
        """
        # update the ini file
        self.config_dict.update(E)
        strconfig = {K: repr(V) for K,V in config_dict}
        self.config['DEFAULT'] = strconfig
        
        with open(self.config_file,'w') as f:
            self.config.write(f)
        
        logger.info('   ini file updated    ')
        
