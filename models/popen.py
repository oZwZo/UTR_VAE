import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import torch
import utils
import json
from models import DL_models
from models import CNN_models
from models import MTL_models
from models import Baseline_models
import configparser
import logging

class Auto_popen(object):
    def __init__(self,config_file):
        """
        read the config_fiel
        """
        # machine config path
        self.script_dir = utils.script_dir
        self.data_dir = utils.data_dir
        self.log_dir = utils.log_dir
        self.pth_dir = utils.pth_dir
        self.te_net_l2 = None
        
        # transform to dict and convert to  specific data type
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.config_file = config_file
        self.config_dict = {item[0]: eval(item[1]) for item in self.config.items('DEFAULT')}
        
        # assign some attr from config_dict         
        self.set_attr_from_dict(self.config_dict.keys())
        self.check_run_and_setting_name()                          # check run name
        self._dataset = "_" + self.dataset if self.dataset != '' else self.dataset
        # the saving direction
        self.vae_log_path = os.path.join(self.log_dir,self.model_type+self._dataset,self.setting_name,self.run_name +'.log')
        self.vae_pth_path = os.path.join(self.pth_dir,self.model_type+self._dataset,self.setting_name,self.run_name + '-model_best.pth')
        self.Resumable = False
        
        # generate self.model_args
        self.get_model_config()
    
    def set_attr_from_dict(self,attr_ls):
        for attr in attr_ls:
            self.__setattr__(attr,self.config_dict[attr])

    def check_run_and_setting_name(self):
        file_name = self.config_file.split("/")[-1]
        dir_name = self.config_file.split("/")[-2]
        self.setting_name = dir_name
        assert self.run_name == file_name.split(".")[0]
        
    def get_model_config(self):
        """
        assert we type in the correct model type and group them into model_args
        """
        if "Conv" in self.model_type:
            assert self.model_type in dir(CNN_models), "model type not correct"
            self.Model_Class = eval("CNN_models.{}".format(self.model_type))
        elif "LSTM" in self.model_type:
            assert self.model_type in dir(DL_models), "model type not correct"
            self.Model_Class = eval("DL_models.{}".format(self.model_type))
        else:
            if self.model_type in dir(Baseline_models):
                self.Model_Class = eval("Baseline_models.{}".format(self.model_type))
            assert self.model_type in dir(MTL_models), "model type not correct"
            self.Model_Class = eval("MTL_models.{}".format(self.model_type))
        
        # teacher foring
        if self.teacher_forcing is True:
            # default setting
            self.teacher_forcing = True
            t_k,t_b = (0.032188758248682,0.032188758248682)  # k = b , k = log5 / 50
        elif type(self.config_dict['teacher_forcing']) == list:
            self.teacher_forcing = True
            t_k,t_b = self.config_dict['teacher_forcing']
        elif self.config_dict['teacher_forcing'] == 'fixed':
            t_k = t_b = 100
        elif self.config_dict['teacher_forcing'] == False:
            t_k = t_b = 100
            
        
        if "LSTM" in self.model_type:
            self.model_args=[self.input_size,
                             self.config_dict["hidden_size_enc"],
                             self.config_dict["hidden_size_dec"],
                             self.config_dict["num_layers"],
                             self.config_dict["latent_dim"],
                             self.config_dict["seq_in_dim"],
                             self.config_dict["decode_type"],
                             self.config_dict['teacher_forcing'],
                             self.config_dict['discretize_input'],
                             t_k,t_b,
                             self.config_dict["bidirectional"],
                             self.config_dict["fc_output"]]
            
        if "Conv" in self.model_type:
            args_to_read = ["channel_ls","padding_ls","diliat_ls","latent_dim","kernel_size"]
            self.model_args=[self.__getattribute__(args) for args in args_to_read]
        
        if  self.model_type in  ['TO_SEQ_TE','TRANSFORMER_SEQ_TE','Baseline']:
            args_to_read = ["channel_ls","padding_ls","diliat_ls","latent_dim","kernel_size","num_label"]
            self.model_args=[self.__getattribute__(args) for args in args_to_read] 
        
        if "TWO_TASK_AT" in self.model_type:
            args_to_read = ["latent_dim","linear_chann_ls","num_label","te_chann_ls","ss_chann_ls","dropout_rate"]
            self.model_args=[self.__getattribute__(args) for args in args_to_read]
    
    def check_experiment(self,logger):
        """
        check any unfinished experiment ?
        """
        log_save_dir = os.path.join(self.log_dir,self.model_type+self._dataset,self.setting_name)
        pth_save_dir = os.path.join(self.pth_dir,self.model_type+self._dataset,self.setting_name)
        # make dirs 
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        if not os.path.exists(pth_save_dir):
            os.makedirs(pth_save_dir)
        
        # check resume
        if os.path.exists(self.vae_log_path) & os.path.exists(self.vae_pth_path):
            self.Resumable = True
            logger.info(' \t \t ==============<<<  Experiment detected  >>>============== \t \t \n')
            
    def update_ini_file(self,E,logger):
        """
        E  is the dict contain the things to update
        """
        # update the ini file
        self.config_dict.update(E)
        strconfig = {K: repr(V) for K,V in self.config_dict.items()}
        self.config['DEFAULT'] = strconfig
        
        with open(self.config_file,'w') as f:
            self.config.write(f)
        
        logger.info('   ini file updated    ')
        
