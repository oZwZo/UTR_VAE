import os
import torch
import logging
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def snapshot(dir_path, run_name, state,logger):
    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_best.pth')
    # torch.save can save any object
    # dict type object in our cases
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))

class Log_parser(object):
    def __init__(self,log_path,val_split_line=False):
        #           --------   read   --------
        self.val_split_line = val_split_line
        
        if os.path.exists(log_path):
            with open(log_path,'r') as f:
                log_file = f.readlines()
                f.close()
            # stripping 
            log_file = np.array([line.strip() for line in log_file])
        else:
            print('log path error !')
        self.log_file = log_file
        
        self.possible_metric = ['LOSS','lr','Avg_ACC','teaching_rate','TOTAL','KLD','MSE','M_N','CrossEntropy','chimerla_weight','Total','TE','Loop','Match','MAE','RMSE','RL_loss','Recons_loss','Motif_loss','RL_Acc','Recons_Acc','Motif_Acc']
        
        #          --------  basic  matcher   --------
        self.epoch_line_matcher = r"\s.* epoch (\d{1,4}).*"
        self.start_val_line_matcher = r"\s*.* start validation .*\s*"
        self.match_logging_time = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} -"
        self.match_percentage = r"\s*\d{1,6} /\s*\d{1,6}\s*\((\d|\.){,6}%\):"
        
        self.match_sub_verbose = lambda x : r"\s*%s:\s*(?P<%s>(-|\d|\.|e|){,40})"%(x,x)
        
        #          -------- high level matcher --------
        self.train_verbose_finder = self.match_logging_time + self.match_percentage
        
        #          --------- get output DF ---------
        self.extract_training_verbose_data()
        self.extract_val_verbose_data()
        
        
    def lines_matching(self,matcher):
        """
        return lines that can match certain syntax
        """
        return [line for line in self.log_file if re.match(matcher,line) is not None]
    
    def position_matching(self,matcher):
        """
        return position of the line that can match certain syntax
        """
        return [i for i,line in enumerate(self.log_file) if re.match(matcher,line) is not None]
    
    def get_metrics_order(self):
        """
        get train verbose line and define train_verbose_matcher automatically
        """
        self.train_verbose_lines = self.lines_matching(self.train_verbose_finder)  # find all the train verbose lines
        
        test_t_v = self.train_verbose_lines[0]                     # a testing train verbose
        
        # using the esting trainverbose to determine metric order
        train_metric = np.array([metric for metric in self.possible_metric if metric in test_t_v])
        train_metric = self.check_dup_metric(train_metric,test_t_v)
        
        train_metric_posi = np.array([test_t_v.index(metric) for metric in train_metric])
        
        order = train_metric_posi.argsort()
        self.train_metric = train_metric[order]
        
        #   ----|| automatically determine train verbose matcher ||----
        self.train_verbose_matcher = self.train_verbose_finder
        for metric in self.train_metric:
            self.train_verbose_matcher += self.match_sub_verbose(metric)
    
    def check_dup_metric(self,train_metric,test_t_v):
        """
        to deal with the problem of `MSE` and `RMSE` 
        """
        train_metric = list(train_metric)
        if ("MSE" in train_metric) & ("RMSE" in train_metric):
            if test_t_v.index('MSE') == test_t_v.index('RMSE')+1:
                train_metric.remove('MSE')
        return np.array(train_metric)
    
    def extract_training_verbose_data(self):
        """
        regular expression to match the printed metric during training and save to pd.DataFrame
        """
        self.get_metrics_order()
        
        self.train_verbose_array = np.array(
            [list(
                re.match(self.train_verbose_matcher,line).groupdict().values()
                    ) for line in self.train_verbose_lines]
            ).astype(np.float64)
        
        self.train_verbose_DF = pd.DataFrame(self.train_verbose_array,columns=self.train_metric)
        
        # return self.train_verbose_DF 
    
    def extract_val_verbose_data(self):
        """
        regular expression to match the printed metric during training and save to pd.DataFrame
        """
        self.start_val_posi = self.position_matching(self.start_val_line_matcher)
        self.val_verbose_posi = np.array(self.start_val_posi) +2  # observe from log
        self.val_verbose_lines = self.log_file[self.val_verbose_posi]
        if self.val_split_line:
            self.val_verbose_lines = ["\t".join(self.log_file[[posi,posi+1,posi+2,posi+3,posi+4]]) for posi in self.val_verbose_posi]
         
        test_v_v = self.val_verbose_lines[1]
        
         # using the esting trainverbose to determine metric order
        val_metric = np.array([metric for metric in self.possible_metric if metric in test_v_v])
        val_metric = self.check_dup_metric(val_metric,test_v_v)
        val_metric_posi = np.array([test_v_v.index(metric) for metric in val_metric])
        order = val_metric_posi.argsort()    # sort 
        self.val_metric = val_metric[order]
        
        
        #   ----|| automatically determine val verbose matcher ||----
        self.val_verbose_matcher = self.match_logging_time 
        
        if re.match(self.match_logging_time + self.match_percentage,test_v_v) is not None:
            self.val_verbose_matcher += self.match_percentage        # detect whether validation set also get percentage info
        
        for metric in self.val_metric:
            self.val_verbose_matcher += self.match_sub_verbose(metric)
        
        self.val_verbose_array = np.array(
            [list(
                re.match(self.val_verbose_matcher,line).groupdict().values()
                    ) for line in self.val_verbose_lines]
            ).astype(np.float64)
        
        self.val_verbose_DF = pd.DataFrame(self.val_verbose_array,columns=self.val_metric)
        
        # return self.val_verbose_DF 
    
    def plot_val_metric(self,fig=None,dataset='val'):
        DF = self.val_verbose_DF if dataset == 'val' else self.train_verbose_DF
        metrics = self.val_metric if dataset == 'val' else self.train_metric
        n = len(metrics)
        
        if fig is None:
            fig = plt.figure(figsize=(18,5*np.ceil(n/3)))
        if n <=3:
            axs = fig.subplots(1,n)
            for i in range(n):
                axs[i].plot(DF[metrics[i]].values)
                axs[i].set_title(dataset.capitalize()+" "+metrics[i])  # TRAIN or VAL
        else:
            axs = fig.add_subplot(n//3+1,n,1+i)
        

def plot_a_exp_set(log_list,log_name_ls,dataset='val',fig=None,layout=None,check_time=10,start_from=0,mean_of_train=None,**kwargs):
    
    fig = plt.figure(figsize=(20,5)) if fig is None else fig

    n = len(log_list[0].__getattribute__(dataset+"_metric"))  # val or train
    if layout is None:
        axs = fig.subplots(1,n);
    else:
        row,column = layout 
        axs = fig.subplots(row,column)
    
    all_metric = [logg.__getattribute__(dataset+"_metric") for logg in log_list]
    share_metric = [all_metric[0]]
    for logg_metric in all_metric[1:]:
        share_metric = np.intersect1d(share_metric,logg_metric)
    
    for i,metric in enumerate(share_metric):
        # layout 
        
        if layout is None:
            ax = axs[i]  
        else:
            ax = axs[i//column,i%column]
        for st,log in enumerate(log_list):
            DF = log.__getattribute__(dataset+"_verbose_DF")
            if (dataset == 'train') & (type(mean_of_train)==int):
                DF = mean_of(mean_of_train,DF)
            X = np.arange(DF.shape[0])*check_time if dataset == 'val' else np.arange(DF.shape[0])/6
            ax.plot(X[start_from:],DF[metric].values[start_from:],label=log_name_ls[st],**kwargs)
            ax.set_title(" ".join([dataset.capitalize(),metric]))
        ax.legend()
        
def mean_of(x,DF):
    
    values = DF.values
    mean_ls = []
    
    for i in range(0,values.shape[0],x):
        
        mean_ls.append(np.mean(values[i:i+x,:],axis=0))
    
    mean_ls = np.array(mean_ls)
    mean_DF = pd.DataFrame(mean_ls,columns=DF.columns)
    
    return mean_DF

def read_log_of_a_dir(log_dir):
    """
    ...log_dir : abs path of log dir
    """
    file_ls = [file for file in os.listdir(log_dir) if ".log" in file]
    log_path = [os.path.join(log_dir,file) for file in file_ls]
    log_name = [file.replace('.log','') for file in file_ls]
    log_ls = [Log_parser(file) for file in log_path]
    return log_ls,log_name