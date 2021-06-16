import os
import sys
import pandas as pd
import numpy as np
import re
import PATH
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader,Dataset,random_split
from tqdm import tqdm

from matplotlib import pyplot as plt
import scipy.stats as stats
import seaborn as sns

from models.popen import Auto_popen
from models import ScheduleOptimizer
from models.ScheduleOptimizer import ScheduledOptim 
from models.log_and_save import Log_parser,plot_a_exp_set
from models import reader
from models import DL_models
from models import CNN_models
from models.loss import Dynamic_Task_Priority

from utils import Seq_one_hot,read_UTR_csv,read_label,resume,setup_logs,load_model,fix_parameter

from importlib import reload
from models import popen

import logging

    # log dir
logger = setup_logs("test_log.log")

def compute_acc(X,out):
    batch_size = X.shape       # B*100*4
    true_max=torch.argmax(X,dim=2)

    recon_max=torch.argmax(out,dim=2)

    return torch.sum(true_max == recon_max).item() 

def reconstruct_seq(out_seq,X):
    seq = torch.zeros_like(X)
#     out_seq = torch.cat(out_seq,dim=1)
    position = torch.argmax(out_seq,dim=2)     # X_reconst : b*100*4

    for batch_idx in range(X.shape):
        for i,j in enumerate(position[batch_idx]):
            seq[batch_idx,i,j.item()] = 1     
            
    return torch.mean(X.mul(seq).sum(dim=2).sum(dim=1)) 

def read_main_(config_file,logger,cuda=None,kfold_index=None):
    """
    return a dict
    """
    POPEN = Auto_popen( config_file)
    if  cuda is not None:
        POPEN.cuda_id =  cuda

    POPEN.kfold_index =  kfold_index
    if POPEN.kfold_cv:
        if  kfold_index is None:
            raise NotImplementedError("please specify the kfold index to perform K fold cross validation")
        POPEN.vae_log_path = POPEN.vae_log_path.replace(".log","_cv%d.log"% kfold_index)
        POPEN.vae_pth_path = POPEN.vae_pth_path.replace(".pth","_cv%d.pth"% kfold_index)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cuda2 = torch.device('cuda:2')

    # Run name
    if POPEN.run_name is None:
        run_name = POPEN.model_type + time.strftime("__%Y_%m_%d_%H:%M")
    else:
        run_name = POPEN.run_name

    # log dir
    # logger = utils.setup_logs(POPEN.vae_log_path)

    #  built model dir or check resume 
    POPEN.check_experiment(logger)


    #                               |=====================================|
    #                               |===========   setup  part  ==========|
    #                               |=====================================|

    # read data
    loader_ls = reader.get_dataloader(POPEN)

    # ===========  setup model  ===========
    # train_iter = iter(train_loader)
    # X,Y  = next(train_iter)
    # -- pretrain -- 
    if POPEN.pretrain_pth is not None:
        # load pretran model
        pretrain_popen = Auto_popen(POPEN.pretrain_pth)
        pretrain_model = pretrain_popen.Model_Class(*pretrain_popen.model_args)
        load_model(pretrain_popen,pretrain_model,logger)  

        # DL_models.LSTM_AE
        if POPEN.Model_Class == pretrain_popen.Model_Class:
            # if not POPEN.Resumable:
            #     # we only load pre-train for the first time 
            #     # later we can resume 
            model = pretrain_model.cuda(POPEN.cuda_id)
            del pretrain_model
        else:
            downstream_model = POPEN.Model_Class(*POPEN.model_args)

            # merge 
            model = MTL_models.Enc_n_Down(pretrain_model,downstream_model).cuda(POPEN.cuda_id)

    # -- end2end -- 
    elif POPEN.path_category == "CrossStitch":
        backbone = {}
        for t in POPEN.tasks:
            task_popen = Auto_popen(POPEN.backbone_config[t])
            task_model = task_popen.Model_Class(*task_popen.model_args)
            load_model(task_popen,task_model,logger)
            backbone[t] = task_model
        POPEN.model_args = [backbone] + POPEN.model_args
        model = POPEN.Model_Class(*POPEN.model_args).cuda(POPEN.cuda_id)
    else:
        Model_Class = POPEN.Model_Class  # DL_models.LSTM_AE
        model = Model_Class(*POPEN.model_args).cuda(POPEN.cuda_id)
    # =========== set optimizer ===========
    if POPEN.optimizer == 'Schedule':
        optimizer = ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                          betas=(0.9, 0.98), 
                                          eps=1e-09, 
                                          weight_decay=1e-4, 
                                          amsgrad=True),
                               n_warmup_steps=20)
    elif type(POPEN.optimizer) == dict:
        optimizer = eval(scheduleoptim_dict_str.format(**POPEN.optimizer))
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=POPEN.lr,
                                betas=(0.9, 0.98), 
                                eps=1e-09, 
                                weight_decay=POPEN.l2)

    if POPEN.loss_schema == 'DTP':
        POPEN.loss_schedualer = Dynamic_Task_Priority(POPEN.tasks,POPEN.gamma,POPEN.chimerla_weight)
    elif POPEN.loss_schema == 'DWA':
        POPEN.loss_schedualer = Dynamic_Weight_Averaging(POPEN.tasks,POPEN.tau,POPEN.chimerla_weight)
    # =========== resume ===========
    best_loss = np.inf
    best_acc = 0
    best_epoch = 0
    previous_epoch = 0
    if POPEN.Resumable:
        previous_epoch,best_loss,best_acc = resume(POPEN,model,optimizer,logger)


    # =========== fix parameters ===========
    if POPEN.modual_to_fix in dir(model):
        model = fix_parameter(model,POPEN.modual_to_fix)
        logger.info(' \t \t ==============<<< %s part is fixed>>>============== \t \t \n'%POPEN.modual_to_fix)
    
    return {"popen":POPEN,"model":model,"loader_ls":loader_ls}

def save_result_to_csv(y_true_ls,y_true_f,y_pred_f,save_path,motif_detection):
    """
    save the result from forward to csv
    """
    motif_dict = {'with_uAUG':np.stack([ary[:,1] for ary in y_true_ls]).flatten()}
    
    # if motif detection is an auxiliary task
    if motif_detection:
        motifs_name = ['with_GCC', 'with_GGC', 'with_CGG', 'with_GGG', 'with_CCC', 'with_TTT', 'with_AAA']
        for i in range(7):
            motif_dict[motifs_name[i]] = np.stack([ary[:,2+i] for ary in y_true_ls]).flatten()
            
    # save to csv
    data_dict = {'Y_true':y_true_f,'Y_pred':y_pred_f,**motif_dict}
    df = pd.DataFrame(data_dict)
    df.to_csv(save_path,index=False)
    return df

def get_4_types_data_from_csv(df):
    """
    given the saved csv, extract 4 data set:
        uAUG_tre,  true values with uAUG
        uAUG_pred, predicted values with uAUG
        nAUG_tre,  true values without uAUG
        nAUG_pred  predicted values without uAUG
    """
    # get basic info from df
    y_true_f = df.Y_true.values
    y_pred_f = df.Y_pred.values
    with_uAUG = df.with_uAUG.values

    # stratify them
    uAUG_tre = y_true_f[with_uAUG == 1]
    nAUG_tre = y_true_f[with_uAUG == 0]

    uAUG_pred = y_pred_f[with_uAUG == 1]
    nAUG_pred = y_pred_f[with_uAUG == 0]

    return uAUG_tre,uAUG_pred,nAUG_tre,nAUG_pred

def my_kde_joint_plot(uAUG_tre,uAUG_pred,nAUG_tre,nAUG_pred,title,r2="",text_posi=(7.6,1.3),**kwargs):
    sns.set_style("white")
    # sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.labelpad'] = 5
    plt.rcParams['axes.linewidth']= 2
    plt.rcParams['xtick.labelsize']= 14
    plt.rcParams['ytick.labelsize']= 14
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.minor.width'] = 2
    plt.rcParams['xtick.major.width'] = 3
    plt.rcParams['ytick.major.width'] = 4
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    c1 = (0.3, 0.45, 0.69)
    c2 = 'r'
    joint = sns.JointGrid(x=uAUG_tre,y=uAUG_pred,height=8,ratio=11,space=0,**kwargs)
    joint.plot_joint(plt.scatter, alpha=0.1,s=6)
    joint.fig.gca().set_title(title+'\n\n',fontsize=25)
    joint.fig.gca().text(*text_posi,r"$R^2 =$"+str(r2),fontsize=20)
    joint.plot_marginals(sns.kdeplot,shade=c1)

    joint.x=nAUG_tre
    joint.y=nAUG_pred
    joint.plot_joint(plt.scatter, alpha=0.1,s=6)
    joint.plot_marginals(sns.kdeplot,shade=c2)


    joint.set_axis_labels('True TE-score', 'Predicted TE-score', **{'size':22});

def compute_r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def compute_acc(X,recon):
    """
    compute the reconstruction accuracy
    """
    batch_size = X.shape[0]       # B*100*4

    true_max=torch.argmax(X,dim=2)
    recon_max=torch.argmax(recon,dim=1)
    return torch.sum(true_max == recon_max,dim=1)

def get_input_grad(test_X,index):
    
    # process X
    sampled_X = test_X[index].unsqueeze(0)
    sampled_X.requires_grad = True
    # forward
    sampled_out = model(sampled_X)['RL']

    # auto grad part
    external_grad = torch.ones_like(sampled_out)
    sampled_out.backward(gradient=external_grad,retain_graph=True) # define \frac{d out}{ }

    return sampled_X.grad.abs()