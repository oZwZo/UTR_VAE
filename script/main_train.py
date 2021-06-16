import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import time
import logging
import inspect
import argparse
import numpy as np 

import torch
from torch import nn, optim
from torch import functional as F
from models import MTL_models,CNN_models,DL_models,reader,train_val, Backbone, Baseline_models
from models.ScheduleOptimizer import ScheduledOptim,scheduleoptim_dict_str
from models.popen import Auto_popen
from models.loss import Dynamic_Task_Priority,Dynamic_Weight_Averaging

parser = argparse.ArgumentParser('the main to train model')
parser.add_argument('--config_file',type=str,required=True)
parser.add_argument('--cuda',type=int,default=None,required=False)
parser.add_argument("--kfold_index",type=int,default=None,required=False)
args = parser.parse_args()

POPEN = Auto_popen(args.config_file)
if args.cuda is not None:
    POPEN.cuda_id = args.cuda

POPEN.kfold_index = args.kfold_index
if POPEN.kfold_cv:
    if args.kfold_index is None:
        raise NotImplementedError("please specify the kfold index to perform K fold cross validation")
    POPEN.vae_log_path = POPEN.vae_log_path.replace(".log","_cv%d.log"%args.kfold_index)
    POPEN.vae_pth_path = POPEN.vae_pth_path.replace(".pth","_cv%d.pth"%args.kfold_index)
    
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# cuda2 = torch.device('cuda:2')

# Run name
if POPEN.run_name is None:
    run_name = POPEN.model_type + time.strftime("__%Y_%m_%d_%H:%M")
else:
    run_name = POPEN.run_name
    
# log dir
logger = utils.setup_logs(POPEN.vae_log_path)

#  built model dir or check resume 
POPEN.check_experiment(logger)


#                               |=====================================|
#                               |===========   setup  part  ==========|
#                               |=====================================|

# read data
train_loader,val_loader,test_loader = reader.get_dataloader(POPEN)

# ===========  setup model  ===========
# train_iter = iter(train_loader)
# X,Y  = next(train_iter)
# -- pretrain -- 
if POPEN.pretrain_pth is not None:
    # load pretran model
    pretrain_popen = Auto_popen(POPEN.pretrain_pth)
    pretrain_model = pretrain_popen.Model_Class(*pretrain_popen.model_args)
    utils.load_model(pretrain_popen,pretrain_model,logger)  
    
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
        utils.load_model(task_popen,task_model,logger)
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
    previous_epoch,best_loss,best_acc = utils.resume(POPEN,model,optimizer,logger)
    

# =========== fix parameters ===========
if POPEN.modual_to_fix in dir(model):
    model = utils.fix_parameter(model,POPEN.modual_to_fix)
    logger.info(' \t \t ==============<<< %s part is fixed>>>============== \t \t \n'%POPEN.modual_to_fix)

#                               |=====================================|
#                               |==========  training  part ==========|
#                               |=====================================|
for epoch in range(POPEN.max_epoch-previous_epoch+1):
    epoch += previous_epoch
    
    #           ----------| train |----------
    logger.info("\n===============================|    epoch {}   |===============================\n".format(epoch))
    train_val.train(dataloader=train_loader,model=model,optimizer=optimizer,popen=POPEN,epoch=epoch)
           
    #         -----------| validate |-----------
    
    if epoch % POPEN.config_dict['setp_to_check'] == 0:
        val_total_loss,val_avg_acc = train_val.validate(val_loader,model,popen=POPEN,epoch=epoch)
        
        DICT ={"ran_epoch":epoch,"n_current_steps":optimizer.n_current_steps,"delta":optimizer.delta} if type(optimizer) == ScheduledOptim else {"ran_epoch":epoch}
        POPEN.update_ini_file(DICT,logger)
        
    #    -----------| compare the result |-----------
        if (best_loss > val_total_loss) | (best_acc < val_avg_acc):
            # update best performance
            best_loss = min(best_loss,val_total_loss)
            best_acc = max(best_acc,val_avg_acc)
            best_epoch = epoch
            
            # save
            utils.snapshot(POPEN.vae_pth_path, {
                        'epoch': epoch + 1,
                        'validation_acc': val_avg_acc,
                        'state_dict': model.state_dict(),
                        'validation_loss': val_total_loss,
                        'optimizer': optimizer.state_dict(),
                    })
            
            # update the popen
            POPEN.update_ini_file({'run_name':run_name,
                                "ran_epoch":epoch,
                             
                                   "best_acc":best_acc},
                                logger)
            
        elif (epoch - best_epoch >= 30)&((type(optimizer) == ScheduledOptim)):
            optimizer.increase_delta()
            
        elif (epoch - best_epoch >= 60)&(epoch > POPEN.max_epoch/2):
            # at the late phase of training
            logger.info("<<<<<<<<<<< Early Stopping >>>>>>>>>>")
            break