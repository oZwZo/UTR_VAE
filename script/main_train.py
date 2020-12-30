import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import time
import logging
import argparse
import numpy as np 

import torch
from torch import nn, optim
from torch import functional as F
from models import MTL_models,CNN_models,DL_models,reader,train_val, Backbone, Baseline_models
from models.ScheduleOptimizer import ScheduledOptim,scheduleoptim_dict_str
from models.popen import Auto_popen

parser = argparse.ArgumentParser('the main to train model')
parser.add_argument('--config_file',type=str,required=True)
parser.add_argument('--cuda',type=int,default=None,required=False)
args = parser.parse_args()

POPEN = Auto_popen(args.config_file)
if args.cuda is not None:
    POPEN.cuda_id = args.cuda

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
if POPEN.dataset == 'mix':
     train_loader,val_loader,test_loader  = reader.get_mix_dataloader(batch_size=POPEN.batch_size,num_workers=4)
elif POPEN.dataset == "mask":
    train_loader,val_loader,test_loader = reader.get_mask_dataloader(batch_size=POPEN.batch_size,num_workers=4)
elif POPEN.dataset == "MTL":
    dataset = reader.MTL_enc_dataset(csv_path=POPEN.csv_path,pad_to=POPEN.pad_to,columns=POPEN.aux_task_columns)
    loader_ls = reader.get_splited_dataloader(dataset,
                                            ratio=[0.8,0.2],
                                            batch_size=POPEN.batch_size,
                                            num_workers=8)
    train_loader = loader_ls[0]
    val_loader = loader_ls[1]

    
else:
    dataset = reader.UTR_dataset(cell_line=POPEN.cell_line)
    train_loader,val_loader,test_loader = reader.get_splited_dataloader(dataset,
                                                                        ratio=[0.7,0.1,0.2],
                                                                        batch_size=POPEN.batch_size,
                                                                        num_workers=4)
# ===========  setup model  ===========
    
# -- pretrain -- 
if POPEN.pretrain_pth is not None:
    # load pretran model
    pretrain_popen = Auto_popen(POPEN.pretrain_pth)
    pretrain_model = pretrain_popen.Model_Class(*pretrain_popen.model_args)
    utils.load_model(pretrain_popen,pretrain_model,logger)  
    
    # DL_models.LSTM_AE
    downstream_model = POPEN.Model_Class(*POPEN.model_args)
    
    # merge 
    model = MTL_models.Enc_n_Down(pretrain_model,downstream_model).cuda(POPEN.cuda_id)
    
# -- end2end -- 
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
# =========== resume ===========
best_loss = np.inf
best_acc = 0
best_epoch = 0
previous_epoch = 0
POPEN.train_mean = -0.0027311546037015106
POPEN.val_mean = 0.010948733354481285
if POPEN.Resumable:
    previous_epoch,best_loss,best_acc = utils.resume(POPEN,model,optimizer,logger)


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