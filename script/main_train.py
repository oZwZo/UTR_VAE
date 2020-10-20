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

from models import DL_models
from models import reader
from models import train_val
from models.ScheduleOptimizer import ScheduledOptim,scheduleoptim_dict_str
from models.popen import Auto_popen

parser = argparse.ArgumentParser('the main to train model')
parser.add_argument('--config_file',type=str,required=True)
args = parser.parse_args()

POPEN = Auto_popen(args.config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
dataset = reader.UTR_dataset(cell_line=POPEN.cell_line)
train_loader,val_loader,test_loader = reader.get_splited_dataloader(dataset,
                                                                    ratio=[0.7,0.1,0.2],
                                                                    batch_size=POPEN.batch_size,
                                                                    num_workers=4)
# ===========  setup model  ===========
Model_Class = POPEN.Model_Class  # DL_models.LSTM_AE
model = Model_Class(*POPEN.model_args).cuda()

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
                            weight_decay=1e-4)
# =========== resume ===========
best_loss = np.inf
best_acc = 0
best_epoch = 0
previous_epoch = 0
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
        best_epoch = epoch
    
    #    -----------| compare the result |-----------
        if (best_loss > val_total_loss) | (best_acc > val_avg_acc):
            # update best performance
            best_loss = min(best_loss,val_total_loss)
            best_acc = max(best_acc,val_avg_acc)
            
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
            
    elif (epoch - best_epoch > 2)&((type(optimizer) == ScheduledOptim)):
        optimizer.increase_delta()