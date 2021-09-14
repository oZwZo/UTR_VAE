import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

parser = argparse.ArgumentParser('the main to train model')
parser.add_argument('--config_file',type=str,required=True)
parser.add_argument('--cuda',type=int,default=None,required=False)
parser.add_argument("--kfold_index",type=int,default=None,required=False)
args = parser.parse_args()

cuda_id = args.cuda if args.cuda is not None else utils.get_config_cuda(args.config_file)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)


import time
import torch
import copy
import utils
from torch import optim
import numpy as np 
from models import MTL_models,reader,train_val
from models.ScheduleOptimizer import ScheduledOptim,scheduleoptim_dict_str
from models.popen import Auto_popen
from models.loss import Dynamic_Task_Priority,Dynamic_Weight_Averaging


POPEN = Auto_popen(args.config_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POPEN.cuda_id = device


POPEN.kfold_index = args.kfold_index
if POPEN.kfold_cv:
    if args.kfold_index is None:
        raise NotImplementedError("please specify the kfold index to perform K fold cross validation")
    POPEN.vae_log_path = POPEN.vae_log_path.replace(".log","_cv%d.log"%args.kfold_index)
    POPEN.vae_pth_path = POPEN.vae_pth_path.replace(".pth","_cv%d.pth"%args.kfold_index)
    

# Run name
if POPEN.run_name is None:
    run_name = POPEN.model_type + time.strftime("__%Y_%m_%d_%H:%M")
else:
    run_name = POPEN.run_name
    
# log dir
logger = utils.setup_logs(POPEN.vae_log_path)
logger.info(f"    ===========================| device {device}{cuda_id} |===========================    ")
#  built model dir or check resume 
POPEN.check_experiment(logger)
#                               |=====================================|
#                               |===========   setup  part  ==========|
#                               |=====================================|
# read data
loader_set = {}
base_path = POPEN.split_like_paper.copy()
for subset in POPEN.cycle_set:
    POPEN.split_like_paper = [path.replace('cycle', subset) for path in base_path]
    loader_set[subset] = reader.get_dataloader(POPEN)
# ===========  setup model  ===========
# train_iter = iter(train_loader)
# X,Y  = next(train_iter)
# -- pretrain -- 
if POPEN.pretrain_pth is not None:
    # load pretran model
    pretrain_popen = Auto_popen(POPEN.pretrain_pth)
    try:
        pretrain_model = pretrain_popen.Model_Class(*pretrain_popen.model_args)

        utils.load_model(pretrain_popen,pretrain_model,logger)
    except:
        pretrain_model = torch.load(pretrain_popen.vae_pth_path, map_location=torch.device('cpu'))['state_dict']

    
    if POPEN.Model_Class == pretrain_popen.Model_Class:
        # if not POPEN.Resumable:
        #     # we only load pre-train for the first time 
        #     # later we can resume 
        model = pretrain_model.to(device)
        del pretrain_model
    elif POPEN.modual_to_fix in dir(pretrain_model):    
        model = POPEN.Model_Class(*POPEN.model_args)
        model.soft_share.load_state_dict(pretrain_model.soft_share.state_dict())
        model =  model.to(device)
    else:
        downstream_model = POPEN.Model_Class(*POPEN.model_args)

        # merge 
        model = MTL_models.Enc_n_Down(pretrain_model,downstream_model).to(device)
    
# -- end2end -- 
elif POPEN.model_type == "CrossStitch_Model":
    backbone = {}
    for t in POPEN.tasks:
        task_popen = Auto_popen(POPEN.backbone_config[t])
        task_model = task_popen.Model_Class(*task_popen.model_args)
        utils.load_model(task_popen,task_model,logger)
        backbone[t] = task_model.to(device)
    POPEN.model_args = [backbone] + POPEN.model_args
    model = POPEN.Model_Class(*POPEN.model_args).to(device)
else:
    Model_Class = POPEN.Model_Class  # DL_models.LSTM_AE 
    model = Model_Class(*POPEN.model_args).to(device)
    
if POPEN.Resumable:
    model = utils.load_model(POPEN, model, logger)
    
# =========== fix parameters ===========
if isinstance(POPEN.modual_to_fix, list):
    for modual in POPEN.modual_to_fix:
        model = utils.fix_parameter(model,modual)
    logger.info(' \t \t ==============<<< %s part is fixed>>>============== \t \t \n'%POPEN.modual_to_fix)
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
    previous_epoch,best_loss,best_acc = utils.resume(POPEN, optimizer,logger)
    

#                               |=====================================|
#                               |==========  training  part ==========|
#                               |=====================================|
for epoch in range(POPEN.max_epoch-previous_epoch+1):
    epoch += previous_epoch
    
    #          
    logger.info("===============================|    epoch {}   |===============================".format(epoch))
    for subset in POPEN.cycle_set:
        logger.info("    ===========================|  set: {} |===========================    ".format(subset))
        #    ----------|switch tower and train |----------
        model.task = subset
        # model = utils.fix_parameter(model,POPEN.modual_to_fix[0])
        # logger.info("        =======================|     fix      |=======================        ")
        train_val.train(dataloader=loader_set[subset][0],model=model,optimizer=optimizer,popen=POPEN,epoch=epoch)
        
        # model = utils.unfix_parameter(model,POPEN.modual_to_fix[0])
        # logger.info("        =======================|    unfix     |=======================        ")
        # train_val.train(dataloader=loader_set[subset][0],model=model,optimizer=optimizer,popen=POPEN,epoch=epoch)

    #              -----------| validate |-----------   
        logger.info("===============================| start validation |===============================")
        val_total_loss,val_avg_acc = train_val.cycle_validate(loader_set,model,optimizer,popen=POPEN,epoch=epoch)
        # matching task performance influence what to save
        
        
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
                        # 'state_dict': model.state_dict(),
                        'state_dict': model,
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