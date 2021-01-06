import torch
from torch import nn
import torch.nn.functional as F


class channel_wise_mul(nn.Module):

    def __init__(self,num_feature):
        """
        multiply all the channel with the a single weight
        """
        super(channel_wise_mul,self).__init__()
        
        self.param = nn.Parameter(torch.FloatTensor(num_feature), requires_grad=True)
    
    def _weight_init_(self,value):
        self.param.data.fill_(value)
    
    def forward(self,x):
        return torch.mul(self.param.view(1,-1,1),x)

class CrossStitch_Unit(nn.Module):
    def __init__(self,tasks,num_feature,alpha,beta):
        """
        a cross_stitch layer which can sharing information between tasks
        Arguments:
            tasks : list , [str]
            num_feature : the dimension of channel at a certain stage 
            alpha : float, 0 ~ 1 , how much lcoal information retained  
            beta  : float,1-alpha , how much information to share  
        """
        super(CrossStitch_Unit,self).__init__()
        
        self.tasks = tasks
         # {i:{ j: layer } }
        self.param_dict = nn.ModuleDict({ task_i : nn.ModuleDict({ task_j : channel_wise_mul(num_feature) for task_j in tasks })  for task_i in tasks  })
        
        # initiate info exchanging weight
        for task_i in tasks:
            for task_j in tasks:
                values = alpha if task_i == task_j else beta
                self.param_dict[task_i][task_j]._weight_init_(values)
        
    def forward(self,task_feature):
        """
        perform infomation exchange between tasks by linear combination of task activation map
        Argument:
           task_feature : dict, {task: stage_out} key should be exatly the same with tasks
        """
        assert list(task_feature.keys()) == self.tasks
        
        out = {}

        for t_i in self.tasks:
            param_i = self.param_dict[t_i]
            prod = torch.stack([param_i[t_j](task_feature[t_j]) for t_j in self.tasks])  # stack at dim 0 ?
            out[t_i] = torch.sum(prod,dim=0)
        
        return out
    
class CrossStitch_Model(nn.Module):
    def __init__(self,backbone,tasks,alpha,beta):
        """
        Implementation of cross-stitch networks.
        We insert a cross-stitch unit, to combine features from the task-specific backbones
        after every stage.
        Arguments:
            backbone : Dict {t:backbone }
            tasks  : list [t ]
            alpha : float, 0 ~ 1 , how much lcoal information retained  
            beta  : float,1-alpha , how much information to share   
        """    
        super(CrossStitch_Model,self).__init__()
        self.backbone = nn.ModuleDict(backbone)
        self.tasks = tasks
        self.alpha = alpha
        self.beta = beta
        self.check_consistance()
        self.cross_stitch_link = nn.ModuleList(
            [CrossStitch_Unit(tasks,self.channel_ls[stage+1],alpha,beta) for stage in self.stage]
        )     # cross stitch is channel- wise multiply, # of channel after `Conv1d` is `channel_ls[i+1]`
        
    def check_consistance(self):        
        """
        check the accordance of model architectrue among tasks and stage
        """
        assert set(list(self.backbone.keys())) == set(self.tasks) , "tasks disaccordant"
        self.stage = self.backbone[self.tasks[0]].stage
        self.channel_ls = self.backbone[self.tasks[0]].soft_share.channel_ls
        
    def forward_stage(self,task_feature,stage):
        # calculate local feature
        task_feature =  {t:self.backbone[t].forward_stage(task_feature[t],stage) for t in self.tasks}
        # exchange by cross_stitch
        crossing_feature = self.cross_stitch_link[stage](task_feature)
        return crossing_feature  
        
    def forward(self,X):
        X = X.transpose(1,2) if X.shape[2] == 4 else X
        task_feature = {t:X for t in self.tasks}
        for stage in self.stage:
            task_feature = self.forward_stage(task_feature,stage)
        
        out = {t:self.backbone[t].forward_tower(task_feature[t]) for t in self.tasks}
        
        return out
    
    def compute_acc(self,out,Y):
        """
        out : dict {task:out}
        Y : dict {task : y}
        """
        (x,y) = Y 
        Y = {"RL":y[:,0], "Recons":x,"Motif":y[:,1:]}
        acc_dict  = {}
        for t in self.tasks:
            backbone = self.backbone[t]
            t_out = out[t]
            t_Y = Y[t]
            acc_dict[t+"_Acc"] = backbone.compute_acc(t_out,t_Y)
        return acc_dict
    
    def compute_loss(self,out,Y,popen):
        """
        out : dict {task:out}
        Y : dict {task : y}
        """
        assert ('rl' in popen.aux_task_columns[0])
        (x,y) = Y 
        Y = {"RL":y[:,0], "Recons":x,"Motif":y[:,1:]}
        loss_dict  = {}
        for t in self.tasks:
            backbone = self.backbone[t]
            t_out = out[t]
            t_Y = Y[t]
            t_loss = backbone.compute_loss(t_out,t_Y,popen)
            loss_dict[t+"_loss"] = t_loss['Total']
        
        loss_dict['Total'] = torch.stack([loss_dict[t+'_loss']*popen.chimerla_weight[t] for t in self.tasks]).sum()
        return loss_dict
        