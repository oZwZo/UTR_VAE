import torch
from torch import nn
import torch.nn.functional as F


class channel_wise_mul(nn.Module):

    def __init__(self,num_feature):
        """
        multiply all the channel with the a single weight
        """
        super(CrossStitch_Unit,self).__init__()
        
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
        self.param_dict = nn.ModuleDict({ task_i : { task_j : channel_wise_mul(num_feature) for task_j in tasks }  for task_i in tasks  })
        
        # initiate info exchanging weight
        for task_i in tasks:
            for task_j in tasks:
                values = alpha if task_i == task_j else beta
                self.param_dict[task_i][task_j]._weight_ini_(values)
        
    def forward(self,task_feature):
        """
        perform infomation exchange between tasks by linear combination of task activation map
        Argument:
           task_feature : dict, {task: stage_out} key should be exatly the same with tasks
        """
        assert list(task_feature.keys()) == tasks
        
        out = {}

        for t_i in self.tasks:
            param_i = self.param_dict[t_i]
            prod = torch.stack([param_i[t_j](task_feature[t_j]) for t_j in self.tasks])  # stack at dim 0 ?
            out[t_i] = torch.sum(prod,dim=0)
        
        return out
                