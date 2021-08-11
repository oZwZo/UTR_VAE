import torch
from torch import nn

class GMP(nn.Module):
    def __init__(self,encoder, out_channel,mode='max'):
        super().__init__()
        self.mode = mode
        self.soft_share = encoder
        for p in self.soft_share.parameters():
            p.requires_grad = False
        self.project_layer = nn.Linear(out_channel,1)
        self.loss_fn = nn.MSELoss()
        
    def forward(self,x):
        out = self.soft_share(x)
        
        max_activation = torch.max(out,axis=2)[0] if self.mode == 'max' else torch.mean(out,axis=2)
        return self.project_layer(max_activation)
    
    def squeeze_out_Y(self,out,Y):
        # ------ squeeze ------
        if len(Y.shape) == 2:
            Y = Y.squeeze(1)
        if len(out.shape) == 2:
            out = out.squeeze(1)
        
        assert Y.shape == out.shape
        return out,Y
    
    def compute_acc(self,out,X,Y,popen=None):
        try:
            epsilon = popen.epsilon
        except:
            epsilon = 0.3
            
        out,Y = self.squeeze_out_Y(out,Y)
        # error smaller than epsilon
        with torch.no_grad():
            acc = torch.sum(torch.abs(Y-out) < epsilon).item() / Y.shape[0]
        return {"Acc":acc}
    
    def compute_loss(self,out,X,Y,popen):
        out,Y = self.squeeze_out_Y(out,Y)
        loss = self.loss_fn(out,Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        return {"Total":loss}