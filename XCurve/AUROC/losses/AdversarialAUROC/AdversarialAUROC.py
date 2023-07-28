# coding=utf-8
import torch
import torch.nn as nn

class AdvAUROCLoss(nn.Module):
    def __init__(self, imratio=None,
                 a=None,
                 b=None,
                 alpha=None):
                         
        super(AdvAUROCLoss, self).__init__()
        
        self.p = imratio
        if a is not None:
            self.a = torch.tensor(a).float().cuda()
            self.a.requires_grad = True
        else:
            self.a = torch.tensor(0.2).float().cuda()
            self.a.requires_grad = True
            # self.a = torch.zeros(1, dtype=torch.float32, requires_grad=True).cuda()
        
        if b is not None:
            self.b = torch.tensor(b).float().cuda()
            self.b.requires_grad = True
        else:
            # self.b = torch.zeros(1, dtype=torch.float32, requires_grad=True).cuda()
            self.b = torch.tensor(0.2).float().cuda()
            self.b.requires_grad = True
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha).float().cuda()
            self.alpha.requires_grad = True
        else:
            # self.alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True).cuda()
            self.alpha = torch.tensor(0.2).float().cuda()
            self.alpha.requires_grad = True
    
    def forward(self, y_pred, y_true):
        if self.p is None:
            self.p = (y_true == 1).float().sum() / y_true.shape[0]
        
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
               self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
               2 * self.alpha * (self.p * (1 - self.p) +
                                 torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (
                                             1 == y_true).float()))) - \
               self.p * (1 - self.p) * self.alpha ** 2
        return loss
    

class RegAdvAUROCLoss(AdvAUROCLoss):
    def __init__(self, imratio=None,
                 a=None,
                 b=None,
                 alpha=None,
                 lambda1=None,
                 lambda2=None) -> None:
        super(RegAdvAUROCLoss, self).__init__()
        if lambda1 is not None:
            self.lambda1 = torch.tensor(lambda1).float().cuda()
            self.lambda1.requires_grad = True
        else:
            self.lambda1 = torch.tensor(1.0).float().cuda()
            self.lambda1.requires_grad = True
        
        if lambda2 is not None:
            self.lambda2 = torch.tensor(lambda2).float().cuda()
            self.lambda2.requires_grad = True
        else:
            self.lambda2 = torch.tensor(1.0).float().cuda()
            self.lambda2.requires_grad = True
    
    def forward(self, y_pred, y_true):
        if self.p is None:
            self.p = (y_true == 1).float().sum() / y_true.shape[0]
        
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        
        loss = self.get_loss(y_pred, y_true)
        loss = loss - self.lambda1*(self.alpha + self.a) - self.lambda2*(self.alpha - self.b + 1)
        return loss

class selfloss(nn.Module):
    def __init__(self):
        super(selfloss, self).__init__()
    
    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        # new_y = y_true
        # new_y[new_y==0] = -1
        loss = torch.mean(-1 * (1 == y_true).float() * y_pred + (0 == y_true).float() * y_pred)
        return loss