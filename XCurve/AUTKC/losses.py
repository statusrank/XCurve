import torch
import torch.nn as nn
from torch.nn.functional import one_hot

class CETopKLoss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x, y):
        s_y = x.gather(-1, y.view(-1, 1))
        a = x - s_y
        _, sorted_index = torch.sort(x.scatter(-1, y.view(-1, 1), float(torch.max(x))), dim=-1)
        m = x.size()[-1]
        s_bottom = a.gather(-1, sorted_index[:, :m - self.k])
        return torch.mean(torch.log(torch.sum(torch.exp(s_bottom), dim=-1) + 1))

class BaseHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        s_y = x.gather(-1, y.view(-1, 1))
        return torch.mean(torch.clamp_min(1 + x - s_y, 0))

class HingeTopKLoss(nn.Module):
    def __init__(self, k, loss_type=''):
        super().__init__()
        self.k = k
        self.loss_type = loss_type
        
    def forward(self, x, y):
        if self.loss_type == 'l1':
            x_no_y = x.scatter(-1, y.view(-1, 1), .0)
            sorted_data, _ = torch.sort(x_no_y, dim=-1, descending=True)
            s_topk = sorted_data[:, self.k - 1]
            s_y = x.gather(-1, y.view(-1, 1))
            return torch.mean(torch.clamp_min(1 + s_topk - s_y, 0))
        elif self.loss_type == 'l2':
            one_no_y = torch.ones_like(x).cuda().scatter(-1, y.view(-1, 1), 0)
            sorted_data, _ = torch.sort(x + one_no_y, dim=-1, descending=True)
            mean_sorted_data = torch.mean(sorted_data[:, :self.k], dim=-1)
            s_y = x.gather(-1, y.view(-1, 1))
            return torch.mean(torch.clamp_min(mean_sorted_data - s_y, 0))
        elif self.loss_type == 'l3':
            one_no_y = torch.ones_like(x).cuda().scatter(-1, y.view(-1, 1), 0)
            sorted_data, _ = torch.sort(x + one_no_y, dim=-1, descending=True)
            s_y = x.gather(-1, y.view(-1, 1))

            ret = torch.clamp_min(sorted_data[:, 0] - s_y, 0)
            for _ in range(1, self.k):
                ret += torch.clamp_min(sorted_data[:, _] - s_y, 0)
            return torch.mean(ret / self.k)
        elif self.loss_type == 'l4':
            x_no_y = x.scatter(-1, y.view(-1, 1), .0)
            sorted_data, _ = torch.sort(x_no_y, dim=-1, descending=True)
            s_topk = sorted_data[:, :self.k]
            s_y = x.gather(-1, y.view(-1, 1))
            return torch.mean(torch.clamp_min(1 + torch.mean(s_topk, dim=-1) - s_y, 0))
        elif self.loss_type == 'l5':
            sorted_data, _ = torch.sort(x, dim=-1, descending=True)
            s_topk = sorted_data[:, self.k]
            s_y = x.gather(-1, y.view(-1, 1))
            return torch.mean(torch.clamp_min(1 + s_topk - s_y, 0))

class StandardAUTKCLoss(nn.Module):
    def __init__(self, surrogate, K, epoch_to_paced):
        super().__init__()
        self.K = K
        self.epoch_to_paced = epoch_to_paced
        self.surrogate = surrogate
        self.paced_loss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred, y, epoch=0):
        if epoch < self.epoch_to_paced:
            print('paced..')
            return self.paced_loss(pred, y)
        else:
            num_class = pred.shape[1]
            target = one_hot(y, num_class)
            if self.surrogate != 'Hinge':
                pred = torch.softmax(pred, dim=-1)

            pred_p = pred[target.eq(1)].view(-1, 1)
            pred_n = pred[target.ne(1)].view(-1, num_class - 1)

            sort_pred_n, _ = torch.sort(pred_n, dim=-1, descending=True)
            top_pred_n  = sort_pred_n[:, :self.K + 1]
            
            if self.surrogate == 'Sq':
                loss = (1 + top_pred_n - pred_p) ** 2
            elif  self.surrogate == 'Exp':
                loss = torch.exp(top_pred_n - pred_p)
            elif  self.surrogate == 'Logit':
                loss = torch.log(1 + torch.exp(top_pred_n - pred_p))
            elif  self.surrogate == 'Hinge':
                loss = torch.clamp_min(1 + top_pred_n - pred_p, 0)

            loss = loss.sum(dim=-1) / self.K
            return loss.mean()