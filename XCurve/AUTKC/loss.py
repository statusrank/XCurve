from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

class Hinge(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        s_y = x.gather(-1, y.view(-1, 1))
        return torch.mean(torch.clamp_min(1 + x - s_y, 0))


class Loss1(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x, y):
        x_no_y = x.scatter(-1, y.view(-1, 1), .0)
        sorted_data, _ = torch.sort(x_no_y, dim=-1, descending=True)
        s_topk = sorted_data[:, self.k - 1]
        s_y = x.gather(-1, y.view(-1, 1))
        return torch.mean(torch.clamp_min(1 + s_topk - s_y, 0))

class Loss2(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x, y):
        one_no_y = torch.ones_like(x).cuda().scatter(-1, y.view(-1, 1), 0)
        sorted_data, _ = torch.sort(x + one_no_y, dim=-1, descending=True)
        mean_sorted_data = torch.mean(sorted_data[:, :self.k], dim=-1)
        s_y = x.gather(-1, y.view(-1, 1))
        return torch.mean(torch.clamp_min(mean_sorted_data - s_y, 0))

class Loss3(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x, y):
        one_no_y = torch.ones_like(x).cuda().scatter(-1, y.view(-1, 1), 0)
        sorted_data, _ = torch.sort(x + one_no_y, dim=-1, descending=True)
        s_y = x.gather(-1, y.view(-1, 1))

        ret = torch.clamp_min(sorted_data[:, 0] - s_y, 0)
        for _ in range(1, self.k):
            ret += torch.clamp_min(sorted_data[:, _] - s_y, 0)
        return torch.mean(ret / self.k)

class Loss4(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x, y):
        x_no_y = x.scatter(-1, y.view(-1, 1), .0)
        sorted_data, _ = torch.sort(x_no_y, dim=-1, descending=True)
        s_topk = sorted_data[:, :self.k]
        s_y = x.gather(-1, y.view(-1, 1))
        return torch.mean(torch.clamp_min(1 + torch.mean(s_topk, dim=-1) - s_y, 0))

class Loss5(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x, y):
        sorted_data, _ = torch.sort(x, dim=-1, descending=True)
        s_topk = sorted_data[:, self.k]
        s_y = x.gather(-1, y.view(-1, 1))
        return torch.mean(torch.clamp_min(1 + s_topk - s_y, 0))

class TopKCE(nn.Module):
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

class AUTKCLoss(nn.Module):
    def __init__(self, weight_scheme, K, epoch_to_paced):
        super().__init__()
        self.K = K
        self.epoch_to_paced = epoch_to_paced
        self.weight_scheme = weight_scheme
        self.paced_loss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred, y, epoch=0):
        if epoch < self.epoch_to_paced:
            print('paced..')
            return self.paced_loss(pred, y)
        else:
            num_class = pred.shape[1]
            target = one_hot(y, num_class)
            if self.weight_scheme != 'Naive_hinge':
                pred = torch.softmax(pred, dim=-1)

            pred_p = pred[target.eq(1)].view(-1, 1)
            pred_n = pred[target.ne(1)].view(-1, num_class - 1)

            sort_pred_n, _ = torch.sort(pred_n, dim=-1, descending=True)
            top_pred_n  = sort_pred_n[:, :self.K + 1]
            
            if self.weight_scheme == 'Naive':
                loss = (1 + top_pred_n - pred_p) ** 2
            elif  self.weight_scheme == 'NaiveExp':
                loss = torch.exp(top_pred_n - pred_p)
            elif  self.weight_scheme == 'NaiveLogit':
                loss = torch.log(1 + torch.exp(top_pred_n - pred_p))
            elif  self.weight_scheme == 'NaiveHinge':
                loss = torch.clamp_min(1 + top_pred_n - pred_p, 0)

            loss = loss.sum(dim=-1) / self.K
            
            return loss.mean()

    def re_weight(self, pred_n, eps=1e-6):
        if self.weight_scheme == 'Poly':
            row_pred_n = torch.pow(pred_n + eps, self.K)
        elif self.weight_scheme == 'Exp':
            row_pred_n = 1 - torch.exp(- self.K * pred_n)
        else:
            raise ValueError

        return row_pred_n