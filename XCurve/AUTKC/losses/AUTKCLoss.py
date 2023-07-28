import torch
import torch.nn as nn
from torch.nn.functional import one_hot

class StandardAUTKCLoss(nn.Module):
    def __init__(self, surrogate, K, epoch_to_paced):
        super().__init__()
        self.K = K
        self.epoch_to_paced = epoch_to_paced
        assert surrogate in ['Sq', 'Exp', 'Logit', 'Hinge']
        self.surrogate = surrogate
        self.paced_loss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred, y, epoch=0):
        if epoch < self.epoch_to_paced:
            # print('paced..')
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