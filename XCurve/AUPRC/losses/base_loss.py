import torch
import torch.nn as nn

from abc import abstractmethod


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        pass

    def feat2sim(self, feats, targets):
        x = torch.mm(feats, feats.t()) # N*N
        label = (targets.view(-1, 1) == targets.view(1, -1)).long()
        return x, label

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    def param_groups(self, lr=None, **kwargs):
        params = list(filter(lambda x:x.requires_grad, self.parameters()))
        if len(params):
            if lr is not None:
                return [{'params': params, 'lr': lr}]
            else:
                return [{'params': params}]
        else:
            return []
