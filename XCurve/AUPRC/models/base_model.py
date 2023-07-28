import torch
import torch.nn as nn
from abc import ABC


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def freezeBN(self):
        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                m.eval()
                m.train = lambda _:None

    def param_groups(self, lr=None, **kwargs):
        params = list(filter(lambda x:x.requires_grad, self.parameters()))

        if len(params):
            if lr is not None:
                return [{'params': params, 'lr': lr}]
            else:
                return [{'params': params}]
        else:
            return []
