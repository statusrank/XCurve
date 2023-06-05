import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABCMeta, abstractmethod

from .base_loss import BaseLoss


class MemoryBlock(BaseLoss):

    def __init__(self, num_sample_per_id, num_id, mem_size_per_id, feat_dim, start_it=0, **kwargs):
        super(MemoryBlock, self).__init__()
        self.K = mem_size_per_id
        self.num_id = num_id
        self.start_it = start_it

        self.feats = torch.ones(num_id, self.K, feat_dim).cuda() / math.sqrt(feat_dim)
        self.targets = torch.ones(num_id, self.K, dtype=torch.long).cuda() * -1
        self.ptr = torch.zeros(self.num_id, dtype=torch.long).cuda()
        self.init = torch.zeros(self.num_id).cuda()

    def enqueue_dequeue(self, feats, targets):
        nc, ns, d = feats.shape
        tg = targets[:, 0]
        if self.init[tg].sum() < len(tg):
            # print('update memory with inplace op', self.init[tg].sum())
            self.init[tg] = True
            self.feats[tg] = feats.data.repeat(1, self.K // feats.shape[1] + 1, 1)[:, :self.K]
            self.targets[tg] = tg.data.unsqueeze(1).repeat(1, self.K)
        else:
            self.feats[tg] = torch.cat([
                feats.data, self.feats[tg, :-ns]
            ], dim=1)

        # ptr = self.ptr[targets[:, 0]].min()
        # if self.ptr + ns > self.K:
        #     self.is_full = True
        #     self.feats[targets[:, 0], -ns:] = feats
        #     self.targets[targets[:, 0], -ns:] = targets
        #     self.ptr = 0
        # else:
        #     self.feats[targets[:, 0], self.ptr: self.ptr + ns] = feats
        #     self.targets[targets[:, 0], self.ptr: self.ptr + ns] = targets
        #     self.ptr += ns

    def get(self, targets):
        return self.feats, self.targets
        # return self.feats[targets[:, 0]], self.targets[targets[:, 0]]

    def __call__(self, feats, targets):
        if self.start_it > 0:
            feats_mem, targets_mem = feats, targets
            self.start_it -= 1
        else:
            self.enqueue_dequeue(feats.data, targets)
            feats_mem, targets_mem = self.get(targets)

        return feats_mem, targets_mem

    def _criterion(self, x):
        pass


class AuxiliaryEmbedBlock(nn.Module):

    def __init__(self, num_sample_per_id, num_id, mem_size_per_id, feat_dim, start_it=0, **kwargs):
        super(AuxiliaryEmbedBlock, self).__init__()
        self.K = mem_size_per_id
        self.num_id = num_id
        self.start_it = start_it
        self.feats = torch.nn.Parameter(torch.ones(num_id, self.K, feat_dim).cuda(), requires_grad=True)
        # self.feats = None
        self.init = torch.zeros(num_id)

    def get(self, feats, targets):
        # if self.feats is None:
        #     self.feats = feats.clone().repeats(1, self.K//feats.shape[1] + 1, 1)[:,:self.K]
        #     self.feats = torch.nn.Parameter(self.feats)

        if self.init[targets[:, 0]].sum() < len(feats):
            self.init[targets[:, 0]] = 1
            self.feats.data[targets[:, 0]] = feats.clone().repeat(1, self.K//feats.shape[1] + 1, 1)[:,:self.K]

        f = torch.cat([feats, self.feats[targets[:, 0]]], dim=1)
        f = F.normalize(f, dim=-1)

        return f, targets[:, :1].repeat(1, self.K + targets.shape[1])
        # if self.is_full:
        #     return self.feats, self.targets
        # else:
        #     return self.feats[targets[:, 0], :self.ptr], self.targets[targets[:, 0], :self.ptr]

    def __call__(self, feats, targets):
        if self.start_it > 0:
            feats_mem, targets_mem = feats, targets
            self.start_it -= 1
        else:
            # self.enqueue_dequeue(feats.data, targets)
            feats_mem, targets_mem = self.get(feats, targets)

        return feats_mem, targets_mem

    def _criterion(self, x):
        pass

    @abstractmethod
    def forward(self, samples):
        pass
