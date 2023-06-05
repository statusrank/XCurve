import torch
from abc import ABCMeta, abstractmethod

from .base_loss import BaseLoss


class XBMLoss(BaseLoss):

    def __init__(self, num_sample_per_id, thres, mem_size, feat_dim, start_it=0, **kwargs):
        super(XBMLoss, self).__init__()
        self.num_sample_per_id = num_sample_per_id
        self.thres = thres
        self.K = mem_size
        self.start_it = start_it

        self.feats = torch.zeros(self.K, feat_dim).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0
        self.is_full = False

    def _check_input(self, targets):
        batch_size = targets.shape[0]
        targets = targets.view(
            batch_size // self.num_sample_per_id,
            self.num_sample_per_id
        )
        diff = targets - targets[:, 0].unsqueeze(1)
        assert diff.sum() == 0
        assert (targets == targets[0]).sum() == self.num_sample_per_id

    def enqueue_dequeue(self, feats, targets):  
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.is_full = True
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    @abstractmethod
    def _criterion(self, x):
        pass

    def forward(self, samples):

        feats = samples['feat']
        targets = samples['target']
        bs = targets.shape[0]
        nc = bs // self.num_sample_per_id
        ns = self.num_sample_per_id
        d = feats.shape[-1]

        if self.start_it > 0:
            feats_mem, targets_mem = feats, targets
            self.start_it -= 1
        else:
            self.enqueue_dequeue(feats.data, targets)
            feats_mem, targets_mem = self.get()

        self._check_input(targets)

        loss = 0
        sim = torch.mm(feats, feats_mem.t())
        for i in range(bs):
            pos_idx_i = (targets[i] == targets_mem.view(-1)).nonzero().view(-1)
            neg_idx_i = (targets[i] != targets_mem.view(-1)).nonzero().view(-1)
            if len(pos_idx_i) == 0:
                continue
            sim_pos = torch.index_select(sim[i].view(-1), 0, pos_idx_i)
            sim_neg = torch.index_select(sim[i].view(-1), 0, neg_idx_i)
            loss += self._criterion(sim_pos.view(1, -1), sim_neg.view(1, -1))

        return loss / len(feats)

class XBMContrastiveLoss(XBMLoss):
    def __init__(self, **kwargs):
        super(XBMContrastiveLoss, self).__init__(**kwargs)

    def _criterion(self, sim_pos, sim_neg):
        assert len(sim_pos.shape) == 2 and len(sim_neg.shape) == 2
        assert sim_pos.shape[0] == sim_neg.shape[0]
        bs, n_pos = sim_pos.shape
        _, n_neg = sim_neg.shape

        loss = torch.clamp(sim_neg - self.thres, min=0).sum(-1) + \
                (1 - sim_pos).sum(-1)

        return loss.mean()

class XBMTripletLoss(XBMLoss):
    def __init__(self, **kwargs):
        super(XBMTripletLoss, self).__init__(**kwargs)

    def _criterion(self, sim_pos, sim_neg):
        assert len(sim_pos.shape) == 2 and len(sim_neg.shape) == 2
        assert sim_pos.shape[0] == sim_neg.shape[0]
        bs, n_pos = sim_pos.shape
        _, n_neg = sim_neg.shape

        loss = torch.clamp(sim_neg.view(bs, n_neg, 1) - sim_pos.view(bs, 1, n_pos) \
                + self.thres, min=0).view(bs, -1).sum(-1)

        return loss.mean()

class XBMMultiSimilarityLoss(XBMLoss):
    def __init__(self, alpha, beta, **kwargs):
        super(XBMMultiSimilarityLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def _criterion(self, sim_pos, sim_neg):
        assert len(sim_pos.shape) == 2 and len(sim_neg.shape) == 2
        assert sim_pos.shape[0] == sim_neg.shape[0]
        bs, n_pos = sim_pos.shape
        _, n_neg = sim_neg.shape
        alpha = self.alpha
        beta = self.beta

        loss = 1/alpha * torch.log(1 + torch.exp(-alpha*(sim_pos - self.thres)).sum(-1)) + \
             1/beta * torch.log(1 + torch.exp(beta*(sim_neg - self.thres)).sum(-1))

        return loss.mean()
