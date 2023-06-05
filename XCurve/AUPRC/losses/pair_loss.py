import torch
from abc import ABCMeta, abstractmethod

from numpy import nonzero

from .base_loss import BaseLoss


class PairLoss(BaseLoss):

    def __init__(self, num_sample_per_id, thres=None, norm=False, **kwargs):
        super(PairLoss, self).__init__()
        self.num_sample_per_id = num_sample_per_id
        self.thres = thres
        self.norm = norm

    def _check_input(self, targets):
        batch_size = targets.shape[0]
        targets = targets.view(
            batch_size // self.num_sample_per_id,
            self.num_sample_per_id
        )
        diff = targets - targets[:, 0].unsqueeze(1)
        assert diff.sum() == 0
        assert (targets == targets[0]).sum() == self.num_sample_per_id

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

        self._check_input(targets)

        mask = torch.block_diag(*([torch.ones(ns, ns) + torch.eye(ns)]*nc)).cuda()

        sim = torch.mm(feats, feats.t())
        sim_pos = torch.index_select(sim.view(-1), 0, mask.view(-1).eq(1).nonzero().view(-1))
        sim_neg = torch.index_select(sim.view(-1), 0, mask.view(-1).eq(0).nonzero().view(-1))

        loss = self._criterion(sim_pos.view(bs, ns - 1), sim_neg.view(bs, (nc - 1) * ns))

        return loss

class ContrastiveLoss(PairLoss):
    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)

    def _criterion(self, sim_pos, sim_neg):
        assert len(sim_pos.shape) == 2 and len(sim_neg.shape) == 2
        assert sim_pos.shape[0] == sim_neg.shape[0]
        bs, n_pos = sim_pos.shape
        _, n_neg = sim_neg.shape

        sim_pos = sim_pos.view(-1)
        sim_neg = sim_neg.view(-1)
        sim_pos = torch.index_select(sim_pos, 0, (sim_pos<self.thres[1]).nonzero().view(-1))
        sim_neg = torch.index_select(sim_neg, 0, (sim_neg>self.thres[0]).nonzero().view(-1))

        if len(sim_pos) > 0:
            loss_pos = (self.thres[1] - sim_pos).sum()
            if self.norm:
                loss_pos /= len(sim_pos)
        else:
            loss_pos = torch.tensor([.0]).cuda()
        
        if len(sim_neg) > 0:
            loss_neg = (sim_neg - self.thres[0]).sum()
            if self.norm:
                loss_neg /= len(sim_neg)
        else:
            loss_neg = torch.tensor([.0]).cuda()

        loss = loss_pos + loss_neg

        return loss.mean()

class TripletLoss(PairLoss):
    def __init__(self, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)

    def _criterion(self, sim_pos, sim_neg):
        assert len(sim_pos.shape) == 2 and len(sim_neg.shape) == 2
        assert sim_pos.shape[0] == sim_neg.shape[0]
        bs, n_pos = sim_pos.shape
        _, n_neg = sim_neg.shape

        loss = torch.clamp(sim_neg.view(bs, n_neg, 1) - sim_pos.view(bs, 1, n_pos) \
                + self.thres, min=0).view(bs, -1).sum(-1)

        return loss.mean()

class MultiSimilarityLoss(PairLoss):
    def __init__(self, alpha, beta, **kwargs):
        super(MultiSimilarityLoss, self).__init__(**kwargs)
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

class VarianceLoss(PairLoss):
    def __init__(self, inter_clss=False, **kwargs):
        super(VarianceLoss, self).__init__(**kwargs)
        self.inter_clss = inter_clss

    def _criterion(self, sim_pos, sim_neg):
        assert len(sim_pos.shape) == 2 and len(sim_neg.shape) == 2
        assert sim_pos.shape[0] == sim_neg.shape[0]

        mean_pos = sim_pos.mean()
        mean_neg = sim_neg.mean()
        target_pos = mean_pos
        target_neg = mean_neg
        diff_pos = (target_pos - sim_pos.view(-1))
        diff_neg = (sim_neg.view(-1) - target_neg)

        diff_pos = torch.index_select(diff_pos, 0, (diff_pos > 0).nonzero().view(-1))
        diff_neg = torch.index_select(diff_neg, 0, (diff_neg > 0).nonzero().view(-1))

        if len(diff_pos) > 0:
            loss_pos = (diff_pos**2).mean()
            # loss_pos = (diff_pos).mean()
        else:
            loss_pos = torch.tensor([.0]).cuda()
        if len(diff_neg) > 0:
            loss_neg = (diff_neg**2).mean()
            # loss_neg = (diff_neg).mean()
        else:
            loss_neg = torch.tensor([.0]).cuda()
        loss = self.thres[0] * loss_pos + self.thres[1] * loss_neg
        # loss = loss_pos + loss_neg

        return loss.mean()
