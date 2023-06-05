import torch
from abc import ABCMeta, abstractmethod

from .base_loss import BaseLoss


class AUCLoss(BaseLoss):

    def __init__(self, num_sample_per_id, temp, **kwargs):
        super(AUCLoss, self).__init__()
        self.num_sample_per_id = num_sample_per_id
        self.temp = temp
    
    def check_input(self, targets):
        batch_size = targets.shape[0]
        targets = targets.view(
            batch_size // self.num_sample_per_id,
            self.num_sample_per_id
        )
        diff = targets - targets[:, 0].unsqueeze(1)
        assert diff.sum() == 0

    @abstractmethod
    def surrogate_fn(self, x):
        pass

    def forward(self, samples):
        feats = samples['feat']
        targets = samples['target']
        batch_size = targets.shape[0]
        nc = batch_size // self.num_sample_per_id
        ns = self.num_sample_per_id

        self.check_input(targets)

        sim = torch.mm(feats, feats.t())
        mask = torch.block_diag(*([torch.ones(ns, ns)]*nc)).cuda()

        sim_pos = torch.index_select(sim.view(-1), 0, mask.view(-1).eq(1).nonzero().view(-1))
        sim_pos = sim_pos.view(batch_size, ns, 1)

        sim_neg = torch.index_select(sim.view(-1), 0, mask.view(-1).eq(0).nonzero().view(-1))
        sim_neg = sim_neg.view(batch_size, 1, (nc - 1)*ns)

        loss = self.surrogate_fn(sim_pos - sim_neg).mean()
        return loss

class AUCLogitLoss(AUCLoss):
    def __init__(self, **kwargs):
        super(AUCLogitLoss, self).__init__(**kwargs)
    
    def surrogate_fn(self, x):
        return torch.log2(1 + torch.exp(-x))

class AUCHuberLoss(AUCLoss):
    def __init__(self, **kwargs):
        super(AUCHuberLoss, self).__init__(**kwargs)
    
    def surrogate_fn(self, x, temp=0.1):
        temp = self.temp
        x = torch.clamp(x, max=temp)
        return torch.where(
            x >= 0,
            (x / temp - 1)**2,
            (-2 * x / temp + 1)
        )

class AUCSigmoidLoss(AUCLoss):
    def __init__(self, **kwargs):
        super(AUCSigmoidLoss, self).__init__(**kwargs)
    
    def surrogate_fn(self, x):
        return torch.sigmoid(-x/self.temp)
