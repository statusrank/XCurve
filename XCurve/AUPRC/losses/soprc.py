import torch
import math

from .base_loss import BaseLoss


class ListStableAUPRCLoss(BaseLoss):

    def __init__(self, num_sample_per_id, tau, beta, prior_mul=0.1, **kwargs):
        super(ListStableAUPRCLoss, self).__init__()

        self.num_sample_per_id = num_sample_per_id
        self.temp = tau
        self.beta = beta
        self.prior_mul = prior_mul
        self.num_v = 100
        self.v = None
        self.alpha = None

    def update_cnt_per_id(self, cnt_per_id):
        if self.prior_mul < 0:
            self.prior = None
            return

        self.cnt_per_id = cnt_per_id
        self.v = torch.stack([torch.arange(self.num_v) / self.num_v for _ in range(len(cnt_per_id))]).cuda()
        self.prior = torch.tensor(cnt_per_id).float().cuda()
        self.prior *= self.prior_mul / self.prior.max()

    def _check_input(self, targets):
        assert self.prior_mul < 0 or self.v is not None
        batch_size = targets.shape[0]
        targets = targets.view(
            batch_size // self.num_sample_per_id,
            self.num_sample_per_id
        )
        diff = targets - targets[:, 0].unsqueeze(1)
        assert diff.sum() == 0
        assert (targets == targets[0]).sum() == self.num_sample_per_id

    def _surrogate_fn_pn(self, x):
        temp = self.temp[0]
        x = torch.clamp(x, max=temp)
        return torch.where(
            x >= 0,
            (x / temp - 1)**2,
            (-2 * x / temp + 1)
        )

    def _surrogate_fn_pp(self, x):
        temp = self.temp[1]
        x = torch.clamp(x, max=0)
        return 2 / (1 + torch.exp(x / temp)) - 1

    def forward(self, feats, targets):
        ## preprocess input 
        targets = targets.view(-1)
        batch_size = targets.shape[0]
        nc = batch_size // self.num_sample_per_id
        ns = self.num_sample_per_id
        self._check_input(targets)

        ## compute similarity scores
        mask = torch.block_diag(*([torch.ones(ns, ns) + torch.eye(ns)]*nc)).cuda()
        sim = torch.mm(feats, feats.t())
        sim_pos = torch.index_select(sim.view(-1), 0, mask.view(-1).eq(1).nonzero().view(-1))
        sim_pos = sim_pos.view(batch_size, ns - 1, 1)
        sim_pos = torch.sort(sim_pos, 1)[0]
        sim_neg = torch.index_select(sim.view(-1), 0, mask.view(-1).eq(0).nonzero().view(-1))
        sim_neg = sim_neg.view(batch_size, 1, (nc - 1)*ns)

        ## l_1
        loss_pn = self._surrogate_fn_pn(sim_pos - sim_neg).mean(-1)

        ## update v by score intp
        with torch.no_grad():
            beta = self.beta
            if self.alpha is None:
                len_intv = math.ceil(self.num_v / (ns-1))
                self.alpha = torch.arange(len_intv).view(1,1,-1) / len_intv
                self.alpha = self.alpha.cuda()

            score = sim_pos.clone()
            score_left = torch.cat([
                torch.clamp(2*score[:, 0] - score[:, 1], min=-1).unsqueeze(1),
                score[:, :-1]
            ], dim=1)
            v_left = (score - score_left) * self.alpha + score_left

            score_right = torch.cat([
                score[:, 1:],
                torch.clamp(2*score[:, -1] - score[:, -2], max=1).unsqueeze(1)
            ], dim=1)
            v_right = (score_right - score) * self.alpha + score

            self.v[targets] = (1-beta) * self.v[targets] + beta / 2 * (v_left + v_right).view(batch_size, -1)[:, :self.num_v]

            ## l_2
            v = torch.index_select(self.v, 0, targets)

            loss_pv = self._surrogate_fn_pp(sim_pos - v.view(batch_size, 1, -1))
            loss_pv = (1 + loss_pv.sum(-1)) / (loss_pv.shape[-1] + 1)

        ## biased estimation of auprc
        if self.prior is not None:
            prior = torch.index_select(self.prior, 0, targets).unsqueeze(1)
        else:
            prior = 1 / nc
        loss = (1 - prior) / prior * loss_pn / loss_pv
        g_loss = (loss / (1 + loss)).mean()

        return g_loss
