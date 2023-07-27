import torch
import torch.nn.functional as F
import numpy as np
import gc


class Evaluator(object):
    def __init__(self):
        self.feats = []
        self.targets = []

    def run_ap(self):
        feats = torch.cat(self.feats, 0).cuda()
        targets = torch.cat(self.targets, 0).cuda()

        i = 0
        gap = 1000
        AP = []

        while i < len(targets):
            sim = torch.mm(feats[i:i+gap], feats.t())
            idx = torch.argsort(-sim)
            tg = targets[i:i+gap].clone()
            tg_gall = targets.clone().view(1,-1).repeat(gap, 1).gather(1, idx)[:, 1:]
            pos_mask = tg_gall == tg.view(-1, 1)

            ## compute AP for each query
            for j in range(len(tg_gall)):
                pos_idx = pos_mask[j].nonzero(as_tuple=False).view(-1) + 1
                ap = (torch.arange(1, len(pos_idx) + 1).cuda() / pos_idx).mean()
                AP.append(ap)

            i += gap

        AP = torch.stack(AP, 0)
        return AP.mean().item()

    def run_rec_at_k(self, k_list=[1, 4, 16, 32]):
        feats = torch.cat(self.feats, 0).cuda()
        targets = torch.cat(self.targets, 0).cuda()

        i = 0
        gap = 1000
        rec_at_k = dict()
        for k in k_list:
            rec_at_k[k] = []

        while i < len(targets):
            sim = torch.mm(feats[i:i+gap], feats.t())
            idx = torch.argsort(-sim)
            tg = targets[i:i+gap].clone()
            tg_gall = targets.clone().view(1,-1).repeat(gap, 1).gather(1, idx)[:, 1:]
            pos_mask = tg_gall == tg.view(-1, 1)

            ## compute recall@k
            for k in k_list:
                rec_at_k[k].append((pos_mask[:, :k].sum(1) > 0).float())

            i += gap

        res = []
        for k in k_list:
            rec_at_k[k] = torch.cat(rec_at_k[k], 0)
            res.append(rec_at_k[k].mean().item())

        if len(k_list) == 1:
            return res[0]

        return res

    def add_batch(self, feat, target):
        self.feats.append(feat.data.cpu())
        self.targets.append(target.data.cpu())

    def reset(self):
        del self.feats
        del self.targets
        gc.collect()
        self.feats = []
        self.targets = []

def _data_preprocess(feat, target):
    if isinstance(feat, torch.Tensor):
        pass
    elif isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)
    else:
        raise NotImplementedError

    if isinstance(target, torch.Tensor):
        pass
    elif isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    else:
        raise NotImplementedError

    feat = F.normalize(feat, p=2, dim=1)
    return feat, target

def AUPRC(feats, targets):
    feats, targets = _data_preprocess(feats, targets)
    evaluator = Evaluator()
    evaluator.add_batch(feats, targets)
    return evaluator.run_ap()

def RecallAtK(feats, targets, k):
    feats, targets = _data_preprocess(feats, targets)
    if isinstance(k, int):
        k = [k]
    evaluator = Evaluator()
    evaluator.add_batch(feats, targets)
    return evaluator.run_rec_at_k(k)


__all__ = [AUPRC, RecallAtK]
