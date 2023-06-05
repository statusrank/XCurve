import numpy as np
import torch
import torch.nn.functional as F
import gc
from tqdm import tqdm


class Evaluator(object):
    def __init__(self):
        self.feats = []
        self.targets = []
        self.imgids = []

    def run(self, k_list=[1, 4, 16, 32]):
        feats = torch.cat(self.feats, 0).cuda()
        targets = torch.cat(self.targets, 0).cuda()

        i = 0
        gap = 1000
        rec_at_k = dict()
        for k in k_list:
            rec_at_k[k] = []
        AP = []
        AP_per_id = [[]] * (1 + targets.max().item())

        while i < len(targets):
            sim = torch.mm(feats[i:i+gap], feats.t())
            idx = torch.argsort(-sim)
            tg = targets[i:i+gap].clone()
            tg_gall = targets.clone().view(1,-1).repeat(gap, 1).gather(1, idx)[:, 1:]
            pos_mask = tg_gall == tg.view(-1, 1)
            # tg = targets.clone().view(1,-1).repeat(gap, 1).gather(1, idx)[:, 1:max(k_list) + 1]

            ## compute AP for each query
            for j in range(len(tg_gall)):
                pos_idx = pos_mask[j].nonzero(as_tuple=False).view(-1) + 1
                ap = (torch.arange(1, len(pos_idx) + 1).cuda() / pos_idx).mean()
                AP.append(ap)
                AP_per_id[targets[j].item()].append(ap)

            ## compute recall@k
            for k in k_list:
                rec_at_k[k].append((pos_mask[:, :k].sum(1) > 0).float())

            i += gap

        AP = torch.stack(AP, 0)
        results = {'mAP': AP.mean()}
        for k in k_list:
            rec_at_k[k] = torch.cat(rec_at_k[k], 0)
            results['recall@%d'%k] = rec_at_k[k].mean()

        return results, AP_per_id

        AP = []
        recall_k = dict()
        for k in k_list:
            recall_k[k] = []
        AP_per_id = dict()

        for i in range(len(targets)):
            sim = torch.mm(feats[i].unsqueeze(0), feats.t())
            sim_gall = torch.cat([sim[0, :i], sim[0, i+1:]], 0)
            target_gall = torch.cat([targets[:i], targets[i+1:]], 0)
            idx = torch.argsort(-sim_gall)
            target_gall = target_gall[idx]

            pos_idx = target_gall.eq(targets[i]).nonzero(as_tuple=False).view(-1) + 1
            assert len(pos_idx) > 0
            ap = (torch.arange(1, len(pos_idx) + 1).cuda() / pos_idx).mean().item()
            AP.append(ap)

            for k in k_list:
                rec = ((target_gall[:k] == targets[i]).sum() > 0)
                recall_k[k].append(rec)
 
            lbl = targets[i].item()
            if not lbl in AP_per_id:
                AP_per_id[lbl] = []
            AP_per_id[lbl].append(ap)

        mAP = sum(AP) / len(AP)
        results = {'mAP': mAP}
        for k in k_list:
            results['recall@%d'%k] = sum(recall_k[k]) / len(recall_k[k])

        return results, AP_per_id

    def pr_curve(self):
        feats = torch.cat(self.feats, 0)
        targets = torch.cat(self.targets, 0)

        # torch.save({
        #     'feats': feats,
        #     'targets': targets
        # }, 'tmp.pth')

        target_gall_all = []
        sim_gall_all = []

        pr = [[0,0] for _ in range(-1000, 1001)]
        for i in tqdm(range(len(targets))):
            sim = torch.mm(feats[i].unsqueeze(0), feats.t())
            sim_gall = torch.cat([sim[0, :i], sim[0, i+1:]], 0)
            target_gall = torch.cat([targets[:i], targets[i+1:]], 0)
            idx = torch.argsort(sim_gall)
            target_gall = target_gall[idx]
            sim_gall = sim_gall[idx]

            tgt = (target_gall.data.cpu().numpy() == int(targets[i])).astype(np.int32)
            sim = sim_gall.data.cpu().numpy()

            num_pos = tgt.sum()
            for j, thres in enumerate(range(-1000, 1001)):
                thres /= 1000

                idx = np.searchsorted(sim, thres)
                if idx == len(sim):
                    pr[j][0] += 1 / len(targets)
                    continue

                tp = tgt[idx:].sum()
                pr[j][1] += tp / num_pos / len(targets)
                pr[j][0] += tp / (len(sim) - idx) / len(targets)

        return pr
    
    def get_ap_per_id(self):
        feats = torch.cat(self.feats, 0)
        targets = torch.cat(self.targets, 0)
        imgids = torch.cat(self.imgids, 0).cpu().numpy()

        AP = dict()

        for i in range(len(targets)):
            sim = torch.mm(feats[i].unsqueeze(0), feats.t())
            sim_gall = torch.cat([sim[0, :i], sim[0, i+1:]], 0)
            target_gall = torch.cat([targets[:i], targets[i+1:]], 0)
            idx = torch.argsort(-sim_gall)
            target_gall = target_gall[idx]

            pos_idx = target_gall.eq(targets[i]).nonzero(as_tuple=False).view(-1) + 1
            assert len(pos_idx) > 0
            ap = (torch.arange(1, len(pos_idx) + 1).cuda() / pos_idx).mean()

            lbl = targets[i].item()
            if not lbl in AP:
                AP[lbl] = []
            AP[lbl].append(ap)

            # AP[int(imgids[i])] = ap.item()
        
        for k in AP.keys():
            AP[k] = sum(AP[k]) / len(AP[k])

        return AP
    
    def get_retrieval_one_id(self, target_id):
        feats = torch.cat(self.feats, 0).cuda()
        targets = torch.cat(self.targets, 0).cuda()
        imgids = torch.cat(self.imgids, 0).view(-1).cuda()
        for i in range(len(imgids)):
            if imgids[i] != target_id:
                continue
            sim = torch.mm(feats[i].unsqueeze(0), feats.t())
            sim_gall = torch.cat([sim[0, :i], sim[0, i+1:]], 0)
            imgids_gall = torch.cat([imgids[:i], imgids[i+1:]], 0)
            target_gall = torch.cat([targets[:i], targets[i+1:]], 0)
            idx = torch.argsort(-sim_gall)
            return imgids_gall[idx].cpu().numpy(), target_gall[idx].cpu().numpy()

        return None

    def add_batch(self, feat, target, imgid=None):
        self.feats.append(feat.data.cpu())
        self.targets.append(target.data.cpu())
        if imgid is not None:
            self.imgids.append(imgid.data.cpu())

    def reset(self):
        del self.feats
        del self.targets
        del self.imgids
        gc.collect()
        self.feats = []
        self.targets = []
        self.imgids = []
