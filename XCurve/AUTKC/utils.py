import random

import torch
import numpy as np
import torch.distributed as dist

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def AUTKC(output, target, topk=(1, )):
    with torch.no_grad():
        s_y = output.gather(-1, target.view(-1, 1))
        s_top, _ = output.topk(max(topk) + 1, 1, True, True)
        tmp = s_y > s_top

        res = []
        for k in topk:
            tmp_k = tmp[:, :k + 1]
            atop_k = torch.sum(tmp_k.float(), dim=-1) / k
            res.append(atop_k)
        return [_.mean() * 100 for _ in res]

def evaluate(output, target, topk=(1, )):
    return accuracy(output, target, topk), AUTKC(output, target, topk)

def adjust_learning_rate(optimizer, epoch, lr, epoch_to_adjust=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = lr * (0.1**(epoch // epoch_to_adjust))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    num_sample, num_class, K = 3, 5, 2
    pred = torch.rand((num_sample, num_class))
    target = torch.randint(num_class, size=(num_sample, ))

    print(pred)
    print(target)
    print(AUTKC(pred, target, K))