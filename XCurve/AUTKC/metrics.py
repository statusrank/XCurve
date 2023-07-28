import torch

def evaluate(output, target, k_list=(1, )):
    return TopkAcc(output, target, k_list), AUTKC(output, target, k_list)


def TopkAcc(output, target, k_list=(1, )):
    """
    Computes the accuracy over the k top predictions for each k in the specified k-list
    """
    with torch.no_grad():
        maxk = max(k_list)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in k_list:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def AUTKC(output, target, k_list=(1, )):
    with torch.no_grad():
        s_y = output.gather(-1, target.view(-1, 1))
        s_top, _ = output.topk(max(k_list) + 1, 1, True, True)
        tmp = s_y > s_top

        res = []
        for k in k_list:
            tmp_k = tmp[:, :k + 1]
            autkc_k = torch.sum(tmp_k.float(), dim=-1) / k
            res.append(autkc_k)
        return [_.mean() * 100 for _ in res]