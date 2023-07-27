import torch

def evaluate(output, target, k_list=(1, )):
    return TopkAcc(output, target, k_list), AUTKC(output, target, k_list)


def TopkAcc(y_pred, y_true, k_list=(1, )):
    """
    Computes the top-k accuracy for each k in the specified k-list.
    :param y_pred: the predictions of a multiclass model (batch_size, n_classes)
    :param y_true: the ground-truth labels (batch_size, )
    :param k_list: the list of the specified k
    :return: top-k accuracy
    :reference: ...
    """
    with torch.no_grad():
        maxk = max(k_list)
        batch_size = y_true.size(0)

        _, pred = y_pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y_true.view(1, -1).expand_as(pred))

        res = []
        for k in k_list:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def AUTKC(y_pred, y_true, k_list=(1, )):
    with torch.no_grad():
        s_y = y_pred.gather(-1, y_true.view(-1, 1))
        s_top, _ = y_pred.topk(max(k_list) + 1, 1, True, True)
        tmp = s_y > s_top

        res = []
        for k in k_list:
            tmp_k = tmp[:, :k + 1]
            autkc_k = torch.sum(tmp_k.float(), dim=-1) / k
            res.append(autkc_k)
        return [_.mean() * 100 for _ in res]