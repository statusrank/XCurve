import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

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
    
# refer to Fig.2 of the paper "Nearest neighbors distance ratio open-set classifier" 
def MacroF(close_pred, close_labels, open_pred, open_labels, t_list):
    ret = list()
    for t in t_list:
        tp, fp, fn = 0, 0, 0
        close_pred_ = np.argmax(close_pred, axis=1)
        open_pred_ = (open_pred > t) * 1
        for c_pred, o_pred, cl, ol in zip(close_pred_, open_pred_, close_labels, open_labels):
            tp += 1 if c_pred == cl and o_pred == 0 and ol == 0 else 0
            fp += 1 if c_pred != cl and o_pred == 0 and ol == 0 else 0
            fp += 1 if o_pred == 0 and ol == 1 else 0
            fn += 1 if c_pred != cl and o_pred == 0 and ol == 0 else 0
            fn += 1 if o_pred == 1 and ol == 0 else 0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        ret.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    # m_ret = max(ret)
    # print(f'M-F-score: {m_ret}', ret)
    return ret

def MicroF(close_pred, close_labels, open_pred, open_labels, t_list):
    ret = list()
    n_class = close_pred.shape[1]
    for t in t_list:
        tp, fp, fn = [0, ] * n_class, [0, ] * n_class, [0, ] * n_class
        close_pred_ = np.argmax(close_pred, axis=1)
        open_pred_ = (open_pred > t) * 1
        for c_pred, o_pred, cl, ol in zip(close_pred_, open_pred_, close_labels, open_labels):
            tp[cl] += 1 if c_pred == cl and o_pred == 0 and ol == 0 else 0
            fp[c_pred] += 1 if c_pred != cl and o_pred == 0 and ol == 0 else 0
            fp[c_pred] += 1 if o_pred == 0 and ol == 1 else 0
            fn[cl] += 1 if c_pred != cl and o_pred == 0 and ol == 0 else 0
            fn[cl] += 1 if o_pred == 1 and ol == 0 else 0
        p = sum([tpi / (tpi + fpi) if (tpi + fpi) > 0 else 0 for tpi, fpi in zip(tp, fp)]) / n_class
        r = sum([tpi / (tpi + fni) if (tpi + fni) > 0 else 0 for tpi, fni in zip(tp, fn)]) / n_class
        ret.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    # m_ret = max(ret)
    # print(f'm-F-score: {m_ret}', ret)
    return ret

def ClosedSetAcc(preds, labels):
    preds = preds.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    # print('Closed Set Accuracy: {:.3f}'.format(acc))
    return acc

def find_nearest(array, value):
    array = np.asarray(array)
    length = len(array)
    abs_diff = np.abs(array - value)

    t_star = abs_diff.min()
    equal_arr = (abs_diff == t_star).astype('float32') + np.linspace(start=0, stop=0.1, num=length)

    idx = equal_arr.argmax()

    return array[idx], idx

def Acc_At_T(preds, labels, t):
    pred_t = np.copy(preds)
    pred_t[pred_t > t] = 1
    pred_t[pred_t <= t] = 0

    acc = accuracy_score(labels, pred_t.astype('int32'))

    return acc

def Acc_At_TPR(open_set_preds, open_set_labels, thresholds, tpr, r=0.95):
    # Error rate at r TAR
    _, idx = find_nearest(tpr, r)
    t = thresholds[idx]
    acc = Acc_At_T(open_set_preds, open_set_labels, t)
    return acc

def AUROC(open_set_preds, open_set_labels):
    auroc = roc_auc_score(open_set_labels, open_set_preds)
    # print(f'AUROC: {auroc}')
    return auroc

def OpenAUC(open_set_pred_known, open_set_pred_unknown, close_set_pred_class, close_set_labels):
    """
    :param open_set_pred_known: open set score for each known class sample (B_k,)
    :param open_set_pred_unknown: open set score for each unknown class sample (B_u,)
    :param close_set_pred_class: predicted class for each known class sample (B_k,)
    :param close_set_labels: correct class for each known class sample (B_k,)
    :return: OpenAUC
    """
    open_set_pred_known, open_set_pred_unknown, correct = open_set_pred_known.tolist(), open_set_pred_unknown.tolist(), (close_set_pred_class == close_set_labels).tolist()
    m_x2 = max(open_set_pred_unknown) + 1e-5
    y_score = [value if hit else m_x2 for value, hit in zip(open_set_pred_known, correct)] + open_set_pred_unknown
    y_true = [0] * len(open_set_pred_known) + [1] * len(open_set_pred_unknown)
    open_auc = roc_auc_score(y_true, y_score)
    # print('OpenAUC:', open_auc)
    return open_auc