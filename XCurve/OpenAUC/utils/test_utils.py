import torch
import numpy as np
import os

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score

from tqdm import tqdm

# refer to Fig.2 of the paper "Nearest neighbors distance ratio open-set classifier" 
def ma_f_score(close_pred, close_labels, open_pred, open_labels, ts):
    ret = list()
    for t in ts:
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
    m_ret = max(ret)
    print(f'M-F-score: {m_ret}', ret)
    return ret

def mi_f_score(close_pred, close_labels, open_pred, open_labels, ts):
    ret = list()
    n_class = close_pred.shape[1]
    for t in ts:
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
    m_ret = max(ret)
    print(f'm-F-score: {m_ret}', ret)
    return ret

def normalised_average_precision(y_true, y_pred):

    from sklearn.metrics._ranking import _binary_clf_curve

    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=None,
                                             sample_weight=None)

    n_pos = np.array(y_true).sum()
    n_neg = (1 - np.array(y_true)).sum()

    precision = tps * n_pos / (tps * n_pos + fps * n_neg)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    precision, recall, thresholds = np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def find_nearest(array, value):

    array = np.asarray(array)
    length = len(array)
    abs_diff = np.abs(array - value)

    t_star = abs_diff.min()
    equal_arr = (abs_diff == t_star).astype('float32') + np.linspace(start=0, stop=0.1, num=length)

    idx = equal_arr.argmax()

    return array[idx], idx

def acc_at_t(preds, labels, t):

    pred_t = np.copy(preds)
    pred_t[pred_t > t] = 1
    pred_t[pred_t <= t] = 0

    acc = accuracy_score(labels, pred_t.astype('int32'))

    return acc

def closed_set_acc(preds, labels):

    preds = preds.argmax(axis=-1)
    acc = accuracy_score(labels, preds)

    print('Closed Set Accuracy: {:.3f}'.format(acc))

    return acc

def tar_at_far_and_reverse(fpr, tpr, thresholds):

    # TAR at FAR
    tar_at_far_all = {}
    for t in thresholds:
        tar_at_far_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(fpr, t)
        tar_at_far = tpr[idx]
        tar_at_far_all[t] = tar_at_far

        print(f'TAR @ FAR {t}: {tar_at_far}')

    # FAR at TAR
    far_at_tar_all = {}
    for t in thresholds:
        far_at_tar_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(tpr, t)
        far_at_tar = fpr[idx]
        far_at_tar_all[t] = far_at_tar

        print(f'FAR @ TAR {t}: {far_at_tar}')

def acc_at_95_tpr(open_set_preds, open_set_labels, thresholds, tpr):

    # Error rate at 95% TAR
    _, idx = find_nearest(tpr, 0.95)
    t = thresholds[idx]
    acc_at_95 = acc_at_t(open_set_preds, open_set_labels, t)
    print(f'Error Rate at TPR 95%: {1 - acc_at_95}')

    return acc_at_95

def compute_auroc(open_set_preds, open_set_labels):

    auroc = roc_auc_score(open_set_labels, open_set_preds)
    print(f'AUROC: {auroc}')

    return auroc

def compute_aupr(open_set_preds, open_set_labels, normalised_ap=False):

    if normalised_ap:
        aupr = normalised_average_precision(open_set_labels, open_set_preds)
    else:
        aupr = average_precision_score(open_set_labels, open_set_preds)
    print(f'AUPR: {aupr}')

    return aupr

def get_curve_online(known, novel, stypes = ['Bas']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95

def metric_ood(x1, x2, stypes = ['Bas'], verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = (-np.trapz(1.-fpr, tpr))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')
    
    return results

def compute_openauc(x1, x2, pred, labels):
    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2, correct = x1.tolist(), x2.tolist(), (pred == labels).tolist()
    m_x2 = max(x2) + 1e-5
    y_score = [value if hit else m_x2 for value, hit in zip(x1, correct)] + x2
    y_true = [0] * len(x1) + [1] * len(x2)
    open_auc = roc_auc_score(y_true, y_score)
    print('OpenAUC:', open_auc)
    return open_auc

class ModelTemplate(torch.nn.Module):

    def forward(self, imgs):
        """
        :param imgs:
        :return: Closed set and open set predictions on imgs
        """
        pass

class EnsembleModel(ModelTemplate):

    def __init__(self, model, use_softmax=False):
        super(ModelTemplate, self).__init__()
        self.model = model
        self.use_softmax = use_softmax

    def forward(self, imgs):
        _, closed_set_preds = self.model(imgs, True)
        if self.use_softmax:
            closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

        open_set_preds = - closed_set_preds.max(dim=-1)[0]
        open_set_preds = torch.nn.Sigmoid()(open_set_preds)

        return closed_set_preds, open_set_preds

class EvaluateOpenSet():

    def __init__(self, model, known_data_loader, unknown_data_loader, save_dir=None, device=None):

        self.model = model
        self.known_data_loader = known_data_loader
        self.unknown_data_loader = unknown_data_loader
        self.save_dir = save_dir

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.device = device

        # Init empty lists for saving labels and preds
        self.closed_set_preds = {0: [], 1: []}
        self.open_set_preds = {0: [], 1: []}

        self.closed_set_labels = {0: [], 1: []}
        self.open_set_labels = {0: [], 1: []}

        if save_dir and not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def predict(self, save=True):

        with torch.no_grad():
            for open_set_label, loader in enumerate((self.known_data_loader, self.unknown_data_loader)):

                if open_set_label:
                    print('Forward pass through Open Set test set...')
                else:
                    print('Forward pass through Closed Set test set...')

                for batch_idx, batch in enumerate(tqdm(loader)):

                    imgs, labels, idxs = [x.to(self.device) for x in batch]

                    # Model forward
                    output = self.model(imgs)
                    closed_set_preds, open_set_preds = [x.cpu().numpy().tolist() for x in output]

                    # Update preds and labels
                    self.closed_set_preds[open_set_label].extend(closed_set_preds)
                    self.open_set_preds[open_set_label].extend(open_set_preds)

                    self.closed_set_labels[open_set_label].extend(labels.cpu().numpy().tolist())
                    self.open_set_labels[open_set_label].extend([open_set_label] * len(labels))

        if save: # Save to disk
            save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']
            save_lists = [self.closed_set_preds, self.open_set_preds, self.closed_set_labels, self.open_set_labels]

            for name, x in zip(save_names, save_lists):

                path = os.path.join(self.save_dir, name)
                torch.save(x, path)
        else:
            return self.closed_set_preds, self.open_set_preds, self.closed_set_labels, self.open_set_labels

    @staticmethod
    def evaluate(self, load=True, preds=None, normalised_ap=False):

        if load:
            save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = \
                [torch.load(os.path.join(self.save_dir, name)) for name in save_names]

        else:

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = preds

        open_set_preds = np.array(open_set_preds[0] + open_set_preds[1])
        open_set_labels = np.array(open_set_labels[0] + open_set_labels[1])

        # ----------------------------
        # CLOSED SET EVALUATION
        # ----------------------------

        test_acc = closed_set_acc(np.array(closed_set_preds[0]), np.array(closed_set_labels[0]))

        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        fpr, tpr, thresh = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)
        acc_95 = acc_at_95_tpr(open_set_preds, open_set_labels, thresh, tpr)
        auroc = compute_auroc(open_set_preds, open_set_labels)
        aupr = compute_aupr(open_set_preds, open_set_labels, normalised_ap=normalised_ap)

        open_set_preds_known_cls = open_set_preds[~open_set_labels.astype('bool')]
        open_set_preds_unknown_cls = open_set_preds[open_set_labels.astype('bool')]
        closed_set_preds_pred_cls = np.array(closed_set_preds[0]).argmax(axis=-1)
        labels_known_cls = np.array(closed_set_labels[0])

        open_auc = compute_openauc(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls)

        ts = [_ * 0.01 for _ in range(1, 100)]
        closed_set_preds = np.array(closed_set_preds[0] + closed_set_preds[1])
        closed_set_labels = np.array(closed_set_labels[0] + closed_set_labels[1])
        ma = ma_f_score(closed_set_preds, closed_set_labels, open_set_preds, open_set_labels, ts)
        mi = mi_f_score(closed_set_preds, closed_set_labels, open_set_preds, open_set_labels, ts)

        results = {
            'Acc': test_acc,
            'Acc_95': acc_95,
            'AUROC': auroc,
            'AUPRC': aupr,
            'OpenAUC': open_auc,
            'ma': ma,
            'mi': mi,
        }

        return results
