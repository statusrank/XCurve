import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

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

def Acc_At_TPR(open_set_preds, open_set_labels, r=0.95):
    _, tpr, thresholds = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)

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

class EnsembleModel:
    def __init__(self, model, use_softmax=False):
        self.model = model
        self.use_softmax = use_softmax

    def forward(self, imgs):
        """
        :param imgs:
        :return: Closed set and open set predictions on imgs
        """
        _, closed_set_preds = self.model(imgs, True)
        if self.use_softmax:
            closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

        open_set_preds = - closed_set_preds.max(dim=-1)[0]
        open_set_preds = torch.nn.Sigmoid()(open_set_preds)

        return closed_set_preds, open_set_preds

class OpenSetEvaluator():

    def __init__(self, model: EnsembleModel, known_data_loader, unknown_data_loader, save_dir=None, device=None):

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
    def evaluate(self, load=True, preds=None):

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

        close_set_acc = ClosedSetAcc(np.array(closed_set_preds[0]), np.array(closed_set_labels[0]))

        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        acc_95 = Acc_At_TPR(open_set_preds, open_set_labels)
        auroc = AUROC(open_set_preds, open_set_labels)

        open_set_preds_known_cls = open_set_preds[~open_set_labels.astype('bool')]
        open_set_preds_unknown_cls = open_set_preds[open_set_labels.astype('bool')]
        closed_set_preds_pred_cls = np.array(closed_set_preds[0]).argmax(axis=-1)
        labels_known_cls = np.array(closed_set_labels[0])

        open_auc = OpenAUC(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls)

        t_list = [_ * 0.01 for _ in range(1, 100)]
        closed_set_preds = np.array(closed_set_preds[0] + closed_set_preds[1])
        closed_set_labels = np.array(closed_set_labels[0] + closed_set_labels[1])
        macroF = MacroF(closed_set_preds, closed_set_labels, open_set_preds, open_set_labels, t_list)
        microF = MicroF(closed_set_preds, closed_set_labels, open_set_preds, open_set_labels, t_list)

        results = {
            'Acc': close_set_acc,
            'Acc_95': acc_95,
            'AUROC': auroc,
            'OpenAUC': open_auc,
            'macroF': macroF,
            'microF': microF,
        }

        return results