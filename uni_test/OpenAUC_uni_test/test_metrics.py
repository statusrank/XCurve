import pytest
import sys
sys.path.append("./")
import torch
import numpy as np
from XCurve.Metrics import OpenAUC, Acc_At_TPR, ClosedSetAcc, MicroF, MacroF


@pytest.mark.skip()
def gen_data():
    n_samples, C = 10, 5
    open_pred = np.random.rand(n_samples)
    open_labels = np.random.randint(low=0, high=2, size=(n_samples, ))

    close_pred = np.random.rand(n_samples, C)
    close_labels = np.random.randint(low=0, high=C, size=(n_samples, ))
    
    n_close_samples, C, n_open_samples = 10, 5, 8
    open_set_pred_known = np.random.rand(n_close_samples)
    open_set_pred_unknown = np.random.rand(n_open_samples)
    close_set_pred_class = np.random.randint(low=0, high=C, size=(n_close_samples, ))
    close_set_labels = np.random.randint(low=0, high=C, size=(n_close_samples, ))
    # print(open_set_pred_known, open_set_pred_unknown, close_set_pred_class, close_set_labels)
    return open_pred, open_labels, close_pred, close_labels, open_set_pred_known, open_set_pred_unknown, close_set_pred_class, close_set_labels

def test_metrics():
    t_list = (0,2, 0.4, 0.6, 0.8)
    
    open_pred, open_labels, close_pred, close_labels, open_set_pred_known, open_set_pred_unknown, close_set_pred_class, close_set_labels = gen_data()
    acc_at_95_tpr = Acc_At_TPR(open_pred, open_labels, 0.95)
    openauc = OpenAUC(open_set_pred_known, open_set_pred_unknown, close_set_pred_class, close_set_labels)
    acc = ClosedSetAcc(close_pred, close_labels)
    MicroF_score = MicroF(close_pred, close_labels, open_pred, open_labels, t_list)
    MacroF_score = MacroF(close_pred, close_labels, open_pred, open_labels, t_list)
    print(openauc)
    assert 0 <= acc_at_95_tpr <= 1
    assert 0 <= openauc <= 1
    assert 0 <= acc <= 1


if __name__ == '__main__':
    test_metrics()