import pytest
import sys
sys.path.append("./")
import torch
import numpy as np

from XCurve.Metrics import AUTKC, TopkAcc, MicroF


@pytest.mark.skip()
def gen_data(emb_dim, num_classes):
    n_samples, C = 2**14, 5
    y_true = np.random.randint(low=0, high=C, size=(n_samples, ))
    y_pred = np.random.rand(n_samples, C)
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    
    return y_pred, y_true

def test_metrics():
    k_list = (1, 3)
    for num_classes in [10, 100, 1000]:
        feats, targets = gen_data(128, num_classes)
        autkc = AUTKC(y_pred=feats, y_true=targets, k_list=k_list)
        topk_acc=TopkAcc(y_pred=feats, y_true=targets, k_list=k_list)


if __name__ == '__main__':
    test_metrics()