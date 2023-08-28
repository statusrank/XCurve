import pytest
import sys
sys.path.append("./")
import torch

from XCurve.Metrics import AUROC


@pytest.mark.skip()
def gen_data(n_items, num_classes):
    if num_classes == 2:
        feats = torch.rand((n_items, 1)).numpy()
    elif num_classes > 2:
        feats = torch.nn.functional.softmax(torch.rand((n_items, num_classes)), dim = -1).numpy()
    targets = torch.tensor(range(0, num_classes))
    while targets.shape[0] < n_items:
        targets = torch.cat((targets, targets.clone()))[:n_items]

    return feats, targets.numpy()

def test_metrics():
    for num_classes in [2, 10, 100, 1000]:
        for multi_type in ['ovo', 'ova']:
            feats, targets = gen_data(2**10, num_classes)
            auroc = AUROC(y_true = targets, y_pred = feats, multi_type = multi_type)
            assert 0 <= auroc <= 1


if __name__ == '__main__':
    test_metrics()
