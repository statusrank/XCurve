import pytest
import sys
sys.path.append("./")
import torch

from XCurve.AUPRC import AUPRC, RecallAtK


@pytest.mark.skip()
def gen_data(emb_dim, num_classes):
    feats = torch.randn((2**14, emb_dim)).numpy()
    targets = torch.randint(0, num_classes, (2**14, 1)).numpy()
    return feats, targets

def test_metrics():

    eps = 1e-4
    for num_classes in [10, 100, 1000]:
        feats, targets = gen_data(128, num_classes)
        auprc = AUPRC(feats, targets)
        rec_at_1_ = RecallAtK(feats, targets, 1)
        rec_at_1, rec_at_10, rec_at_100 = RecallAtK(feats, targets, [1, 10, 100])
        assert 0 < auprc < 1
        assert abs(rec_at_1_ - rec_at_1) < eps


if __name__ == '__main__':
    test_metrics()
