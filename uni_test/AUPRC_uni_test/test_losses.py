import pytest
import torch
import random
import sys
from torch.utils.data import DataLoader
sys.path.append("./")

from XCurve.AUPRC import (RetrievalDataset,\
     RetrievalModel, ListStableAUPRC, DefaultLossCfg)


@pytest.mark.skip()
def load_cfg(dataset):
    if dataset == 'inat':
        from XCurve.AUPRC import DefaultInatDatasetCfg as DefaultDatasetCfg
    elif dataset == 'sop':
        from XCurve.AUPRC import DefaultSOPDatasetCfg as DefaultDatasetCfg
    elif dataset == 'vehid':
        from XCurve.AUPRC import DefaultVehIDDatasetCfg as DefaultDatasetCfg
    else:
        assert NotImplementedError

    return DefaultDatasetCfg

def test_losses_decrese():
    '''
        check if the loss can drop normally
    '''
    for dataset_name in ['inat', 'sop', 'vehid']:
        args = load_cfg(dataset_name)
        dataset = RetrievalDataset(**args, split='val')
        dataloader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            num_workers=4
        )

        model = RetrievalModel(512).cuda()
        criterion = ListStableAUPRC(**DefaultLossCfg)
        criterion.update_cnt_per_id(dataset.get_cnt_per_id())

        optimizer = torch.optim.SGD(model.param_groups(), lr=0.001)
        train_loss = []
        for i, (images, targets) in enumerate(dataloader):
            images = images.cuda()
            targets = targets.cuda()
            feats = model(images)
            loss = criterion(feats, targets)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1)%100 == 0:
                break

        assert sum(train_loss[:10]) > sum(train_loss[-10:]), train_loss


if __name__ == '__main__':
    test_losses_decrese()
