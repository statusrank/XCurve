import pytest
import random
import sys
from torch.utils.data import DataLoader
sys.path.append("./")

from XCurve.AUPRC import RetrievalDataset


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

def test_single_dataset():
    for dataset_name in ['inat', 'sop', 'vehid']:
        args = load_cfg(dataset_name)
        dataset = RetrievalDataset(**args)

        for rep in range(10):
            idx = random.randint(0, len(dataset) - 1)
            img, lbl = dataset.__getitem__(idx)
        assert img.shape == (3, args.input_size, args.input_size)
        assert lbl.shape == (1,)

def test_dataloader():

    for dataset_name in ['inat', 'sop', 'vehid']:
        args = load_cfg(dataset_name)

        for subset in ['train', 'val', 'test']:
            dataset = RetrievalDataset(**args, split=subset)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batchsize,
                num_workers=4
            )

            for i, (img, lbl) in enumerate(dataloader):
                assert img.shape[0] == lbl.shape[0] == 56
                assert img.shape[1:] == (3, args.input_size, args.input_size)
                assert len(lbl.shape) == 2
                assert lbl.shape[1] == 1
                if (i+1)%10 == 0:
                    break

            assert len(dataloader.dataset.get_cnt_per_id()) > 0
            dataloader.dataset.reset()


if __name__ == '__main__':
    test_single_dataset()
    test_dataloader()
