import pytest
import sys
sys.path.append("./")

from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from XCurve.AUPRC import RetrievalDataset, RetrievalModel


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

def test_model():
    for dataset_name in ['inat', 'sop', 'vehid']:
        args = load_cfg(dataset_name)
        dataset = RetrievalDataset(**args, split='val')
        dataloader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            num_workers=4
        )

        for frz in [True, False]:
            for frz_bn in [True, False]:
                model = RetrievalModel(512, frz, frz_bn).cuda()
                for i, (img, _) in enumerate(dataloader):
                    img = img.cuda()
                    out = model(img)
                    assert out.shape == (img.shape[0], model.output_channels)                
                    if (i+1)%4 == 0:
                        break


if __name__ == '__main__':
    test_model()
