import pytest
import random
import sys
import os
from easydict import EasyDict as edict
sys.path.append("./")
from XCurve.AUTKC.dataloaders import get_data_loader, MyDataset

@pytest.mark.skip()
def load_cfg(dataset_name):
    if dataset_name == 'cifar-10':
        dataset_args = edict({
            "dataset_root": 'D:/dataset',
            "dataset": 'cifar-10',
            "input_size": [32, 32],
            "norm_params": {
                "mean": [123.675, 116.280, 103.530],
                "std": [58.395, 57.120, 57.375]
                },
            "num_classes": 10,
        })
    elif dataset_name == 'cifar-100':
        dataset_args = edict({
            "dataset_root": 'D:/dataset',
            "dataset": 'cifar-100',
            "input_size": [32, 32],
            "norm_params": {
                "mean": [123.675, 116.280, 103.530],
                "std": [58.395, 57.120, 57.375]
                },
            "num_classes": 100
        })
    elif dataset_name == 'tiny-imagenet-200':
        dataset_args = edict({
            "dataset_root": 'D:/dataset',
            "dataset": 'tiny-imagenet-200',
            "input_size": [64, 64],
            "norm_params": {
                "mean": [123.675, 116.280, 103.530],
                "std": [58.395, 57.120, 57.375]
                },
            "num_classes": 100
        })
    return dataset_args

def test_single_dataset():
    for dataset_name in ['tiny-imagenet-200', 'cifar-10', 'cifar-100']:
        # print(dataset_name)
        args = load_cfg(dataset_name)     
        dataset_dir = os.path.join(args['dataset_root'], args['dataset'])
        train_dataset, val_dataset = MyDataset(dataset_dir, split='train'), MyDataset(dataset_dir, split='val')
        
        for rep in range(10):
            idx = random.randint(0, len(train_dataset) - 1)
            img, lbl = train_dataset.__getitem__(idx)
        assert img.shape == (3, args['input_size'][0],args['input_size'][1])

def test_dataloader():
    for dataset_name in ['tiny-imagenet-200', 'cifar-10', 'cifar-100']:
        args = load_cfg(dataset_name)     
        dataset_dir = os.path.join(args['dataset_root'], args['dataset'])
        train_loader, val_loader, _, num_class = get_data_loader(dataset_dir, batch_size=32, workers=4, train_ratio=0.9)
        # print(test_loader)
        batch_size = 32
        for subset in [val_loader]:
            for i, (img, lbl) in enumerate(subset):
                assert img.shape[0] == lbl.shape[0] == batch_size
                assert img.shape[1:] == (3, args['input_size'][0],args['input_size'][1])

                if (i+1)%10 == 0:
                    break
        
        for i, (img, lbl) in enumerate(train_loader):
            assert img.shape[0] == lbl.shape[0] == train_loader.batch_size
            assert img.shape[1:] == (3, args['input_size'][0],args['input_size'][1])

            if (i+1)%10 == 0:
                break
                       

if __name__ == '__main__':
    test_single_dataset()
    # test_dataloader()
