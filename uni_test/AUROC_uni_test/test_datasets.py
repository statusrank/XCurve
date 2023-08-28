import pytest
import random
import sys
from easydict import EasyDict as edict
sys.path.append("./")

from XCurve.AUROC.dataloaders import get_datasets, get_data_loaders

@pytest.mark.skip()
def load_cfg(dataset_name):
    if dataset_name == 'cifar-10':
        dataset_args = edict({
            "data_dir": "cifar-10-long-tail/",
            "input_size": [32, 32],
            "norm_params": {
                "mean": [123.675, 116.280, 103.530],
                "std": [58.395, 57.120, 57.375]
                },
            "use_lmdb": True,
            "resampler_type": "None",
            "sampler": { # only used for binary classification
                "rpos": 1,
                "rneg": 10
                },
            "npy_style": True,
            "aug": True, 
            "num_classes": 10
        })
    elif dataset_name== 'cifar-100':
        dataset_args = edict({
            "data_dir": "cifar-100-long-tail/",
            "input_size": [32, 32],
            "norm_params": {
                "mean": [123.675, 116.280, 103.530],
                "std": [58.395, 57.120, 57.375]
                },
            "use_lmdb": True,
            "resampler_type": "None",
            "sampler": { # only used for binary classification
                "rpos": 1,
                "rneg": 10
                },
            "npy_style": True,
            "aug": True, 
            "num_classes": 100
        })

    return dataset_args

def test_single_dataset():
    for dataset_name in ['cifar-10', 'cifar-100']:
        args = load_cfg(dataset_name)
        train_set, val_set, test_set = get_datasets(args)
        
        for rep in range(10):
            idx = random.randint(0, len(train_set) - 1)
            img, lbl = train_set.__getitem__(idx)
        assert img.shape == (3, args['input_size'][0],args['input_size'][1])
        assert lbl.unsqueeze(0).shape == (1,)

def test_dataloader():

    for dataset_name in ['cifar-10', 'cifar-100']:
        args = load_cfg(dataset_name)
        batch_size = 32
        train_set, val_set, test_set = get_datasets(args)
        trainloader, valloader, testloader = get_data_loaders(train_set, 
                                                              val_set, 
                                                              test_set, 
                                                              train_batch_size=batch_size, 
                                                              test_batch_size =batch_size)
        
        for subset in [testloader, valloader]:
            for i, (img, lbl) in enumerate(subset):
                assert img.shape[0] == lbl.shape[0] == batch_size
                assert img.shape[1:] == (3, args['input_size'][0],args['input_size'][1])

                if (i+1)%10 == 0:
                    break
        
        for i, (img, lbl) in enumerate(trainloader):
            assert img.shape[0] == lbl.shape[0] == trainloader.batch_size
            assert img.shape[1:] == (3, args['input_size'][0],args['input_size'][1])

            if (i+1)%10 == 0:
                break
                       

if __name__ == '__main__':
    test_single_dataset()
    test_dataloader()
