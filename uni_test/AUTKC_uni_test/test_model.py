import pytest
import torch
import random
import sys
import os
from easydict import EasyDict as edict
sys.path.append("./")

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from XCurve.AUTKC.dataloaders import get_data_loader

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
    elif dataset_name== 'cifar-100':
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
    return dataset_args

def test_model_decrese():
    '''
        check if the loss can drop normally
    '''
    from XCurve.AUTKC.losses.AUTKCLoss import StandardAUTKCLoss
    for dataset_name in ['cifar-10', 'cifar-100']:
        args = load_cfg(dataset_name)     
        dataset_dir = os.path.join(args['dataset_root'], args['dataset'])
        train_loader, val_loader, _, num_class = get_data_loader(dataset_dir, batch_size=32, workers=4, train_ratio=0.9)
        
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)  
        model = model.cuda()

        for epoch in range(1):
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                targets = targets.squeeze().cuda(non_blocking =True)
                inputs = inputs.float().cuda(non_blocking =True)
                outputs = model(inputs).squeeze()
            
            assert sum(train_loss[:min(len(train_loss)//2, 10)]) >= sum(train_loss[-min(len(train_loss)//2, 10)//2:]), train_loss



if __name__ == '__main__':
    test_model_decrese()