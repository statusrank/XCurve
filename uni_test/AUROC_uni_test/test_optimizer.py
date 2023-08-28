import pytest
import torch
import random
import sys
from easydict import EasyDict as edict
sys.path.append("./")

from XCurve.AUROC.dataloaders import get_datasets, get_data_loaders
from XCurve.AUROC.models import generate_net
from XCurve.AUROC.losses import get_losses

@pytest.mark.skip()
def load_cfg(dataset_name):
    if dataset_name == 'cifar-10-m':
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
            "num_classes": 10,
            "model_type": "resnet18",
            "pretrained": None
        })
    elif dataset_name== 'cifar-100-m':
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
            "num_classes": 100,
            "model_type": "resnet18",
            "pretrained": None
        })
    elif dataset_name== 'cifar-10-b':
        dataset_args = edict({
            "data_dir": "cifar-10-long-tail/", # relative path of dataset
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
            "class2id": { # positive (minority) class idx
                "1": 1, "0":0, "2":0, "3":0, "4":0, "5":0,
                "6":0, "7":0, "8":0, "9":0
            },
            "num_classes": 2,
            "model_type": "resnet18",
            "pretrained": None
        })
    elif dataset_name== 'cifar-100-b':
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
            "class2id": { 
                "0": 1, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0, "7":0, "8":0, "9":0 , 
                "10": 0, "11":0, "12":0, "13":0, "14":0, "15":0, "16":0, "17":0, "18":0, "19":0 , 
                "20": 0, "21":0, "22":0, "23":0, "24":0, "25":0, "26":0, "27":0, "28":0, "29":0 ,
                "30": 0, "31":0, "32":0, "33":0, "34":0, "35":0, "36":0, "37":0, "38":0, "39":0 ,
                "40": 0, "41":0, "42":0, "43":0, "44":0, "45":0, "46":0, "47":0, "48":0, "49":0 , 
                "50": 0, "51":1, "52":0, "53":1, "54":0, "55":0, "56":0, "57":1, "58":0, "59":0 ,
                "60": 0, "61":0, "62":0, "63":0, "64":0, "65":0, "66":0, "67":0, "68":0, "69":0 ,
                "70": 0, "71":0, "72":0, "73":0, "74":0, "75":0, "76":0, "77":0, "78":0, "79":0 , 
                "80": 0, "81":0, "82":0, "83":1, "84":0, "85":0, "86":0, "87":0, "88":0, "89":0 ,
                "90": 0, "91":0, "92":0, "93":0, "94":0, "95":0, "96":0, "97":0, "98":0, "99":0 ,
            },
            "num_classes": 2,
            "model_type": "resnet18",
            "pretrained": None
        })
    return dataset_args



def test_minimax_optimizer_decrese():
    from XCurve.AUROC.optimizer import SGD4MinMaxPAUC
    args_training = edict({
        "train_batch_size": 32,
        "test_batch_size": 32,
        "num_workers": 4,
        "loss_type": "InsRelaxedPAUCLoss",
        "loss_params": {
            "num_classes": 2,
            "gamma": 1.0,
            "E_k": 3,
            "weight_scheme": "Poly",
            "reduction": "mean",
            "AUC_type": "OP",
            "first_state_loss": torch.nn.BCELoss(),
            "eps": 1e-6,
            "reg_a":0.1,
            "reg_b":0.2
        },
        "lr": 0.001,
        "weight_decay": 1e-5,
        "momentum": 0.9,
        "nesterov": True,
        "lr_decay_rate": 0.99,
        "lr_decay_epochs": 1,
        "epoch_num": 50,
        "metric_params": {
            "alpha": 0.4,
            "beta": 0.1
        },
        "save_path": "./save/",
        "seed": 7,
        "model_type": "resnet18",
        "num_classes": 2,
        "pretrained": None
    })

    for dataset_name in ['cifar-10-b', 'cifar-100-b']:
        args = load_cfg(dataset_name)
        batch_size = 32
        train_set, val_set, test_set = get_datasets(args) 
        trainloader, valloader, testloader = get_data_loaders(
            train_set,
            val_set,
            test_set,
            train_batch_size=batch_size,
            test_batch_size =batch_size
        )

        model = generate_net(args).cuda()
        a, b = torch.tensor(0.5, requires_grad=True), torch.tensor(0.5, requires_grad=True)
        optimizer = SGD4MinMaxPAUC(
            params=model.parameters(),
            a=a,
            b=b,
            lr=args_training ['lr'],
            weight_decay=args_training['weight_decay'],
            clip_value=5,
            epoch_to_opt=1
        )

        # create loss criterion
        train_loss = []
        criterion = get_losses(args_training)
        for i, (images, targets) in enumerate(trainloader):
            images, targets  = images.cuda(), targets.cuda().reshape((-1, 1))
            feats = torch.sigmoid(model(images))
            loss = criterion(feats, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if (i+1)%100 == 0:
                break
        assert sum(train_loss[:min(len(train_loss)//2, 10)]) > sum(train_loss[-min(len(train_loss)//2, 10)//2:]), train_loss

def test_ASGDA_optimizer_decrese():
    from XCurve.AUROC.optimizer import ASGDA
    from XCurve.AUROC.losses.PartialAUROC import UnbiasedPAUCLoss
    hyper_param = {
        'mini-batch':    1024,
        'alpha':         1.0,
        'beta':          0.3,
        'weight_decay':  1e-5,
        'init_lr': 		 0.001
    }

    args = edict({
        "model_type": "resnet18", # (support resnet18,resnet20, densenet121 and mlp)
        "num_classes": 2,
        "pretrained": None
    })

    model = generate_net(args).cuda()
    for dataset in ['cifar-10-b', 'cifar-100-b']:
        args = load_cfg(dataset)
        batch_size = 32
        train_set, val_set, test_set = get_datasets(args) 
        trainloader, valloader, testloader = get_data_loaders(
            train_set,
            val_set,
            test_set,
            train_batch_size=batch_size,
            test_batch_size =batch_size
        )
        criterion = UnbiasedPAUCLoss(hyper_param['alpha'], hyper_param['beta'], 'cuda')
        if dataset == 'cifar-10-b':
            hparams = {
                "k": 1,
                "c1": 3,
                "c2": 3,
                "lam": 0.02,
                "nu": 0.02,
                "m": 500,
                "device" : 'cuda'
            }
        elif dataset == 'cifar-100-b':
            hparams = {
                "k": 1,
                "c1": 3,
                "c2": 3,
                "lam": 0.035,
                "nu": 0.035,
                "m": 1000,
                "device" : 'cuda'
            }
        
        optimizer = ASGDA([
                    {'params': model.parameters(), 'name':'net'},
                    {'params': [criterion.a, criterion.b], 'clip':(0, 1), 'name':'ab'},
                    {'params': criterion.s_n, 'clip':(0, 5), 'name':'sn'},
                    {'params': criterion.s_p, 'clip':(-4, 1), 'name':'sp'},
                    {'params': criterion.lam_b, 'clip':(0, 1e9), 'name':'lamn'},
                    {'params': criterion.lam_a, 'clip':(0, 1e9), 'name':'lamp'},
                    {'params': criterion.g, 'clip':(-1, 1), 'name':'g'}],
                    weight_decay=hyper_param['weight_decay'], hparams=hparams)

        train_loss = []
        for i, (images, targets) in enumerate(trainloader):
            images, targets  = images.cuda(), targets.cuda().reshape((-1, 1))
            feats = torch.sigmoid(model(images))
            loss = criterion(feats, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if (i+1)%100 == 0:
                break
        assert sum(train_loss[:min(len(train_loss)//2, 10)]) > sum(train_loss[-min(len(train_loss)//2, 10)//2:]), train_loss

if __name__ == '__main__':
    # test_minimax_optimizer_decrese()
    test_ASGDA_optimizer_decrese()