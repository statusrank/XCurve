import pytest
import torch
import random
import sys
from easydict import EasyDict as edict
sys.path.append("./")

from XCurve.AUROC.dataloaders import get_datasets, get_data_loaders
from XCurve.AUROC.models import generate_net

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

def test_mAUC_losses_decrese():
    '''
        check if the loss can drop normally
    '''
    from XCurve.AUROC.losses import SquareAUCLoss, ExpAUCLoss, HingeAUCLoss
    for dataset_name in ['cifar-10-m', 'cifar-100-m']:
        args = load_cfg(dataset_name)
        batch_size = 8
        train_set, val_set, test_set = get_datasets(args)
        trainloader, valloader, testloader = get_data_loaders(train_set, 
                                                              val_set, 
                                                              test_set, 
                                                              train_batch_size=batch_size, 
                                                              test_batch_size =batch_size)
        
        model = generate_net(args).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for loss in [SquareAUCLoss, ExpAUCLoss, HingeAUCLoss]:
            criterion = loss(
                num_classes=args['num_classes'], 
                gamma=1.0, 
                transform="ovo" 
            )
            
            train_loss = []
            for i, (images, targets) in enumerate(trainloader):
                images = images.cuda()
                targets = targets.cuda()
                feats = torch.sigmoid(model(images))
                loss = criterion(feats, targets)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1)%100 == 0:
                    break

            assert sum(train_loss[:min(len(train_loss)//2, 10)]) > sum(train_loss[-min(len(train_loss)//2, 10)//2:]), train_loss


def test_pAUC_losses_decrese():
    from XCurve.AUROC.losses.PartialAUROC import BaseAUCLoss, RelaxedPAUCLoss, InsRelaxedPAUCLoss, UnbiasedPAUCLoss
    for dataset_name in ['cifar-10-b', 'cifar-100-b']:
        args = load_cfg(dataset_name)
        batch_size = 8
        train_set, val_set, test_set = get_datasets(args)
        trainloader, valloader, testloader = get_data_loaders(train_set, 
                                                              val_set, 
                                                              test_set, 
                                                              train_batch_size=batch_size, 
                                                              test_batch_size =batch_size)
        
        model = generate_net(args).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for loss in [ BaseAUCLoss, 
                     RelaxedPAUCLoss, 
                     # InsRelaxedPAUCLoss, 
                     # UnbiasedPAUCLoss
                     ]:
            if loss is BaseAUCLoss:
                criterion = BaseAUCLoss(
                                gamma=1, # safe margin
                                E_k=0, # warm-up epoch.
                                weight_scheme="Poly", # weight scheme
                                num_classes=2, # number of classes
                                reduction="mean", # loss aggregated manne
                                first_state_loss=torch.nn.BCELoss(), # warm-up loss
                                eps=1e-6 # avoid zero gradient
                            )
            elif loss is RelaxedPAUCLoss:
                criterion = RelaxedPAUCLoss(
                                gamma=1, # safe margin
                                E_k=0, # warm-up epoch.
                                weight_scheme="Poly", # weight scheme
                                num_classes=2, # number of classes
                                reduction="mean", # loss aggregated manne
                                AUC_type='OP', # (OPAUC or TPAUC optimization)
                                first_state_loss=torch.nn.BCELoss() # warm-up loss
                            )
            elif loss is InsRelaxedPAUCLoss:
                pass
                # test in test_optimizer.pys
                # criterion = InsRelaxedPAUCLoss(
                #                 gamma=1, # safe margin
                #                 E_k=0, # warm-up epoch.
                #                 weight_scheme="Poly", # weight scheme
                #                 num_classes=2, # number of classes
                #                 eps=1e-6, # avoid zero gradient
                #                 AUC_type='OP', # (OPAUC or TPAUC optimization)
                #                 first_state_loss=torch.nn.BCELoss(), # warm-up loss
                #                 reg_a=0.1,
                #                 reg_b=0.2, # reg_a and reg_b: weight of the strong convex constraint
                #             )
            elif loss is UnbiasedPAUCLoss:
                pass
                # test in test_optimizer.pys
                # criterion = UnbiasedPAUCLoss(
                #     alpha=1.0, # optimization variable, 1 for OPAUC and 2 for TPAUC
                #     beta=0.3, # optimization variable
                #     device=torch.device('cuda:0')
                # )

            train_loss = []
            for i, (images, targets) in enumerate(trainloader):
                images = images.cuda()
                targets = targets.cuda()
                feats = torch.sigmoid(model(images))
                loss = criterion(feats, targets)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1)%100 == 0:
                    break

            assert sum(train_loss[:min(len(train_loss)//2, 10)]) > sum(train_loss[-min(len(train_loss)//2, 10)//2:]), train_loss
'''
def test_AdAUC_losses_decrese():
    from XCurve.AUROC.losses import AdvAUROCLoss, PGDAdversary, RegAdvAUROCLoss
    from XCurve.AUROC.optimizer import AdvAUCOptimizer, RegAdvAUCOptimizer
    for dataset_name in ['cifar-10-b', 'cifar-100-b']:
        args = load_cfg(dataset_name)
        batch_size = 8
        train_set, val_set, test_set = get_datasets(args)
        trainloader, valloader, testloader = get_data_loaders(train_set, 
                                                              val_set, 
                                                              test_set, 
                                                              train_batch_size=batch_size, 
                                                              test_batch_size =batch_size)
        
        model = generate_net(args).cuda()
        for loss in [# AdvAUROCLoss, 
                     RegAdvAUROCLoss, 
                     ]:
            criterion = loss(imratio=0.1)
            if loss == AdvAUROCLoss:
                optimizer = AdvAUCOptimizer(model.parameters(), 
                                            criterion.a, criterion.b, criterion.alpha,
                                            lr=0.01, momentum=0.9,
                                            weight_decay=1e-5)
            elif loss == RegAdvAUROCLoss:
                optimizer = RegAdvAUCOptimizer(model.parameters(), 
                                            criterion.a, criterion.b, criterion.alpha, criterion.lambda1, criterion.lambda2,
                                            lr=0.01,)

            model.train()
            train_loss = []
            lower_limit, upper_limit = 0.0, 1.0
            for i, (images, targets) in enumerate(trainloader):
                images = images.cuda()
                targets = targets.cuda()
                delta = PGDAdversary(model, images, targets, criterion, epsilon=8.0/255, alpha=2.0/255, 
                         attack_iters=10, restarts=1, norm='linf')
                adv_input = torch.clamp(images + delta, min=lower_limit, max=upper_limit)
                adv_input.requires_grad_(requires_grad=False)

                robust_output = model(adv_input).view_as(targets)
                pred = torch.sigmoid(robust_output)
                robust_loss = criterion(pred, targets)
                train_loss.append(robust_loss.item())
                optimizer.zero_grad()
                robust_loss.backward()
                optimizer.step()
                if (i+1)%100 == 0:
                    break

            assert sum(train_loss[:min(len(train_loss)//2, 10)]) > sum(train_loss[-min(len(train_loss)//2, 10)//2:]), train_loss
'''


if __name__ == '__main__':
    # test_mAUC_losses_decrese()
    test_pAUC_losses_decrese()
    # test_AdAUC_losses_decrese()