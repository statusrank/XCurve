import torch
from easydict import EasyDict as edict

# import loss of AUROC
from XCurve.AUROC.losses import get_losses

# import optimier (or one can use any optimizer supported by PyTorch)
from XCurve.AUROC.optimizer import SGD4MinMaxPAUC

# create model or you can adopt any DNN models by Pytorch
from XCurve.AUROC.models import generate_net

# set params to create model
args = edict({
    "model_type": "resnet18", # (support resnet18,resnet20, densenet121 and mlp)
    "num_classes": 2,
    "pretrained": None
})

args_training = edict({
    "train_batch_size": 32,
    "test_batch_size": 32,
    "num_workers": 4,
    "loss_type": "MinMaxPAUC",
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
    "seed": 7
})

model = generate_net(args).cuda()

num_classes = 2
# create optimizer
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
criterion = get_losses(args_training)

# create Dataset (train_set, val_set, test_set) and dataloader (trainloader)
# You can construct your own dataset/dataloader 
# but must ensure that there at least one sample for every class in each mini-batch 
# to calculate the AUROC loss. Or, you can do this:
from XCurve.AUROC.dataloaders import get_datasets
from XCurve.AUROC.dataloaders import get_data_loaders

# set dataset params, see our doc. for more details.
dataset_args = edict({
    "data_dir": "data/cifar-10-long-tail/",
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
    }
})

train_set, val_set, test_set = get_datasets(dataset_args)
trainloader, valloader, testloader = get_data_loaders(
    train_set,
    val_set,
    test_set,
    train_batch_size=32,
    test_batch_size =64
)
# Note that, in the get_datasets(), we conduct stratified sampling for train_set  
# using the StratifiedSampler at from XCurve.AUROC.dataloaders import StratifiedSampler

# forward of model
for x, target in trainloader:
    x, target  = x.cuda(), target.cuda().reshape((-1, 1))
    # target.shape => [batch_size, ]
    # Note that we ask for the prediction of the model among [0,1] 
    # for any binary (i.e., sigmoid) or multi-class (i.e., softmax) AUROC optimization.

    pred = torch.sigmoid(model(x)) # [batch_size, num_classess] when num_classes > 2, o.w. output [batch_size, ] 

    loss = criterion(pred, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
