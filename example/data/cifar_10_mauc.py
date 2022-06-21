import torch
from easydict import EasyDict as edict

# import loss of AUROC
from XCurve.AUROC.losses import SquareAUCLoss

# import optimier (or one can use any optimizer supported by PyTorch)
from torch.optim import SGD

# create model or you can adopt any DNN models by Pytorch
from XCurve.AUROC.models import generate_net

# set params to create model
args = edict({
    "model_type": "resnet18", # (support resnet18,resnet20, densenet121 and mlp)
    "num_classes": 10,
    "pretrained": None
})
model = generate_net(args).cuda()

num_classes = 10
# create optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# create loss criterion
criterion = SquareAUCLoss(
    num_classes=num_classes, # number of classes
    gamma=1.0, # safe margin
    transform="ovo" # the manner of computing the multi-classes AUROC Metric ('ovo' or 'ova').
)

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
    "num_classes": num_classes
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

    x, target  = x.cuda(), target.cuda()
    # target.shape => [batch_size, ]
    # Note that we ask for the prediction of the model among [0,1] 
    # for any binary (i.e., sigmoid) or multi-class (i.e., softmax) AUROC optimization.

    pred = torch.sigmoid(model(x)) # [batch_size, num_classess] when num_classes > 2, o.w. output [batch_size, ] 

    loss = criterion(pred, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()