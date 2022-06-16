<div align=center>
<img src="https://github.com/statusrank/XCurveOpt/blob/master/img/Xcurve-logo.png">
</div>
</center>

***
# XCurveOpt: Machine Learning with X-Curve Metrics

- [XCurveOpt: Machine Learning with X-Curve Metrics](#xcurveopt-machine-learning-with-x-curve-metrics)
  - [Latest News](#latest-news)
  - [Introduction](#introduction)
  - [Supported Curves in XCurveOpt](#supported-curves-in-xcurveopt)
  - [Installation](#installation)
  - [Qucikstart](#qucikstart)
  - [Contact & Contribution](#contact--contribution)
  - [Citation](#citation)
  - [License](#license)


***<center><font color='#dd00dd'> Please visit the [website](https://xcurveopt.org.cn) for more details on XCurveOpt!</font></center>***

---

## Latest News
- <font color='red'> (New!)</font> <font color='blue'> 2022.6：</font> The XCurveOpt-v1.0.0 is released! Please Try now!

## Introduction
XCurveOpt is an end-to-end machine learning PyTorch library for X-Curve metrics optimizations.

## Supported Curves in XCurveOpt
| X-Curve | Description |
| :----: | :----: |
| [XCurveOpt.AUROC]() | an efficient optimization library for Area Under the ROC curve (AUROC), such as <font color='blue'>multi-class AUROC</font> and <font color='blue'>partial AUROC</font> optimization. |
| ... | ... |

***<center><font color='#dd00dd'>More X-Curves are stepping up the development. Please stay tuned! </font></center>***

## Installation

You need the following packages to install XCurveOpt:
```python
- Python >= 3.6+
- Pytorch >= 1.8+
- Numpy >= 1.21+
- scikit-learn >= 1.0+
```
Then you can get XCurveOpt 
```sh
pip install XCurveOpt
```

## Qucikstart
Let us take the multi-class AUROC optimization as an example curve here. Detailed tutorial could be found in the website (https://xcurveopt.org.cn).

```python
'''
We refer the reader to see our paper <Learning with Multiclass AUC: Theory and Algorithms>
if they are interested in the technical details of this example. 
'''
import torch
from easydict import EasyDict as edict

# import loss of AUROC
from XCurveOpt.AUROC.losses import SquareAUCLoss

# import optimier (or one can use any optimizer supported by PyTorch)
from XCurveOpt.AUROC.optimizer import SGD

# create model or you can adopt any DNN models by Pytorch

from XCurveOpt.AUROC.models import generate_net

# set params to create model
args = edict({
    "model_type": "resnet18", # (support resnet18,resnet20, densenet121 and mlp)
    "num_classes": 1,
    "pretrained": None
})
model = generate_net(args).cuda()

num_classes = 2
# create optimizer
optimizer = SGD([params of your model], lr=...)

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
from XCurveOpt.AUROC.dataloaders import get_datasets
from XCurveOpt.AUROC.dataloaders import get_data_loaders

dataset_args = {
    'some needed args to get datasets, see the doc.'
}
loader_args = {
    'mini-batch':64
}
train_set, val_set, test_set = get_datasets(dataset_args)
trainloader, valloader, testloader = get_data_loaders(
    train_set,
    val_set,
    test_set,
    loader_args['mini-batch'], # train_batch_size
    loader_args['mini-batch'] # test_batch_size
)
# Note that, in the get_datasets(), we conduct stratified sampling for train_set  
# using the StratifiedSampler at from XCurveOpt.AUROC.dataloaders import StratifiedSampler

# forword of model
for x, target in trainloader:

    x, target  = x.cuda(), target.cuda()
    # Note that we ask for the prediction of the model among [0,1] 
    # for ant binary (i.e., sigmoid) or multi-class (i.e., softmax) AUROC optimization.

    pred = model(x) 
    loss = criterion(pred, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
## Contact & Contribution
If you find any issues or plan to contribute back bug-fixes, please contact us by [Shilong Bao](https://scholar.google.com.hk/citations?user=5ZCgkQkAAAAJ&hl=zh-CN) (Email: baoshilong@iie.ac.cn) or [Zhiyong Yang](https://joshuaas.github.io/) (Email: yangzhiyong21@ucas.ac.cn)

***<center><font color='#dd00dd'> The authors appreciate all contributions!</font></center>***
## Citation
Please cite our paper if you use this library in your own work:
```
@inproceedings{DBLP:conf/icml/YQBYXQ, 
author    = {Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang},
  title     = {When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC},
  booktitle = {ICML},
  pages     = {11820--11829},
  year      = {2021}
```
## License

