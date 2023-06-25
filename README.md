<div align=center>
<img src="https://github.com/statusrank/XCurve/blob/master/img/Xcurve-logo.png">
</div>

***
# XCurve: Machine Learning with X-Curve Metrics

- [XCurve: Machine Learning with X-Curve Metrics](#xcurve-machine-learning-with-x-curve-metrics)
  - [Latest News](#latest-news)
  - [Introduction](#introduction)
    - [Wide Real-World Applications](#wide-real-world-applications)
  - [Supported Curves in XCurve](#supported-curves-in-xcurve)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Contact \& Contribution](#contact--contribution)
  - [Citation](#citation)


***<center><font color='#dd00dd'> Please visit the [website](https://XCurveOpt.org.cn) for more details on XCurve!</font></center>***

---

## Latest News
- <font color='red'> (New!)</font> <font color='blue'> 2022.6：</font> The XCurve-v1.0.0 has been released! Please Try now!

## Introduction
In recent years, Machine Learning (ML) has achieved significant advances in many domains, such as image recognition, machine translation, and biological information processing, promoting AI development. However, despite great success, it is well-known that the data often exhibits a **long-tailed/imbalanced property** in real-world applications, posing a critical challenge for the practical performances of deployed ML algorithms. Why? This is because the current studies are mainly established by **minimizing accuracy (or cross-entropy) criteria**, and then one needs to figure out a decision threshold to determine the category of samples on top of their prediction scores. In practice, such limited consideration of the decision threshold cannot adapt to the changes in data distributions and the growing business requirements, leading to unsatisfactory performance in real-world applications. 

To overcome this, **XCurve focuses on the design criteria of the objective function for ML tasks, which can be formulated as a series of X-metric (say AUROC, AUPRC, AUTKC) optimization problems considering the average performance of all decision thresholds during the training phase.**

Here is the high-level insights of XCurve:
<div align=center>
<img src="https://github.com/statusrank/XCurve/blob/master/img/xcurve-insight.png">
</div>

and this is the library of XCurve:
<div align=center>
<img src="https://github.com/statusrank/XCurve/blob/master/img/xcurve-framework.png">
</div>


To better understand how the XCurve achieves such a goal, let us take AUROC as an example in a high-level manner, as shown in the following figure:
<div align=center>
<img src="https://github.com/statusrank/XCurve/blob/master/img/AUROC-curve.png">
</div>
</center>

 
### Wide Real-World Applications
There is a wide range of applications for XCurve in the real world, especially the data following a long-tailed/imbalanced distribution. 
Several cases are listed below:
<div align=center>
<img src="https://github.com/statusrank/XCurve/blob/master/img/applications1.png">
</div>


## Supported Curves in XCurve
| X-Curve | Description |
| :----: | :----: |
| [XCurve.AUROC]() | an efficient optimization library for Area Under the ROC curve (AUROC). |
| [XCurve.AUPRC]() | an efficient optimization library for Area Under the Precision-Recall curve (AUPRC). |
| [XCurve.AUTKC]() | an efficient optimization library for Area Under the Top-K curve (AUPRC). |
| [XCurve.OpenAUC]() | an efficient optimization library for Area Under the Open ROC curve (OpenAUC). |
| ... | ... |

***<center><font color='#dd00dd'>More X-Curves are stepping up the development. Please stay tuned! </font></center>***

## Installation
<!--
You need the following packages to install XCurve:
```python
- Python >= 3.6+
- Pytorch >= 1.8+
- Numpy >= 1.21+
- scikit-learn >= 1.0+
```-->
You can get XCurve by
```sh
pip install XCurve
```

## Quickstart
Let us take the multi-class AUROC optimization as an example curve here. Detailed tutorial could be found in the website (https://XCurve.org.cn).

```python
'''
We refer the reader to see our paper <Learning with Multiclass AUC: Theory and Algorithms>
if they are interested in the technical details of this example. 
'''
import torch
from easydict import EasyDict as edict

# import loss of AUROC
from XCurve.AUROC.losses import SquareAUCLoss

# import optimier (or one can use any optimizer supported by PyTorch)
from XCurve.AUROC.optimizer import SGD

# create model or you can adopt any DNN models by Pytorch
from XCurve.AUROC.models import generate_net

# set params to create model
args = edict({
    "model_type": "resnet18", # (support resnet18,resnet20, densenet121 and mlp)
    "num_classes": 2,
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
from XCurve.AUROC.dataloaders import get_datasets
from XCurve.AUROC.dataloaders import get_data_loaders

# set dataset params, see our doc. for more details.
dataset_args = edict({
    "data_dir": "...",
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
        "1": 1
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

    x, target  = x.cuda(), target.cuda()
    # target.shape => [batch_size, ]
    # Note that we ask for the prediction of the model among [0,1] 
    # for any binary (i.e., sigmoid) or multi-class (i.e., softmax) AUROC optimization.

    pred = model(x) # [batch_size, num_classess] when num_classes > 2, o.w. output [batch_size, ] 

    loss = criterion(pred, target)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Contact & Contribution
If you find any issues or plan to contribute back bug-fixes, please contact us by [Shilong Bao](https://statusrank.github.io/) (Email: baoshilong@iie.ac.cn) or [Zhiyong Yang](https://joshuaas.github.io/) (Email: yangzhiyong21@ucas.ac.cn)

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
