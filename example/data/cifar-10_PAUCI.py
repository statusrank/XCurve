import sys
import os
import copy
sys.path.append(os.pardir)
import numpy as np
import torch
import torch.nn as nn
from XCurve.AUROC.losses.PAUCI import PAUCI
from XCurve.AUROC.optimizer.ASGDA import ASGDA
from XCurve.AUROC.dataloaders import get_datasets
from XCurve.AUROC.dataloaders import get_data_loaders
from XCurve.AUROC.models import generate_net
from XCurve.AUROC.metrics.partial_AUROC import p2AUC
from easydict import EasyDict as edict

method='SPAUCI'

# hyper parameters
hyper_param = {
	'mini-batch':    256,
	'alpha':         1.0,
	'beta':          0.3,
	'weight_decay':  1e-5,
	'init_lr': 		 0.001
}

if hyper_param['alpha'] == 1:
	metrics = 'OPAUC'
else:
	metrics = 'TPAUC'

sigmoid = nn.Sigmoid() # Limit the output score between 0 and 1

# load data and dataloader
args_dataset = edict({
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

args_model = edict({
	"model_type": "resnet18", # (support resnet18,resnet20, densenet121 and mlp)
	"num_classes": 2,
	"pretrained": None
})

train_set, val_set, test_set = get_datasets(args_dataset)
train_loader, val_loader, test_loader = get_data_loaders(
	train_set,
	val_set,
	test_set,
	hyper_param['mini-batch'],
	hyper_param['mini-batch']
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model (train model from the scratch, using model: resnet18)
model = generate_net(args_model).to(device)

# load hyper parameters from json file
hparams = {
	"k": 1,
	"c1": 3,
	"c2": 3,
	"lam": 0.02,
	"nu": 0.02,
	"m": 500,
	"device": device,
}

# define loss and optimizer
criterion = PAUCI(hyper_param['alpha'], hyper_param['beta'], device)
optimizer = ASGDA([
		{'params': model.parameters(), 'name':'net'},
		{'params': [criterion.a, criterion.b], 'clip':(0, 1), 'name':'ab'},
		{'params': criterion.s_n, 'clip':(0, 5), 'name':'sn'},
		{'params': criterion.s_p, 'clip':(-4, 1), 'name':'sp'},
		{'params': criterion.lam_b, 'clip':(0, 1e9), 'name':'lamn'},
		{'params': criterion.lam_a, 'clip':(0, 1e9), 'name':'lamp'},
		{'params': criterion.g, 'clip':(-1, 1), 'name':'g'}], 
		weight_decay=hyper_param['weight_decay'], hparams=hparams)
		
best_model = model.state_dict()
best_perf = 0
all_counter = 0

# train 50 epoch
for epoch in range(50):
	all_pauc = 0
	counter = 0
	model.train()
	for i, (img, lbl) in enumerate(train_loader):
		optimizer.zero_grad()
		img = img.to(device)
		lbl = lbl.to(device).float()
		out = sigmoid(model(img))
		loss = criterion(out, lbl)
		loss.backward()
		optimizer.step(pre=True, t=all_counter)

		optimizer.zero_grad()
		out = sigmoid(model(img))
		loss = criterion(out, lbl)
		loss.backward()
		optimizer.step(pre=False, t=all_counter)
		label = lbl.cpu().detach().numpy().reshape((-1, ))
		pred = out.cpu().detach().numpy().reshape((-1, ))
		all_pauc += p2AUC(label, pred, hyper_param['alpha'], hyper_param['beta'])
		counter += 1
		all_counter += 1
	# record instances' prediction and label of val set
	model.eval()
	val_pred = np.array([])
	val_label = np.array([])
	for i, (img, lbl) in enumerate(val_loader):
		img = img.to(device)
		lbl = lbl.to(device).float()
		out = sigmoid(model(img))
		label = lbl.cpu().detach().numpy().reshape((-1, ))
		pred = out.cpu().detach().numpy().reshape((-1, ))
		val_pred = np.hstack([val_pred, pred])
		val_label = np.hstack([val_label, label])
	pauc = p2AUC(val_label, val_pred, hyper_param['alpha'], hyper_param['beta'])
	print('epoch:{} val pauc:{}'.format(epoch, pauc))
	if pauc > best_perf:
		best_perf = pauc
		best_model = copy.deepcopy(model.state_dict())


# record instances' prediction and label of test set
model.load_state_dict(best_model)
model.eval()
test_pred = np.array([])
test_label = np.array([])
for i, (img, lbl) in enumerate(test_loader):
	img = img.to(device)
	lbl = lbl.to(device)
	out = sigmoid(model(img))
	label = lbl.cpu().detach().numpy().reshape((-1, ))
	pred = out.cpu().detach().numpy().reshape((-1, ))
	test_pred = np.hstack([test_pred, pred])
	test_label = np.hstack([test_label, label])
pauc = p2AUC(test_label, test_pred, hyper_param['alpha'], hyper_param['beta'])
print('test pauc:{}'.format(pauc))