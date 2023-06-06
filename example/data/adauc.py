#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
from tqdm import tqdm
import random
import numpy as np
import time
from tqdm import tqdm
import gc
import nni
from torch.optim import lr_scheduler

# from utils import *
from XCurve.AUROC.dataloaders.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from mnist_dataset import IMBALANCEMNIST
from XCurve.AUROC.models.wideresnetwideresnet import *
from mnist_widesesnet import WideResNet_mnist
from XCurve.AUROC.optimizer import ours_opt
from XCurve.AUROC.losses.ADAUC import *
from XCurve.AUROC.dataloaders.sampler import *



def normalize(X):
	global  mu, std
	return (X - mu)/std


def auc_binary(y_true, y_pred):
	if len(y_true.shape) > 1:
		y_true = y_true.squeeze()
	
	if len(y_pred.shape) > 1:
		y_pred = y_pred.squeeze()
	# print('y_true', y_true)
	# print('y_pred', y_pred)
	label = y_true == 1
	nP = label.sum()
	nN = label.shape[0] - nP
	sindex = np.argsort(y_pred)
	lSorted = label[sindex]
	auc = (np.where(lSorted != True) - np.arange(nN)).sum()
	auc /= (nN * nP)
	
	return 1 - auc

# 设置pgd映射的上下界
lower_limit = 0.
upper_limit = 1.

def clamp(X, lower_limit, upper_limit):
	return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
			   norm, early_stop=False, early_stop_pgd_max=1,
			   device=None, criterion=None,
			   p_hat=0, args=None):
	"""gen adv example...
	input: model, X, y
	output: delta
	"""
	max_loss = torch.zeros(y.shape[0], device=device)
	max_delta = torch.zeros_like(X, device=device)
	early_stop = False
	if args.loss_type == 'auc':
		attack_iters = 1
	for _ in range(restarts):  # for _ in range(1)
		# early stop pgd counter for each x
		# early_stop_pgd_count = early_stop_pgd_max * torch.ones(y.shape[0], dtype=torch.int32).to(device)
		# early_stop_pgd_count = [1,1,...,1]
		
		# initialize perturbation
		delta = torch.zeros_like(X, device=device)

		delta.uniform_(-epsilon, epsilon)
		# print('1', delta)
		# pgd
		
		delta = clamp(delta, lower_limit - X, upper_limit - X)
		delta.requires_grad = True
		# print('2', delta)
		
		iter_count = torch.zeros(y.shape[0])  # [0,0,...,0]
		
		for _ in range(attack_iters):
			if args.dataset == 'cifar10' or args.dataset == 'cifar100':
				output = model(normalize(X + delta)).view_as(y)
			elif args.dataset == 'mnist':
				output = model(X + delta).view_as(y)
			
			
			index = slice(None, None, None)
			
			loss = criterion(output, y)
			loss.backward()
			grad = delta.grad.detach()
			
			d = delta[index, :, :, :]
			g = grad[index, :, :, :]
			x = X[index, :, :, :]

			d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
			
			d = clamp(d, lower_limit - x, upper_limit - x)
			delta.data[index, :, :, :] = d
			delta.grad.zero_()
		
		if args.dataset == 'cifar10' or args.dataset == 'cifar100':
			all_loss = criterion(model(normalize(X + delta)).view_as(y), y)
		elif args.dataset == 'mnist':
			all_loss = criterion(model(X + delta).view_as(y), y)
			
		max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
		max_loss = torch.max(max_loss, all_loss)
	
	return max_delta, iter_count


def attack_pgd_imbalance(model, X, y, epsilon, alpha, attack_iters, restarts,
			   norm, early_stop=False, early_stop_pgd_max=1,
			   device=None, criterion=None,
			   p_hat=0, args=None):
	"""gen adv example...
	input: model, X, y
	output: delta
	"""
	max_loss = torch.zeros(y.shape[0]).to(device)
	max_delta = torch.zeros_like(X).to(device)
	early_stop = False
	if args.loss_type == 'auc':
		attack_iters = 1
	y_label = y.clone().detach()

	for _ in range(restarts):  # for _ in range(1)
		# early stop pgd counter for each x
		early_stop_pgd_count = early_stop_pgd_max * torch.ones(y.shape[0], dtype=torch.int32).to(device)
		# early_stop_pgd_count = [1,1,...,1]

		# initialize perturbation
		delta = torch.zeros_like(X).to(device)

		delta.uniform_(-epsilon, epsilon)
		# print('1', delta)
		# pgd

		delta = clamp(delta, lower_limit - X, upper_limit - X)
		delta.requires_grad = True
		# print('2', delta)

		iter_count = torch.zeros(y.shape[0])  # [0,0,...,0]

		for _ in range(attack_iters):
			if args.dataset == 'cifar10' or args.dataset == 'cifar100':
				output = model(normalize(X + delta)).view_as(y)
			elif args.dataset == 'mnist':
				output = model(X + delta).view_as(y)
			# if use early stop pgd
			if early_stop:
				# calculate mask for early stop pgd
				pred = output.clone().detach()
				pred[pred > 0.5] = 1
				pred[pred <= 0.5] = 0
				if_success_fool = (pred != y).to(dtype=torch.int32)
				early_stop_pgd_count = early_stop_pgd_count - if_success_fool
				index = torch.where(early_stop_pgd_count > 0)[0]
				iter_count[index] = iter_count[index] + 1
			else:
				index = slice(None, None, None)
			if not isinstance(index, slice) and len(index) == 0:
				break

			loss = criterion(output, y)
			# print(loss)
			loss.backward()
			grad = delta.grad.detach()

			d =  torch.zeros_like(delta, device=device)
			g = grad[index, :, :, :]
			x = X[index, :, :, :]

			d[y_label==1] = torch.clamp(delta[y_label==1] + alpha * torch.sign(g[y_label==1]), min=-epsilon*(1+p_hat), max=epsilon*(1+p_hat))
			d[y_label==0] = torch.clamp(delta[y_label==0] + alpha * torch.sign(g[y_label==0]), min=-epsilon, max=epsilon)

			d = clamp(d, lower_limit - x, upper_limit - x)
			delta.data[index, :, :, :] = d
			delta.grad.zero_()
		
		if args.dataset == 'cifar10' or args.dataset == 'cifar100':
			all_loss = criterion(model(normalize(X + delta)).view_as(y), y)
		elif args.dataset == 'mnist':
			all_loss = criterion(model(X + delta).view_as(y), y)
			
		max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
		max_loss = torch.max(max_loss, all_loss)

	return max_delta, iter_count



def get_filename(args):
	filename = args.train_type + '_' + \
			   args.test_type + '_' + args.loss_type +\
			   str(args.lr_max) + '_' + time.strftime('%Y_%m_%d_%H_%M_%S')
	print('Store Filename: ', filename)
	
	return filename

def nni_update_parameters(parser):
	tuner_params = nni.get_next_parameter()
	parser.set_defaults(lr_max=tuner_params['lr_max'])
	parser.set_defaults(momentum=tuner_params['momentum'])
	parser.set_defaults(weight_decay=tuner_params['weight_decay'])
	
	# parser.set_defaults(lr_a=tuner_params['lr_a'])
	# parser.set_defaults(lr_b=tuner_params['lr_b'])
	# parser.set_defaults(lr_alpha=tuner_params['lr_alpha'])
	parser.set_defaults(loss_type=tuner_params['loss_type'])
	parser.set_defaults(train_type=tuner_params['train_type'])
	parser.set_defaults(test_type=tuner_params['test_type'])
	
	parser.set_defaults(lr_scheduler_type=tuner_params['lr_scheduler_type'])
	parser.set_defaults(lr_decay_rate=tuner_params['lr_decay_rate'])
	parser.set_defaults(T_max=tuner_params['T_max'])
	parser.set_defaults(factor=tuner_params['factor'])

	parser.set_defaults(imbalance=tuner_params['imbalance'])
	args = parser.parse_args()
	return args, parser


def get_parameters():
	parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
	# adv
	parser.add_argument('--seed', default=1234, type=int)
	parser.add_argument('--momentum', default=0.9, type=float)
	parser.add_argument('--weight_decay', default=5e-4, type=float)
	
	parser.add_argument('--epsilon', default=8.0, type=float)
	parser.add_argument('--test_epsilon', default=8.0, type=float)
	parser.add_argument('--pgd-alpha', default=2, type=float)
	parser.add_argument('--test-pgd-alpha', default=2, type=float)
	
	parser.add_argument('--epochs', default=201, type=int)
	parser.add_argument('--attack_iters', default=10, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--resume', '-r', default=None, type=int, help='resume from checkpoint')
	
	parser.add_argument('--device_id', default=0, type=int)
	parser.add_argument('--loss_type', default='auc', type=str, help='auc, ce')
	parser.add_argument('--train_type', default='pgd', type=str, help='adv, none')
	parser.add_argument('--test_type', default='pgd', type=str, help='adv, none')
	parser.add_argument('--data_dir', default='../', type=str)
	parser.add_argument('--dataset', default='cifar100', type=str)
	
	parser.add_argument('--lr_max', default=0.1, type=float)
	parser.add_argument('--lr-one-drop', default=0.01, type=float)
	
	parser.add_argument('--earlystopPGD', action='store_true')  # whether use early stop in PGD
	parser.add_argument('--earlystopPGDepoch1', default=60, type=int)
	parser.add_argument('--earlystopPGDepoch2', default=100, type=int)
	
	parser.add_argument('--attack', default='none', type=str)
	
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--norm', default='l_inf', type=str)
	parser.add_argument('--restarts', default=1, type=int)
	# auc
	parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
	parser.add_argument('--imbalance', default=0, type=int, help='imbalance type')
	parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
	parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
	parser.add_argument('--workers', default=4, type=int)
	parser.add_argument('--fname', default='', type=str)
	
	parser.add_argument('--lr_a', default=1e-5, type=float)
	parser.add_argument('--lr_b', default=1e-5, type=float)
	parser.add_argument('--lr_alpha', default=1e-5, type=float)
	parser.add_argument('--lr_decay_rate', default=0.99, type=float)
	parser.add_argument('--lr_decay_epochs', default=1, type=float)
	parser.add_argument('--T_max', default=20, type=float)
	
	parser.add_argument('--rpos', default=1, type=float)
	parser.add_argument('--rneg', default=9, type=float)
	
	args, parser = nni_update_parameters(parser)
	return args
	

def init_setting(args):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	
	# get save filename
	args.fname = get_filename(args)
	
	# Random
	print('==> Init Random Seed..')
	seed = args.seed
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	phat = args.rpos / (args.rpos + args.rneg)
	return device, phat


def get_dataset(args):
	# Dataset
	if args.dataset == 'cifar10':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 将这一步放在生成对抗样本的时候了
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		train_dataset = IMBALANCECIFAR10(root=args.data_dir,
										 train=True,
										 download=True,
										 transform=transform_train)
		test_dataset = IMBALANCECIFAR10(root=args.data_dir,
										train=False,
										transform=transform_test)
	elif args.dataset == 'mnist':
		transform_train = transforms.Compose([
			# transforms.RandomCrop(32, padding=4),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 将这一步放在生成对抗样本的时候了
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		train_dataset = IMBALANCEMNIST(root=args.data_dir,
										 train=True,
										 download=True,
										 transform=transform_train)
		test_dataset = IMBALANCEMNIST(root=args.data_dir,
										train=False,
										transform=transform_test)
	elif args.dataset == 'cifar100':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 将这一步放在生成对抗样本的时候了
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		train_dataset = IMBALANCECIFAR100(root=args.data_dir,
										 train=True,
										 download=True,
										 transform=transform_train)
		test_dataset = IMBALANCECIFAR100(root=args.data_dir,
										train=False,
										transform=transform_test)
	
	
	
	train_sampler = StratifiedSampler(class_vector=train_dataset.targets,
									  batch_size=args.batch_size,
									  rpos=args.rpos,
									  rneg=args.rneg)
	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=train_sampler.real_batch_size,
		shuffle=(train_sampler is None),
		num_workers=args.workers,
		pin_memory=True,
		sampler=train_sampler,
		drop_last=True)
	
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=100,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True)
	return train_loader, test_loader, train_dataset.phat


if __name__ == '__main__':
	args = get_parameters()
	
	# Init
	device, p_hat = init_setting(args)
	
	mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1).to(device)
	std = torch.tensor((0.2023, 0.1994, 0.2010)).view(3, 1, 1).to(device)
	
	# Set adv perturbations
	print('==> Set adv perturbations...')
	epsilon = (args.epsilon / 255.)
	test_epsilon = (args.test_epsilon / 255.)
	pgd_alpha = (args.pgd_alpha / 255.)
	test_pgd_alpha = (args.test_pgd_alpha / 255.)
	
	# dataset
	print('==> Data Loading..')
	train_loader, test_loader, p_hat = get_dataset(args)

	# set models
	print('==> Set models...')
	if args.dataset == 'mnist':
		model = WideResNet_mnist(28, 1)
	elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
		model = WideResNet(28, 1)
	
	model.to(device)
	model.train()
	
	# set training parameters
	params = model.parameters()
	
	if args.loss_type=='auc':
		criterion = AUCLoss(imratio=p_hat, device=device)
		# optimizer
		opt = ours_opt(params, criterion.a, criterion.b, criterion.alpha,
					   lr=args.lr_max, momentum=args.momentum,
					   weight_decay=args.weight_decay)
		opt.zero_grad()
	elif args.loss_type=='ce':
		criterion = torch.nn.BCELoss()
		
		opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
		opt.zero_grad()
		
	if args.lr_scheduler_type == 'exp':
		scheduler = lr_scheduler.ExponentialLR(opt,
										   args.lr_decay_rate)
	elif args.lr_scheduler_type == 'cos':
		scheduler = lr_scheduler.CosineAnnealingLR(opt,
											T_max=args.T_max)
	elif args.lr_scheduler_type == 'reduce':
		scheduler = lr_scheduler.ReduceLROnPlateau(opt,
												   mode='max',
												   factor=args.factor,
												   patience=1)

	epochs = args.epochs
	
	
	def eps_alpha_schedule(t):  # Schedule number 0
		return epsilon, pgd_alpha, args.restarts
	
	# Records per epoch for savetxt
	train_loss_record, train_auc_record, train_robust_loss_record, train_robust_auc_record = [], [], [], []
	
	test_loss_record, test_auc_record, test_robust_loss_record, test_robust_auc_record = [], [], [], []

	best_test_robust_auc = 0


	print('==> Start Training')
	for epoch in range(0, epochs):
		model.train()
		start_time = time.time()
		
		train_loss, train_auc, train_robust_loss, train_robust_auc = 0, 0, 0, 0
		train_n = 0
		
		target, pred, pred_robust = [], [], []
		
		iterator = tqdm(train_loader, ncols=0, leave=False)
		for batch_id, (X, y) in enumerate(iterator):
			epoch_now = epoch + (batch_id+1) / len(train_loader)

			X = X.to(device)
			y = y.float().to(device)
			if args.train_type == 'pgd':
				epsilon_sche, pgd_alpha_sche, restarts_sche = eps_alpha_schedule(epoch_now)
				if args.imbalance == 1:
					delta, _ = attack_pgd_imbalance(model, X, y, epsilon_sche, pgd_alpha_sche, args.attack_iters,
													restarts_sche, args.norm,
													early_stop=args.earlystopPGD,
													device=device,
													criterion=criterion,
													p_hat=p_hat,
													args=args)
				else:
					delta, _ = attack_pgd(model, X, y, epsilon_sche, pgd_alpha_sche, args.attack_iters,
										  restarts_sche, args.norm,
										  early_stop=args.earlystopPGD,
										  device=device,
										  criterion=criterion,
										  p_hat=p_hat,
										  args=args)

				delta = delta.detach()
			elif args.train_type == 'none':
				delta = torch.zeros_like(X)
			
			if args.dataset == 'cifar10' or args.dataset == 'cifar100':
				adv_input = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
			elif args.dataset == 'mnist':
				adv_input = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
			
			adv_input.requires_grad_()
			robust_output = model(adv_input).view_as(y)
			# print(robust_output[y==1].mean(), robust_output[y==0].mean())
			
			robust_loss = criterion(robust_output, y)
			
			# print(param_a.item(), param_b.item(), param_alpha.item())
			opt.zero_grad()
			robust_loss.backward()
			opt.step()

			train_robust_loss += robust_loss.item() * y.size(0)
			train_n += y.size(0)
			target.append(y.cpu().detach().numpy())
			pred_robust.append(robust_output.cpu().detach().numpy())
			
			auc = auc_binary(y_true=y.cpu().detach().numpy(), y_pred=robust_output.cpu().detach().numpy())
			iterator.set_description('train_robust_loss: ' + str(robust_loss.item()) +
									 ' auc: ' + str(auc))
		target = np.concatenate(target)
		pred_robust = np.concatenate(pred_robust)
		auc = auc_binary(y_true=target, y_pred=pred_robust)
		print(epoch, 'train_auc:', auc)
		
		if args.lr_scheduler_type == 'reduce':
			scheduler.step(auc)
		else:
			if (epoch + 1) % args.lr_decay_epochs == 0:
				scheduler.step()
		
		train_time = time.time()
		
		model.eval()
		test_loss, test_auc, test_robust_loss, test_robust_auc = 0, 0, 0, 0
		test_n = 0
		target, pred_robust, pred_clean = [], [], []

		iterator = tqdm(test_loader, ncols=0, leave=False)
		for batch_id, (X, y) in enumerate(iterator):
			X = X.to(device)
			y = y.float().to(device)
			if epoch % 10 == 0:
				delta, _ = attack_pgd(model, X, y,
									  test_epsilon, test_pgd_alpha,
									  args.attack_iters, args.restarts,
									  args.norm, early_stop=False,
									  device=device,
									  criterion=criterion,
									  p_hat=p_hat,
									  args=args)
				delta = delta.detach()
				if args.dataset == 'cifar10' or args.dataset == 'cifar100':
					adv_input = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
				elif args.dataset == 'mnist':
					adv_input = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
				
				# print("test mean ", adv_input.mean(), "std ", adv_input.std())
				# adv_input = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
				adv_input.requires_grad_()
				
				robust_output = model(adv_input).view_as(y)
				# print(robust_output)
				# print(y)
				robust_loss = criterion(robust_output, y)
				test_robust_loss += robust_loss.item() * y.size(0)

				pred_robust.append(robust_output.cpu().detach().numpy())
				
			target.append(y.cpu().detach().numpy())

			test_n += y.size(0)
			if args.dataset == 'cifar10' or args.dataset == 'cifar100':
				clean_output = model(normalize(X)).view_as(y)
			elif args.dataset == 'mnist':
				clean_output = model(X).view_as(y)
			pred_clean.append(clean_output.cpu().detach().numpy())
			clean_loss = criterion(clean_output, y)
			test_loss += clean_loss.item() * y.size(0)
			iterator.set_description('test_robust_loss: ' + str(clean_loss.item()))
		
			
		target = np.concatenate(target)
		pred_clean = np.concatenate(pred_clean)
		if len(pred_robust) == 0:
			test_robust_auc = -1
		else:
			pred_robust = np.concatenate(pred_robust)
			test_robust_auc = auc_binary(target, pred_robust)
		
		test_clean_auc = auc_binary(target, pred_clean)
		
		nni.report_intermediate_result({"default": test_clean_auc,
										"train_auc": auc,
										"test_robust_auc": test_robust_auc})
		test_time = time.time()
		
		print(epoch, ' test_robust_auc: ', test_robust_auc)
		
		
		if args.loss_type == 'auc':
			# save checkpoint
			print('==> save checkpoint..')
			
			if not os.path.exists('./checkpoint'):
				os.mkdir('./checkpoint')
			if not os.path.exists('./checkpoint/' + args.fname):
				os.mkdir('./checkpoint/' + args.fname)
				
			# if epoch > 99 or (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
			# 	torch.save(model.state_dict(), os.path.join('./checkpoint', args.fname, 'model_'+str(epoch)+'.pth'))
			# 	torch.save(opt.state_dict(), os.path.join('./checkpoint', args.fname, 'opt_'+str(epoch)+'.pth'))
			# 	torch.save({'param_a': criterion.a,
			# 				'param_b': criterion.b,
			# 				'param_alpha': criterion.alpha,}, os.path.join('./checkpoint', args.fname, 'param_minmax_'+str(epoch)+'.pth'))
			
			# save best
			if test_clean_auc > best_test_robust_auc:
				torch.save({
					'state_dict': model.state_dict(),
					'param_a': criterion.a,
					'param_b': criterion.b,
					'param_alpha': criterion.alpha,
					'test_robust_auc': test_robust_auc,
					'test_robust_loss': test_robust_loss,
				}, os.path.join('./checkpoint', args.fname, 'model_best.pth'))
				best_test_robust_auc = test_clean_auc
			if epoch % 5 ==0 :
				torch.save({
					'state_dict': model.state_dict(),
					'param_a': criterion.a,
					'param_b': criterion.b,
					'param_alpha': criterion.alpha,
					'test_robust_auc': test_robust_auc,
					'test_robust_loss': test_robust_loss,
				}, os.path.join('./checkpoint', args.fname, 'model_'+ str(epoch) +'.pth'))
		else:
			print('==> save checkpoint..')
			if not os.path.exists('./checkpoint'):
				os.mkdir('./checkpoint')
			if not os.path.exists('./checkpoint/' + args.fname):
				os.mkdir('./checkpoint/' + args.fname)
			if test_clean_auc > best_test_robust_auc:
				torch.save({
					'state_dict': model.state_dict(),
					'test_robust_auc': test_robust_auc,
					'test_robust_loss': test_robust_loss,
				}, os.path.join('./checkpoint', args.fname, 'model_best.pth'))
				best_test_robust_auc = test_clean_auc
	
	nni.report_final_result(best_test_robust_auc)
			
			
			
		
			
	