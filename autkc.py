import os
import time
import argparse
import shutil
from copy import deepcopy
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.models as models

from XCurve.AUTKC.dataloaders import get_data_loader
from XCurve.AUTKC.utils.common_utils import AverageMeter, dict2obj, setup_seed
from XCurve.AUTKC.optimizer import adjust_learning_rate
from XCurve.AUTKC.losses.AUTKCLoss import StandardAUTKCLoss
from XCurve.AUTKC.losses.TopKLoss import BaseHingeLoss, HingeTopKLoss, CETopKLoss
from XCurve.AUTKC.metrics import evaluate

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar-100', type=str, required=True)
parser.add_argument('--loss', default='ce', type=str, required=True)
parser.add_argument('--ft', default=True, type=bool)
parser.add_argument('-k', default=5, type=int)
parser.add_argument('--surrogate', default='Exp', type=str)

def train(train_loader, model, criterion, optimizer, epoch, print_freq, k_list):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    topks = [AverageMeter('Acc@%d' % k, ':6.2f') for k in k_list]
    autkcs = [AverageMeter('AUTKC@%d' % k, ':6.2f') for k in k_list]

    model.train()
    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.squeeze().cuda(non_blocking =True)
        inputs = inputs.float().cuda(non_blocking =True)
        optimizer.zero_grad()

        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets, epoch) if hasattr(criterion, 'epoch_to_paced') else criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        precs, autkc= evaluate(outputs.data, targets, k_list)
        for _ in range(len(k_list)):
            topks[_].update(precs[_], inputs.size(0))
            autkcs[_].update(autkc[_], inputs.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            epoch_str = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(train_loader))
            time_str = 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
            loss_str = 'Loss {loss.val:.2f} ({loss.avg:.2f})'.format(loss=losses)
            autkc_str = '  '.join(['AUTKC@{} {autkcs.val:.2f} ({autkcs.avg:.2f})'.format(k_list[_], autkcs=autkcs[_]) for _ in range(len(k_list))])
            topks_str = '  '.join(['Prec@{} {topk.val:.2f} ({topk.avg:.2f})'.format(k_list[_], topk=topks[_]) for _ in range(len(k_list))])
            print(epoch_str, time_str, loss_str, autkc_str, topks_str, sep='  ')


def validate(val_loader, model, print_freq, k_list):
    batch_time = AverageMeter('Time', ':6.3f')
    topks =  [AverageMeter('Acc@%d' % k, ':6.2f') for k in k_list]
    autkcs = [AverageMeter('AUTKC@%d' % k, ':6.2f') for k in k_list]

    model.eval()
    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        targets = targets.squeeze().cuda(non_blocking =True)
        inputs = inputs.float().cuda(non_blocking =True)
        with torch.no_grad():
            outputs = model(inputs).squeeze()
            precs, autkc = evaluate(outputs.data, targets, k_list)
            for _ in range(len(k_list)):
                topks[_].update(precs[_], inputs.size(0))
                autkcs[_].update(autkc[_], inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                iteration_str = 'Test: [{0}/{1}]'.format(i, len(val_loader))
                time_str = 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
                autkc_str = '  '.join(['AUTKC@{} {autkcs.val:.2f} ({autkcs.avg:.2f})'.format(k_list[_], autkcs=autkcs[_]) for _ in range(len(k_list))])
                topks_str = '\t'.join(['Prec@{} {topk.val:.2f} ({topk.avg:.2f})'.format(k_list[_], topk=topks[_]) for _ in range(len(k_list))])
                print(iteration_str, time_str, autkc_str, topks_str, sep='\t')
    
    autkc_str = '\t'.join(['AUTKC@{} {autkc.avg:.2f}'.format(k_list[_], autkc=autkcs[_]) for _ in range(len(k_list))])
    topks_str = '\t'.join(['topk@{} {topk.avg:.2f}'.format(k_list[_], topk=topks[_]) for _ in range(len(k_list))])
    print('[val]', autkc_str, topks_str)
    return [float(autkc.avg) for autkc in autkcs], [float(topk.avg) for topk in topks]

######## Load args ########
args_parser = parser.parse_args()
args_dict = {
    'dataset_dir': 'example/data',
    'workers': 4,
    'print_freq': 10,
    'train_ratio': 0.9,
    'epoch_to_adjust_lr': 30,

    'dataset': args_parser.dataset,
    'loss': args_parser.loss,
    'ft': args_parser.ft,
}
args_dict['epochs'] = 90 if args_dict['dataset'] in ['cifar-10', 'cifar-100', 'place-365'] else 50
if args_dict['loss'] in ['l1', 'l2', 'l3', 'l4', 'l5', 'topkce']:
    args_dict['k'] = args_parser.k
elif args_dict['loss'] == 'autkc':
    args_dict['surrogate'] = args_parser.surrogate

default_args = {
    'batch_size': 128, 
    'lr': 0.01, 
    'weight_decay': 0.001, 
    'momentum': 0.9, 
    'rand': 0,
    'K': 5,
    'epoch_to_paced': 0
}
args_dict.update(default_args)

args = dict2obj(args_dict)
print(args)

######## Prepare data, model, loss, optimizer ########
setup_seed(args.rand)
train_loader, val_loader, _, num_class = get_data_loader(os.path.join(args.dataset_dir, args.dataset), args.batch_size, args.workers, args.train_ratio)
args.k_list = [_ for _ in range(1, min(num_class, 11))]

if args.dataset in ['place-365', ]:
    model = nn.Sequential(
        nn.Linear(2048, 512, bias=True), 
        nn.ReLU(),
        nn.Linear(512, 256, bias=True), 
        nn.ReLU(),
        nn.Linear(256, num_class, bias=True), 
    )
else:
    model = models.resnet18(pretrained=args.ft)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)  
model = model.cuda()

if args.loss == 'ce':
    criterion = nn.CrossEntropyLoss().cuda()
elif args.loss == 'hinge':
    criterion = BaseHingeLoss().cuda()
elif args.loss in ['l1', 'l2', 'l3', 'l4', 'l5']:
    criterion = HingeTopKLoss(args.k, args.loss).cuda()
elif args.loss == 'topkce':
    criterion = CETopKLoss(args.k).cuda()
elif args.loss == 'autkc':
    criterion = StandardAUTKCLoss(args.surrogate, args.K, args.epoch_to_paced).cuda()
else:
    raise ValueError
    
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

######## Construct the dir of checkpoint ########
if args.loss == 'autkc':
    loss_name = 'autkc-{}-{}'.format(args.surrogate, args.K)
else:
    loss_name = args.loss if 'k' not in args_dict.keys() else '{}-{}'.format(args.loss, args.k)
loss_name = 'ft-' + loss_name if args.ft else loss_name
save_dir = os.path.join(
    'checkpoints', 
    '{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, loss_name, args.epoch_to_paced, args.batch_size, args.lr, args.momentum, args.weight_decay, args.rand)
)
os.makedirs(save_dir) if not os.path.exists(save_dir) else None

######## Train the model ########
best_precs = [0.0, ] * len(args.k_list)
best_autkcs = [0.0, ] * len(args.k_list)
for epoch in range(args.epochs):

    adjust_learning_rate(optimizer, epoch, args.lr, args.epoch_to_adjust_lr)
    train(train_loader, model, criterion, optimizer, epoch, args.print_freq, args.k_list)
    autkc, precs = validate(val_loader, model, args.print_freq, args.k_list)

    state = deepcopy(args_dict)
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['epoch'] = epoch
    state['precs'] = precs
    state['autkc'] = autkc
    
    save_name = '{}.pth'.format(state['epoch'])
    save_path = os.path.join(save_dir, save_name)
    torch.save(state, save_path)

    for _ in range(len(precs)):
        if precs[_] > best_precs[_]:
            best_precs[_] = precs[_]
            save_name = 'best_prec{}.pth'.format(args.k_list[_])
            shutil.copyfile(save_path, os.path.join(save_dir, save_name))
        if autkc[_] > best_autkcs[_]:
            best_autkcs[_] = autkc[_]
            save_name = 'best_autkc{}.pth'.format(args.k_list[_])
            shutil.copyfile(save_path, os.path.join(save_dir, save_name))
    print('[best]', best_autkcs, best_precs)