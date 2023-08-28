import pytest
import torch
import random
import sys
from torch.utils.data import DataLoader
sys.path.append("./")

from XCurve.OpenAUC.dataloaders.open_set_datasets import get_class_splits, get_datasets
import argparse

from XCurve.OpenAUC.utils.model_utils import get_model
from XCurve.OpenAUC.optimizers import get_optimizer, get_scheduler
from XCurve.OpenAUC.models.wrapper_classes import Classifier32Wrapper
from XCurve.OpenAUC.losses.OpenAUCLoss import StandardOpenAUCLoss
from XCurve.OpenAUC.losses.Softmax import Softmax
from tqdm import tqdm
from XCurve.OpenAUC.utils.common_utils import AverageMeter


@pytest.mark.skip()
def load_cfg(dataset_name):
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
    parser.add_argument('--model', type=str, default='classifier32')
    parser.add_argument('--image_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--transform', type=str, default='rand-augment')
    parser.add_argument('--num_workers', type=int, default=0) # zero for windows
    parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing.")
    parser.add_argument('--temp', type=float, default=1.0, help="temp for label_smoothing")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2, help="parameter for openauc loss")
    parser.add_argument('--lamda', type=float, default=0.05, help="parameter for openauc loss")
    parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
    parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
    args, _ = parser.parse_known_args() # for VS code
    args.dataset = dataset_name
    return args

def test_losses_decrese():
    '''
        check if the loss can drop normally
    '''
    args = load_cfg('mnist')
    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx)
    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            open_set_classes=args.open_set_classes, image_size=args.image_size, seed=args.seed)
    dataloader = DataLoader(
        datasets,
        batch_size=args.batch_size,
        num_workers=4
    )

    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size, shuffle=shuffle, sampler=None, num_workers=args.num_workers)

    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']
    
    net = get_model(args, wrapper_class=Classifier32Wrapper)
    criterion = StandardOpenAUCLoss(loss_close=Softmax(**{'temp': args.temp, 'label_smoothing': args.label_smoothing}), alpha=args.alpha, lambd=args.lamda)

    params_list = [{'params': net.parameters()}, {'params': criterion.parameters()}]
    optimizer = get_optimizer(args=args, params_list=params_list, **{'dataset': args.dataset})
    # scheduler = get_scheduler(optimizer, args)

    train_loss = []
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()

    loss_all = 0
    for data, labels, _ in tqdm(trainloader):
        data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        embedding, logits = net(data, True)
        _, loss = criterion(logits, labels, net.net.fc, embedding)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), data.size(0))
        loss_all += losses.avg
        # assert losses.avg <= loss.item()
        # assert sum(train_loss[:min(len(train_loss)//2, 10)]) >= sum(train_loss[-min(len(train_loss)//2, 10)//2:]), train_loss


if __name__ == '__main__':
    test_losses_decrese()
