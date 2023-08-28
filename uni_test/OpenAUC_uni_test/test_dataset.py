import pytest
import random
import sys
import os
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
sys.path.append("./")
from XCurve.OpenAUC.dataloaders.open_set_datasets import get_class_splits, get_datasets
import argparse

from XCurve.OpenAUC.utils.model_utils import get_model
from XCurve.OpenAUC.optimizers import get_optimizer, get_scheduler
from XCurve.OpenAUC.models.wrapper_classes import Classifier32Wrapper
from XCurve.OpenAUC.dataloaders.tinyimagenet import create_val_img_folder
from tqdm import tqdm

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


def test_single_dataset():
    create_val_img_folder
    for dataset_name in ['tinyimagenet', 'mnist', 'cifar-10-10', 'cifar-10-100']:
        args = load_cfg(dataset_name)
        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx)
        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, image_size=args.image_size, seed=args.seed)
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        for rep in range(10):
            idx = random.randint(0, len(train_dataset) - 1)
            img, lbl, _ = train_dataset.__getitem__(idx)
        # assert img.shape == (3, args['input_size'][0],args['input_size'][1])

def test_dataloader():
    for dataset_name in ['tinyimagenet', 'mnist', 'cifar-10-10', 'cifar-10-100']:
        args = load_cfg(dataset_name)
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

        train_loader = dataloaders['train']
        test_loader = dataloaders['val']
        out_loader = dataloaders['test_unknown']
        batch_size = 32
        for subset in [test_loader, out_loader]:
            for i, (img, lbl, _) in enumerate(subset):
                assert img.shape[0] == lbl.shape[0] == batch_size
                # assert img.shape[1:] == (3, args['input_size'][0],args['input_size'][1])

                if (i+1)%10 == 0:
                    break
        
        for i, (img, lbl, _) in enumerate(train_loader):
            assert img.shape[0] == lbl.shape[0] == train_loader.batch_size
            # assert img.shape[1:] == (3, args['input_size'][0],args['input_size'][1])

            if (i+1)%10 == 0:
                break
                       

if __name__ == '__main__':
    test_single_dataset()
    # test_dataloader()
