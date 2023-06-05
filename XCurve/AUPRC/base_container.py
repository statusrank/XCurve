import os
import torch
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict as edict
import copy
import sys
import numpy as np
import random

from models import generate_net
from dataloaders.dataset import Dataset
from dataloaders.collate_fn import collate_fn
from losses import LossWarpper
from utils.metrics import Evaluator
from utils.utils import load_pretrained_model, load_checkpoint


def worker_init_fn_seed(worker_id):
    seed = torch.initial_seed()
    seed = (worker_id + seed) % (2**32)
    np.random.seed(seed)

class BaseContainer(object):
    def __init__(self):
        # init the args necessary
        fi = open(sys.argv[1],'r')
        args = yaml.load(fi, Loader=yaml.FullLoader)
        self.args = edict(args)
        random.seed(self.args.training.seed)
        np.random.seed(self.args.training.seed)
        torch.manual_seed(self.args.training.seed)
        torch.cuda.manual_seed(self.args.training.seed)
        torch.cuda.manual_seed_all(self.args.training.seed)
        torch.backends.cudnn.benchmark = True

        self.Dataset_train = Dataset
        self.Dataset_val = Dataset
        self.args.training.cuda = not self.args.training.get('no_cuda',False)
        self.args.training.gpus = torch.cuda.device_count()
        self.batchsize = self.args.training.batchsize  ##// self.args.dataset.num_sample_per_id
        self.evaluator = Evaluator()

        self.args.dataset['batchsize'] = self.args.training.batchsize
    
    def init_training_container(self):
        # Define dataset
        self.train_set = self.Dataset_train(self.args.dataset, split='train')    
        self.val_set = self.Dataset_val(self.args.dataset, split='val')
        # self.test_set = self.Dataset_val(self.args.dataset, split='test')
        self.setup_dataloader()

        # Define network
        self.model = generate_net(self.args.models)
        self.model = self.model.cuda()

        start_it = 0
        best = 0.0
        # # Resuming checkpoint
        if self.args.training.resume_train != 'none':
            state_dict, optimizer, start_it, best = load_checkpoint(checkpoint_path=self.args.training.resume_train)

            load_pretrained_model(self.model, state_dict)
            self.model = self.model.cuda()

            if not self.args.training.ft and optimizer is not None:
                for name in self.args.training.optimizer.keys():
                    if name in optimizer.keys():
                        self.optimizer[name].load_state_dict(optimizer[name])
            else:
                start_it = 0

        self.start_it = start_it
        self.best = best

        # Define Criterion
        self.criterion = LossWarpper(self.args.training.losses)
        self.criterion.update_cnt_per_id(self.train_set.get_cnt_per_id())

        self.gen_optimizer(self.model.param_groups(
            self.args.training.optimizer.lr, 
            self.args.training.optimizer.lr_fc_mul) + \
                self.criterion.param_groups(self.args.training.optimizer.lr))

    def init_validation_container(self):
        # Define network
        self.model = generate_net(self.args.models)

        # # Resuming checkpoint
        state_dict, _, _, _ = load_checkpoint(checkpoint_path=self.args.evaluation.resume_eval)
        load_pretrained_model(self.model, state_dict)
        self.model = self.model.cuda()

        self.val_set = self.Dataset_val(self.args.dataset, split='val')
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=448,
            worker_init_fn=worker_init_fn_seed,
            collate_fn=collate_fn,
            num_workers=self.args.training.get('num_workers', 4)
        )

    def init_evaluation_container(self):
        # Define network
        self.model = generate_net(self.args.models)

        # # Resuming checkpoint
        state_dict, _, _, _ = load_checkpoint(checkpoint_path=self.args.evaluation.resume_eval)
        load_pretrained_model(self.model, state_dict)
        self.model = self.model.cuda()

        dataset_test = self.args.evaluation.dataset.dataset_test
        if not isinstance(dataset_test, list):
            dataset_test = [dataset_test]

        self.test_set = []
        for idx in range(len(dataset_test)):
            tmp_args = self.args.dataset
            tmp_args.update(self.args.evaluation.dataset)
            tmp_args.dataset_test = dataset_test[idx]
            self.test_set.append(self.Dataset_val(tmp_args, split='test'))
        self.args.evaluation.dataset.dataset_test = dataset_test
        self.args.evaluation.dataset.dataset_val = dataset_test

        self.test_loader = [DataLoader(
            d,
            batch_size=448,
            worker_init_fn=worker_init_fn_seed,
            collate_fn=collate_fn,
            num_workers=self.args.training.get('num_workers', 4)
        ) for d in self.test_set]

    def gen_optimizer(self, params):
        item = self.args.training.optimizer
        assert len(params) > 0
        lr = item.lr
        for it in item.get('lr_decay_iter', []):
            if self.start_it > it:
                lr *= item.lr_decay

        if item.optim_method == 'sgd':
            self.optimizer = torch.optim.SGD(
                params,
                momentum=item.get('momentum', 0.0),
                lr=lr,
                weight_decay=item.get('weight_decay', 0),
                nesterov=item.get('nesterov', False)
            )
        elif item.optim_method == 'adagrad':
            self.optimizer = torch.optim.Adagrad(
                params,
                lr=lr,
                weight_decay=item.get('weight_decay', 0),
            )
        elif item.optim_method == 'adam':
            self.optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=item.get('weight_decay', 0),
                betas=item.get('betas', (0.9, 0.999))
            )
        elif item.optim_method == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=item.get('weight_decay', 0),
                betas=item.get('betas', (0.9, 0.999))
            )
        else:
            raiseNotImplementedError(
                "optimizer %s not implemented!"%item.optim_method)

    def setup_dataloader(self):
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batchsize,
            worker_init_fn=worker_init_fn_seed,
            shuffle=not self.train_set.inst_blc,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=self.args.training.get('num_workers', 4)
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=448,
            worker_init_fn=worker_init_fn_seed,
            collate_fn=collate_fn,
            num_workers=self.args.training.get('num_workers', 4)
        )
        # self.test_loader = DataLoader(
        #     self.test_set,
        #     batch_size=112,
        #     worker_init_fn=worker_init_fn_seed,
        #     collate_fn=collate_fn,
        #     num_workers=self.args.training.get('num_workers', 4)
        # )

    def training(self):
        pass

    def validation(self):
        pass
