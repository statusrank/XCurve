import os
import logging
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import math
import numpy as np
from collections import defaultdict
import cv2
from easydict import EasyDict
import yaml

from .logger import logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count
            
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Saver(object):
    def __init__(self, args):
        self.directory = os.path.join(args.training.save_dir, args.dataset.dataset_train, args.models.model)
        self.experiment_dir = os.path.join(self.directory, args.training.experiment_id)
     
        logfile = os.path.join(self.experiment_dir, 'parameters.yaml')
        log_file = open(logfile, 'w')
        temp_args = {}
        for k,v in args.items():
            if not isinstance(v,(dict,EasyDict)):
                temp_args[k] = v
            else:
                temp_args[k] = v.__dict__
        yaml.dump(temp_args,log_file)
        log_file.close()

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk
            is_best: help to judge if it's the best state,if true,backup to the directory
        """
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

def load_checkpoint(checkpoint_path=None):
    assert(checkpoint_path is not None)
    if not os.path.isfile(checkpoint_path):
        logger.debug("=> no checkpoint found at '{}'" .format(checkpoint_path))
        raise RuntimeError("=> no checkpoint found at '{}'" .format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    start_it = 0
    optimizer = None

    if 'optimizer' in checkpoint.keys():
        start_it = checkpoint['start_it']
        best = checkpoint.get('best', 0.0)
        optimizer = checkpoint['optimizer']
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('module.','',1): v for k,v in state_dict.items()}
    else:
        logger.info('this checkpoint has no optimizer')
        state_dict = {k.replace('module.','',1): v for k,v in checkpoint.items()}
    logger.info('iteration: %.06d'%(start_it))

    return state_dict, optimizer, start_it, best

def load_pretrained_model(model, state_dict):
    model_state = model.state_dict()
    model_params = len(model_state.keys())
    checkpoint_params = len(state_dict.keys())
    logger.info('this model has {} params; this checkpoint has {} params'.format(model_params,checkpoint_params))
    if model_params > checkpoint_params:
        for i,param in model_state.items():
            if i not in state_dict.keys():
                logger.debug('this param of the model dont in the checkpoint: {} ,required grad: {}'.format(i,str(param.requires_grad)))
    num = 0
    total = 0
    for k,v in state_dict.items():
        total += 1
        if k in model_state.keys():
            if not isinstance(v,bool):
                if (v.size() != model_state[k].size()):
                    logger.info('this param {} of the checkpoint dont match the model in size: '.format(k) + str(v.size()) + ' ' + str(model_state[k].size()))
                    continue
            model_state[k] = v
            num += 1
        else:
            logger.info('this param of the checkpoint dont in the model: {}'.format(k))
    model.load_state_dict(model_state, strict=False)
    logger.info('success for loading pretrained model params {}/{}!'.format(str(num), str(total)))
