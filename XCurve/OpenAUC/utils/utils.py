import os
import os.path as osp
import sys
import errno
import torch
import random
import numpy as np
import argparse

from datetime import datetime



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def init_experiment(args, runner_name=None):

    args.cuda = torch.cuda.is_available()

    if args.device == 'None':
        args.device = torch.device("cuda:0" if args.cuda else "cpu")
    else:
        args.device = torch.device(args.device if args.cuda else "cpu")

    print(args.gpus)

    # Get filepath of calling script
    # if runner_name is None:
    #     runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]
    # root_dir = os.path.join(args.exp_root, *runner_name)

    root_dir = args.exp_root
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Unique identifier for experiment
    now = str(datetime.now()).replace(' ', '-')

    log_dir = os.path.join(root_dir, now)
    while os.path.exists(log_dir):
        now = str(datetime.now()).replace(' ', '-')
        log_dir = os.path.join(root_dir, now)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    os.mkdir(model_root_dir)

    args.model_dir = model_root_dir

    print(f'Experiment saved to: {args.log_dir}')
    print(runner_name)
    print(args)

    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()