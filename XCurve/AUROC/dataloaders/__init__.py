from torch.utils.data import DataLoader
from .dataset import DatasetRaw, DatasetFile
from .sampler import StratifiedSampler
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

def get_datasets(args):
    train_set = DatasetFile(args, 'train')
    val_set = DatasetFile(args, 'test')
    test_set = DatasetFile(args, 'val')

    return train_set, val_set, test_set


def get_data_loaders(train_set,
                     val_set,
                     test_set,
                     train_batch_size,
                     test_batch_size,
                     num_workers=4,
                     rpos = 1,
                     rneg = 4):
    sampler = StratifiedSampler(train_set.get_labels(),
                                train_batch_size,
                                rpos = rpos,
                                rneg = rneg)
                                
    train_loader = DataLoader(train_set,
                              batch_size=sampler.real_batch_size,
                            #   shuffle=True,
                              sampler=sampler,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=test_batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=test_batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


__all__ = ['IMBALANCECIFAR10', 
           'IMBALANCECIFAR100',
           'DatasetRaw', 
           'DatasetFile', 
           'get_datasets', 
           'get_data_loaders', 
           'StratifiedSampler', 
           ]
