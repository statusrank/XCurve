import os
import os.path as osp
from time import time
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import scipy.io.wavfile as wf
from multiprocessing import shared_memory
from functools import reduce

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision import transforms as tr
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler

def pil_loader(filename, label=False):
    ext = os.path.splitext(filename)[-1]
    ext = ext.lower()
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename).resize((256, 256)) 
        if not label:
            img = img.convert('RGB')
            img = np.array(img).astype(dtype=np.uint8)
            img = img[:,:,::-1]  #convert to BGR
        else:
            if img.mode != 'L' and img.mode != 'P':
                img = img.convert('L')
            img = np.array(img).astype(dtype=np.uint8)
        return img
    elif ext == '.wav':
        rate, data = wf.read(filename)
        if rate != 16000:
            raise RuntimeError('input wav must be sampled at 16,000 Hz, get %d Hz'%rate)
        if data.ndim > 1:
            # take the left channel
            data = data[:, 0]
        if data.shape[0] < 16000*10:
            # make the wav at least 10-second long
            data = np.tile(data, (16000*10 + data.shape[0] - 1) // data.shape[0])
        # take the first 10 seconds
        data = np.reshape(data[:16000*10], [-1]).astype(np.float32)
        return data
    elif ext == '.npy':
        data = np.load(filename, allow_pickle=True)
        return data.T.reshape((1, 64, -1))[:,:,:1000]
    else:
        raise NotImplementedError('Unsupported file type %s'%ext)

class MemoryDataset(Dataset):
    def __init__(self, data_dir='', input_size=224, split='train', mode='use', shm_dir='shm'):
        self.input_size = input_size
        self.split = split
        self.dataset_name = os.path.basename(data_dir)
        self.shm_dir = os.path.join(shm_dir, self.dataset_name)
        if mode == 'create':
            print('loading data to memory...')
            if self.dataset_name == 'tiny-imagenet-200':
                _dataset = datasets.ImageFolder(osp.join(data_dir, split))
                self.num_class = len(_dataset.classes)
                data, labels = list(), list()
                for file_path, label in tqdm(_dataset.imgs):
                    data.append(pil_loader(file_path))
                    labels.append(label)
                self.load_data_to_memory(np.array(data), np.array(labels))
            elif self.dataset_name == 'cifar-100':
                split = True if split == 'train' else False
                _dataset = CIFAR100(root=data_dir, train=split, download=False)
                self.num_class = len(_dataset.classes)
                self.load_data_to_memory(np.array(_dataset.data), np.array(_dataset.targets))
            elif self.dataset_name == 'cifar-10':
                split = True if split == 'train' else False
                _dataset = CIFAR10(root=data_dir, train=split, download=True)
                self.num_class = len(_dataset.classes)
                self.load_data_to_memory(np.array(_dataset.data), np.array(_dataset.targets))
            elif self.dataset_name == 'place-365':
                self.num_class = 365
                _data_dir = osp.join(data_dir, '{}_feature_resnet50'.format(split))
                data = torch.load(osp.join(_data_dir, 'features.pth'))
                labels = torch.load(osp.join(_data_dir, 'labels.pth'))
                self.load_data_to_memory(data.numpy(), labels.numpy())
            else:
                raise ValueError
                
        elif mode == 'use':
            self.feature = True if self.dataset_name == 'place-365' else False
            self.load_data_from_memory()
        else:
            raise ValueError        
    
    def load_data_to_memory(self, data, labels):
        os.makedirs(self.shm_dir) if not os.path.exists(self.shm_dir) else None

        print('creating shared memory...')
        self.shm_data = shared_memory.SharedMemory(create=True, size=data.nbytes)
        shm_info_data = {
            'shm_name': self.shm_data.name, 
            'data_shape': data.shape,
            'data_type': data.dtype.str,
            'num_class': self.num_class
        }
        print(shm_info_data), print(json.dumps(shm_info_data), file=open(osp.join(self.shm_dir, '{}_{}'.format(self.split, 'data')), 'w'))

        self.shm_labels = shared_memory.SharedMemory(create=True, size=labels.nbytes)
        shm_info_labels = {
            'shm_name': self.shm_labels.name, 
            'data_shape': labels.shape,
            'data_type': labels.dtype.str,
            'num_class': self.num_class
        }
        print(shm_info_labels), print(json.dumps(shm_info_labels), file=open(osp.join(self.shm_dir, '{}_{}'.format(self.split, 'labels')), 'w'))

        print('copying data to shared memory...')
        data_buffer = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm_data.buf)
        data_buffer[:] = data[:]
        labels_buffer = np.ndarray(labels.shape, dtype=labels.dtype, buffer=self.shm_labels.buf)
        labels_buffer[:] = labels[:]

    def load_data_from_memory(self):
        print('loading data from', self.shm_dir, '...')

        self.shm_info_data = json.load(open(osp.join(self.shm_dir, '{}_{}'.format(self.split, 'data')), 'r'))
        self.num_class = self.shm_info_data['num_class']
        self.data_shape = self.shm_info_data['data_shape']
        self.shm_data = shared_memory.SharedMemory(name=self.shm_info_data['shm_name'])
        self.data_len = reduce(lambda x, y: x* y, self.shm_info_data['data_shape'][1:]) * int(self.shm_info_data['data_type'][2:])

        self.shm_info_labels = json.load(open(osp.join(self.shm_dir, '{}_{}'.format(self.split, 'labels')), 'r'))
        self.shm_labels = shared_memory.SharedMemory(name=self.shm_info_labels['shm_name'])
        self.label_len = int(self.shm_info_labels['data_type'][2:])

        self.transform = self.transform_train() if self.split == 'train' else self.transform_validation()

    def __getitem__(self, idx):
        data = np.frombuffer(
            self.shm_data.buf[idx * self.data_len: (idx + 1) * self.data_len], 
            dtype=self.shm_info_data['data_type']
        )
        data = data.reshape(self.shm_info_data['data_shape'][1:])
        label = np.frombuffer(
            self.shm_labels.buf[idx * self.label_len: (idx + 1) * self.label_len], 
            dtype=self.shm_info_labels['data_type']
        )
        return self.transform(data) if not self.feature else data, label
    
    def __len__(self):
        return self.shm_info_data['data_shape'][0]

    def transform_train(self):
        if self.data_shape[1] <= self.input_size:
            trs = tr.Compose([
                    tr.ToPILImage(),
                    tr.RandomHorizontalFlip(),
                    tr.ToTensor()
                ])
        else:
            trs = tr.Compose([
                    tr.ToPILImage(),
                    tr.RandomHorizontalFlip(),
                    tr.RandomCrop(self.input_size),
                    tr.ToTensor()
                ])            
        return trs

    def transform_validation(self):
        return tr.Compose([
                    tr.ToPILImage(),
                    tr.ToTensor()
                ])


def data_loader(data_dir, batch_size, workers, train_ratio=1):
    train_dataset, val_dataset = MemoryDataset(data_dir, split='train'), MemoryDataset(data_dir, split='val')
    if train_ratio == 1:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        return train_loader, val_loader, val_loader, train_dataset.num_class
    else:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(num_train * train_ratio)
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            sampler=train_sampler
        )
        val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            sampler=valid_sampler
        )
        test_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader, train_dataset.num_class


if __name__ == "__main__":
    # tiny-imagenet-200, cifar-100, cifar-10, place-365
    data_dir, dataset, mode = 'dataset', 'place-365', 'create'
    data_dir = os.path.join(data_dir, dataset)
    val_dataset = MemoryDataset(data_dir, split='val', mode=mode)
    train_dataset = MemoryDataset(data_dir, split='train', mode=mode)

    t = time()
    while True:
        tmp = val_dataset
        if time() - t > 5:
            t = time()
            print('%s: providing [%s] data in shared memory... %s and %s' % (t, dataset, train_dataset.shm_data.name, val_dataset.shm_data.name))
