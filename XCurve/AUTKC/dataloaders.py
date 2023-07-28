import os
import os.path as osp
import numpy as np
from PIL import Image
import scipy.io.wavfile as wf

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

class MyDataset(Dataset):
    def __init__(self, data_dir='', input_size=224, split='train'):
        self.input_size = input_size
        self.split = split
        self.dataset_name = os.path.basename(data_dir)
        self.transform = self.transform_train if self.split == 'train' else self.transform_validation

        if self.dataset_name == 'tiny-imagenet-200':
            self.num_class = 200
            self.dataset = datasets.ImageFolder(osp.join(data_dir, split))
        elif self.dataset_name == 'cifar-10':
            self.num_class = 10
            split = (split == 'train')
            self.dataset = CIFAR10(root=data_dir, train=split, download=True)
        elif self.dataset_name == 'cifar-100':
            self.num_class = 100
            split = (split == 'train')
            self.dataset = CIFAR100(root=data_dir, train=split, download=True)
        elif self.dataset_name == 'place-365': # first run utils.extract_features_places365.py
            self.num_class = 365
            _data_dir = osp.join(data_dir, '{}_feature_resnet50'.format(split))
            self.data = torch.load(osp.join(_data_dir, 'features.pth'))
            self.labels = torch.load(osp.join(_data_dir, 'labels.pth'))

    def __getitem__(self, idx):
        if self.dataset_name == 'place-365':
            _data, _label = self.data[idx], self.label[idx]
            return _data, _label
        else:
            data, label = self.dataset.__getitem__(idx)
            self.data_shape = data.size
            return self.transform()(data), label
    
    def __len__(self):
        return len(self.data) if self.dataset_name == 'place-365' else self.dataset.__len__()

    def transform_train(self):
        if self.data_shape[1] <= self.input_size:
            trs = tr.Compose([
                    # tr.ToPILImage(),
                    tr.RandomHorizontalFlip(),
                    tr.ToTensor()
                ])
        else:
            trs = tr.Compose([
                    # tr.ToPILImage(),
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


def get_data_loader(data_dir, batch_size, workers, train_ratio=1):
    train_dataset, val_dataset = MyDataset(data_dir, split='train'), MyDataset(data_dir, split='val')
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
    