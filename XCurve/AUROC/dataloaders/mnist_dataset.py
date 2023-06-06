import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
from sampler import *
import warnings
import os
from PIL import Image
import numpy as np


class IMBALANCEMNIST(torchvision.datasets.MNIST):
    cls_num = 10
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCEMNIST, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.phat = 0.1
        self.gen_imbalanced_data(img_num_list)

    def __getitem__(self, index):
        """
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
        img, target = self.data[index], int(self.targets[index])
    
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')
    
        if self.transform is not None:
            img = self.transform(img)
    
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return img, target
        
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        targets_np = np.array(self.targets, dtype=np.int64)
        # img_max = len(self.data) / cls_num
        img_num_per_cls = []
        
        for cls_idx in range(cls_num):
            img_max = np.where(targets_np == cls_idx)[0].__len__()
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        
        return img_num_per_cls
    
    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            # print('need', the_class, the_img_num)
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # print(len(selec_idx))
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        new_targets = self.get_two_class(new_targets)
        self.data = new_data
        self.targets = new_targets
        # print(len(self.targets))
        # print(self.data.shape)
    
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def get_two_class(self, Y):
        Y = np.array(Y)
        loc_0 = np.where(Y <= 4)[0]
        loc_1 = np.where(Y > 4)[0]
        Y[loc_1] = 1
        Y[loc_0] = 0
        self.phat = len(np.where(Y == 1)[0]) / len(Y)
        return Y.tolist()
    
if __name__=='__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    train_dataset = IMBALANCEMNIST(root='../', train=True, download=True, transform=transform_train)
    
    train_sampler = StratifiedSampler(class_vector=train_dataset.targets,
                                      batch_size=128,
                                      rpos=1,
                                      rneg=9)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=(train_sampler is None),
        num_workers=2,
        # pin_memory=True,
        sampler=train_sampler)
    
    for i, (x, y) in enumerate(train_loader):
        print(i, x.min(), x.max(), x.mean(), x.std())