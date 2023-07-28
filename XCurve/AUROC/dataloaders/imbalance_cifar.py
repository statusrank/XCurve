#coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
# from sampler import *

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.phat = 0.1
        self.gen_imbalanced_data(img_num_list)

        # print(type(self.data.shape))
        
        
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        new_targets = self.get_two_class(new_targets)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_two_class(self, Y):
        Y = np.array(Y)
        # for i in range(10):
        #     print(i, len(np.where(Y ==i)[0]))
        loc_0 = np.where(Y <= (self.cls_num/2-1))[0]
        loc_1 = np.where(Y > (self.cls_num/2-1))[0]
        Y[loc_1] = 1
        Y[loc_0] = 0
        # for i in range(2):
        #     print(i, len(np.where(Y == i)[0]))
        self.phat = len(np.where(Y == 1)[0])/len(Y)
        return Y.tolist()

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

    def get_two_class(self, Y):
        Y = np.array(Y)
        # for i in range(10):
        #     print(i, len(np.where(Y ==i)[0]))
        loc_0 = np.where(Y <= (self.cls_num/2-1))[0]
        loc_1 = np.where(Y > (self.cls_num/2-1))[0]
        Y[loc_1] = 1
        Y[loc_0] = 0
        print('positive sample', len(loc_1))
        print('negative sample', len(loc_0))
        self.phat = len(np.where(Y == 1)[0]) / len(Y)
        return Y.tolist()


# if __name__ == '__main__':
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     train_dataset = IMBALANCECIFAR100(root='../', train=True, download=True, transform=transform_train)

#     train_sampler = StratifiedSampler(class_vector=train_dataset.targets,
#                                       batch_size=128,
#                                       rpos=1,
#                                       rneg=9)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=128, shuffle=(train_sampler is None),
#         num_workers=2,
#         # pin_memory=True,
#         sampler=train_sampler)


#     for i, (x, y) in enumerate(train_loader):
#         print(i, x.min(), x.max(), x.mean(), x.std())
