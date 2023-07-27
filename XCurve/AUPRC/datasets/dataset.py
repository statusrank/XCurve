import os
import os.path as osp
import torch
import random
import copy

from .base_dataset import BaseDataset
from .base_dataset import pil_loader


class RetrievalDataset(BaseDataset):
    def __init__(self, data_dir, list_dir, subset, input_size, batchsize, num_sample_per_id, normal_mean=[0.485, 0.456, 0.406], normal_std=[0.229, 0.224, 0.225], split='train', **kwargs):
        super().__init__()

        self.input_size = input_size
        self.batchsize = batchsize
        self.num_sample_per_id = num_sample_per_id
        self.normal_mean = normal_mean
        self.normal_std = normal_std

        if subset is None:
            subset = kwargs['dataset_' + split]

        self.data_dir = data_dir
        self.split = split
        if split == 'train':
            _data_list = os.path.join(list_dir, subset + '.txt')
            self.transform = self.transform_train()
        elif split == 'val':
            _data_list = os.path.join(list_dir, subset + '.txt')
            self.transform = self.transform_validation()
        elif split == 'test':
            _data_list = os.path.join(list_dir, subset + '.txt')
            self.transform = self.transform_validation()
        else:
            raise ValueError
        self._data_list = _data_list
        self.split = split

        with open(_data_list) as f:
            lines = f.read().splitlines()[1:]

        self.metas = []
        id_mp = {}
        for i, line in enumerate(lines):
            id_ = int(line.split(' ')[1])

            if not id_ in id_mp.keys():
                id_mp[id_] = len(id_mp)
                self.metas.append([])

            id_ = id_mp[id_]
            path = osp.join(self.data_dir, line.split(' ')[-1])
            self.metas[id_].append((path, id_))

        cnt_per_id = [len(i) for i in self.metas]
        max_cnt_per_id = max(cnt_per_id)
        self.cnt_per_id = cnt_per_id

        for id_ in range(len(self.metas)):
            for i in range(len(self.metas[id_])):
                self.metas[id_][i] = tuple(list(self.metas[id_][i]) + \
                    [cnt_per_id[id_] / max_cnt_per_id])

        self.reset()
        print('%s set has %d samples per epoch' % (self.split, self.__len__()))

    def reset(self):
        print('\nshuffling data...')
        datalist = []

        metas = copy.deepcopy(self.metas)
        num_classes = len(metas)
        for i in range(num_classes):
            random.shuffle(metas[i])

        bs = self.batchsize
        ns = self.num_sample_per_id

        classes = [i for i in range(num_classes)]
        random.shuffle(classes)

        mini_batch = []
        while True:
            for clss in classes:
                if len(metas[clss]) >= ns and len(mini_batch) < bs:
                    mini_batch += metas[clss][:ns]
                    metas[clss] = metas[clss][ns:]

                if len(mini_batch) == bs:
                    break

            if len(mini_batch) == bs:
                datalist.append(mini_batch)
                mini_batch = []
            else:
                break

        random.shuffle(datalist)
        self.datalist = []
        for i in datalist:
            self.datalist += i

    def __len__(self):
        return len(self.datalist)

    def __str__(self):
        return self.data_dir + '  split=' + str(self.split)

    def get_cnt_per_id(self):
        return self.cnt_per_id

    def __getitem__(self, idx):
        img_filename = self.datalist[idx][0]
        img = pil_loader(img_filename)
        img = self.transform(img)
        
        return img, torch.tensor([int(self.datalist[idx][1])])
