import numpy as np
import os
import os.path as osp
import pandas as pd
import pickle
import lmdb

from .base_dataset import BaseDataset
from .base_dataset import pil_loader
from .utils import build_lmdb, get_all_files


class DatasetRaw(BaseDataset):
    def __init__(self, data, targets, 
            split='train',
            input_size=224,
            norm_params={'mean': [123.675, 116.280, 103.530], 
                    'std': [58.395, 57.120, 57.375]},
            resampler_type='None',
            load_data_from_file=False,
            lmdb_dir=None,
            **kwargs
        ):
        super().__init__(split, input_size, norm_params, resampler_type)
        assert targets.min() >= 0 and targets.max() > 0
        # print(targets.max(), targets.min(), len(targets), targets.sum())
        self.targets = targets

        ## a list of meta data, each element is (feature, target) or (img path, target)
        self.metas = [(d,t) for d,t in zip(data, targets)]

        ## directory path to save lmdb files
        if lmdb_dir is not None:
            lmdb_dir = osp.join(lmdb_dir, split + '.lmdb')
        self._lmdb_dir = lmdb_dir
        ## number of meta data
        self._num = len(self.metas)
        ## number of positive classes
        self._n_cls_p = int(targets.max())

        # ## info for resampling
        # if self._n_cls_p == 1:
        #     self._labels = [int(i[1].sum() != 0) for i in self.metas]
        # else:
        #     self._labels = [np.argmax(i[1]) for i in self.metas]
        # print(self._labels)

        self._cls_num_list = pd.Series(targets).value_counts().sort_index().values
        self._freq_info = [
            num * 1.0 / sum(self._cls_num_list) for num in self._cls_num_list
        ]

        self._num_classes = len(self._cls_num_list)
        # self._class_dim = len(set(self._labels))
        self._class_dim = len(set(targets))

        if load_data_from_file:
            self._getitem = self._getitem_file
            if self._lmdb_dir:
                build_lmdb(self._lmdb_dir, self.metas)
                self.initialized = False
                self._load_image = self._load_image_lmdb
            else:
                self._load_image = self._load_image_pil            
        else:
            self._getitem = self._getitem_data

    def _init_lmdb(self):
        if not self.initialized:
            env = lmdb.open(self._lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
            self.lmdb_txn = env.begin(write=False)
            self.meta_info = pickle.load(open(os.path.join(self._lmdb_dir, 'meta_info.pkl'), "rb"))
            self.initialized = True

    def _load_image_lmdb(self, img_filename):
        self._init_lmdb()
        img_buff = self.lmdb_txn.get(img_filename.encode('ascii'))
        C, H, W = [int(i) for i in self.meta_info[img_filename].split('_')]
        img = np.frombuffer(img_buff, dtype=np.uint8).reshape(C, H, W)
        return img

    def _load_image_pil(self, img_filename):
        return pil_loader(img_filename)

    def get_class_dim(self):
        return self._class_dim

    def get_labels(self):
        return self.targets

    def get_cls_num_list(self):
        return self._cls_num_list

    def get_freq_info(self):
        return self._freq_info

    def get_num_classes(self):
        return self._num_classes

    def __len__(self):
        return self._num

    def __str__(self):
        return 'dataset: split=' + str(self.split)

    def _getitem_file(self, idx):
        sample = {
            'image': self._load_image(self.metas[idx][0]),
            'label': self.metas[idx][1]
        }
        sample = self.transform(sample)
        return sample['image'], sample['label']

    def _getitem_data(self, idx):
        sample = {
            'image': self.metas[idx][0],
            'label': self.metas[idx][1]
        }
        sample = self.transform(sample)
        return sample['image'], sample['label']
    
    def __getitem__(self, idx):
        return self._getitem(idx)


class DatasetFile(DatasetRaw):
    def __init__(self, args, split='train'):
        data_dir = osp.join(args.data_dir, split)

        ## get classes
        if not 'class2id' in args.keys():
            class2id = dict()
            for i in range(args.num_classes):
                class2id[str(i)] = i
        else:
            class2id = args.get('class2id')

        ## check file type
        if osp.isdir(data_dir):
            npy_style = False
        elif osp.exists(data_dir + '.npy'):
            data_dir += '.npy'
            npy_style = True
        else:
            raise IOError('data directory not exists: %s'%data_dir)

        ## load meta data
        data = None
        targets = None
        if npy_style:
            tmp = np.load(data_dir, allow_pickle=True).item()
            ori_data = tmp['data']
            ori_targets = tmp['targets']
            data = []
            targets = []
            for i in range(len(ori_data)):
                id_ = class2id.get(str(int(ori_targets[i])), 0)
                if id_ >= 0:
                    data.append(ori_data[i])
                    targets.append(id_)
            data = np.array(data)
            targets = np.array(targets)
            assert len(data) == len(targets)
        else:
            img_list = get_all_files(data_dir, ['jpg', 'jpeg', 'png'])
            data, targets = self._gen_metas(img_list, class2id)

        args['data'] = data
        args['targets'] = targets
        args['load_data_from_file'] = not npy_style
        args['split'] = split
        super().__init__(**args)

    def _gen_metas(self, img_list, class2id):
        data, targets = [], []
        for i in img_list:
            cls_id = class2id.get(i.split('/')[-2], 0)
            if cls_id < 0:
                continue
            data.append(i)
            targets.append(cls_id)
        targets = np.array(targets)
        return data, targets
