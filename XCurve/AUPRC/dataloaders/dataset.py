from .base_dataset import BaseDataset
from .base_dataset import pil_loader
import numpy as np
import os
import os.path as osp
import torch
from utils.logger import logger
import random
import cv2
import lmdb
from tqdm import tqdm
import pickle
import fcntl
import copy

debug = False

def build_lmdb(save_path, metas, commit_interval=1000):
    with open('lock', 'w') as f:
        if not save_path.endswith('.lmdb'):
            raise ValueError("lmdb_save_path must end with 'lmdb'.")

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        if osp.exists(save_path):
            # print('Folder [{:s}] already exists.'.format(save_path))
            return

        if not osp.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))

        metas_new = []
        for i in metas:
            metas_new += i
        metas = metas_new

        data_size_per_img = cv2.imread(metas[0][0], cv2.IMREAD_UNCHANGED).nbytes
        data_size = data_size_per_img * len(metas)
        env = lmdb.open(save_path, map_size=data_size * 10)
        txn = env.begin(write=True)

        shape = dict()

        print('Building lmdb...')
        for i in tqdm(range(len(metas))):
            image_filename = metas[i][0]
            img = pil_loader(filename=image_filename)
            img = np.array(img).astype(dtype=np.uint8)
            assert img is not None and len(img.shape) == 3

            txn.put(image_filename.encode('ascii'), img.copy(order='C'))
            shape[image_filename] = '{:d}_{:d}_{:d}'.format(img.shape[0], img.shape[1], img.shape[2])

            if i % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

        pickle.dump(shape, open(os.path.join(save_path, 'meta_info.pkl'), "wb"))

        txn.commit()
        env.close()
        print('Finish writing lmdb.')

class Dataset(BaseDataset):
    def __init__(self, args, split='train', **kwargs):
        super().__init__(args)
        self.data_dir = args.data_dir
        self.split = split
        if split == 'train':
            args.train_list = os.path.join(args.list_dir, args.dataset_train + '.txt')
            self.transform = self.transform_train()
            _data_list = args.train_list
        elif split == 'val':
            args.val_list = os.path.join(args.list_dir, args.dataset_val+'.txt')
            self.transform = self.transform_validation()
            _data_list = args.val_list
        elif split == 'test':
            args.test_list = os.path.join(args.list_dir, args.dataset_test+'.txt')
            self.transform = self.transform_validation()
            _data_list = args.test_list
        else:
            raise ValueError
        self._data_list = _data_list
        self.split = split
        self.inst_blc = self.args.get('inst_blc', True) and split == 'train'

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
        logger.info('%s set has %d samples per epoch' % (self.split, self.__len__()))
    
        if self.args.get('lmdb_dir', None) is not None:
            self._load_image = self._load_image_lmdb
        else:
            self._load_image = self._load_image_pil

        self.initialized = False


    def reset(self):
        if self.inst_blc:
            print('\nshuffling data...')
            datalist = []

            metas = copy.deepcopy(self.metas)
            num_classes = len(metas)
            for i in range(num_classes):
                random.shuffle(metas[i])

            bs = self.args.batchsize
            ns = self.args.num_sample_per_id

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
        else:
            self.datalist = []
            for i in self.metas:
                self.datalist += i

    def _init_lmdb(self):
        if not self.initialized:
            lmdb_dir = osp.join(self.args.lmdb_dir, self._data_list.split('/')[-1].split('.')[0] + '.lmdb')
            build_lmdb(lmdb_dir, self.metas)
            env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
            self.lmdb_txn = env.begin(write=False)
            self.meta_info = pickle.load(open(os.path.join(lmdb_dir, 'meta_info.pkl'), "rb"))
            self.initialized = True

    def __len__(self):
        # if self.args.get('one_sample', False):
        #     return len(self.metas)

        return len(self.datalist)

    def __str__(self):
        return self.args.data_dir + '  split=' + str(self.split)

    def _load_image_pil(self, filename):
        img = pil_loader(filename=filename)
        return img
    
    def get_cnt_per_id(self):
        return self.cnt_per_id

    def _load_image_lmdb(self, filename):
        self._init_lmdb()
        img_buff = self.lmdb_txn.get(filename.encode('ascii'))
        C, H, W = [int(i) for i in self.meta_info[filename].split('_')]
        img = np.frombuffer(img_buff, dtype=np.uint8).reshape(C, H, W)
        return img
    
    def _load_one_sample(self, idx, with_imgid=False):
        sample = {
            'image': self._load_image(self.metas[idx][0]),
            'label': torch.tensor([self.metas[idx][1]]).long(),
            'freq': torch.tensor([self.metas[idx][3]]).float(),
        }
        if with_imgid:
            sample['imgid'] = self.metas[idx][2]
        sample['image'] = self.transform(sample['image'])
        return sample

    def _getitem_inst_blc(self, idx):
        img_filename = self.datalist[idx][0]
        img = self._load_image(img_filename)
        img = self.transform(img)
        return {
            'image': img.unsqueeze(0),
            'freq': torch.tensor([1]),
            'label': torch.tensor([int(self.datalist[idx][1])])
        }


    # def _getitem_cls_blc(self, cls_id):
    #     l, r = self.cls_id2interal[cls_id]
    #     if self.split == 'train':
    #         n_sample = self.args.num_sample_per_id
    #         idx = np.random.choice(range(l, r+1), n_sample, replace=r-l+1<n_sample)
    #         assert False
    #     else:
    #         idx = np.arange(l, r+1)
            
    #     imgs = []
    #     lbls = []
    #     freqs = []
    #     for i in idx:
    #         s = self._load_one_sample(i)
    #         imgs.append(s['image'])
    #         lbls.append(s['label'])
    #         freqs.append(s['freq'])

    #     imgs = torch.stack(imgs, 0)
    #     lbls = torch.cat(lbls, 0)
    #     freqs = torch.cat(freqs, 0)

    #     return {
    #         'image': imgs,
    #         'label': lbls,
    #         'freq': freqs
    #     }

    def __getitem__(self, idx):
        # if self.args.get('one_sample', False):
        #     return self._load_one_sample(idx, True)

        return self._getitem_inst_blc(idx)

    def load_by_imgid(self, imgid):
        idx = self.imgid2metaidx[imgid]
        sample = {
            'image': self._load_image(self.metas[idx][0]),
            'label': self.metas[idx][1],
            'imgid': self.metas[idx][2],
            'freq': self.metas[idx][3]
        }

        assert sample['imgid'] == imgid
        return sample
