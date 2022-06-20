import os
import os.path as osp
import cv2
import pickle
import lmdb
from tqdm import tqdm

from .base_dataset import pil_loader


def build_lmdb(save_path, metas, commit_interval=1000):
    if not save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if osp.exists(save_path):
        print('Folder [{:s}] already exists.'.format(save_path))
        return

    if not osp.exists('/'.join(save_path.split('/')[:-1])):
        os.makedirs('/'.join(save_path.split('/')[:-1]))

    data_size_per_img = cv2.imread(metas[0][0], cv2.IMREAD_UNCHANGED).nbytes
    data_size = data_size_per_img * len(metas)
    env = lmdb.open(save_path, map_size=data_size * 10)
    txn = env.begin(write=True)

    shape = dict()

    print('Building lmdb...')
    for i in tqdm(range(len(metas))):
        image_filename = metas[i][0]
        img = pil_loader(filename=image_filename)
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

def get_all_files(dir, ext):
    for e in ext:
        if dir.lower().endswith(e):
            return [dir]

    if not osp.isdir(dir):
        return []

    file_list = os.listdir(dir)
    ret = []
    for i in file_list:
        ret += get_all_files(osp.join(dir, i), ext)
    return ret