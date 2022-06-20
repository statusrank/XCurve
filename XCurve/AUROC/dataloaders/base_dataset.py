import numpy as np
import os
import scipy.io as scio

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from imblearn.under_sampling import TomekLinks,\
    InstanceHardnessThreshold, NearMiss
from imblearn.over_sampling import ADASYN, BorderlineSMOTE

from . import custom_transforms as tr


ImageFile.LOAD_TRUNCATED_IMAGES = True
def pil_loader(filename, label=False):
    ext = (os.path.splitext(filename)[-1]).lower()
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename)
        if not label:
            img = img.convert('RGB')
            img = np.array(img).astype(dtype=np.uint8)
            img = img[:,:,::-1]  #convert to BGR
        else:
            if img.mode != 'L' and img.mode != 'P':
                img = img.convert('L')
            img = np.array(img).astype(dtype=np.uint8)
    elif ext == '.mat':
        img = scio.loadmat(filename)
    elif ext == '.npy':
        img = np.load(filename, allow_pickle=True)
    else:
        raise NotImplementedError('Unsupported file type %s'%ext)

    return img


class BaseDataset(Dataset,ABC):
    def __init__(self, split, input_size, norm_params, resampler_type):
        super().__init__()
        self.input_size = input_size
        self.norm_params = norm_params

        if split == 'train':
            self.transform = self.transform_train()
        elif split == 'val':
            self.transform = self.transform_validation()
            resampler_type = 'None'
        elif split == 'test':
            self.transform = self.transform_validation()
            resampler_type = 'None'
        else:
            raise ValueError

        resample_dict = self.resample_dict()
        if resampler_type not in resample_dict.keys():
            raise RuntimeError('Unknown sampler_type: %s'%resampler_type)
        self.resampler = resample_dict[resampler_type]

    def resample_dict(self):
        return {
            'TL': TLResampler,
            'IHT': IHTResampler,
            'NM': NMResampler,
            'BS': BSResampler,
            'ADA': ADAResampler,
            'None': EmptyResampler
        }
    
    def resample(self, x, y):
        return self.resampler(x, y)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __str__(self):
        pass
    
    @staticmethod
    def modify_commandline_options(parser,istrain=False):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def transform_train(self):
        temp = []
        temp.append(tr.Resize(self.input_size))

        temp.append(tr.RandomHorizontalFlip())
        temp.append(tr.RandomRotate(15))
        temp.append(tr.RandomCrop(self.input_size))

        temp.append(tr.Normalize(self.norm_params.mean, self.norm_params.std))
        temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)
        return composed_transforms

    def transform_validation(self):
        temp = []
        temp.append(tr.Resize(self.input_size))
        # temp.append(tr.RandomCrop(self.args.input_size))
        temp.append(tr.Normalize(self.norm_params.mean, self.norm_params.std))
        temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)
        return composed_transforms


class EmptyResampler():
    """
    Empty Resampler. Do nothing.
    """
    def resample(self, x, y):
        return x, y


class TLResampler():
    """
    Dataset resampled by TomekLinks
    """
    def resample(self, x, y):
        resampler = TomekLinks(sampling_strategy='majority')
        return resampler.fit_resample(x, y)


class IHTResampler():
    """
    Dataset resampled by InstanceHardnessThreshold
    """
    def resample(self, x, y):
        resampler = InstanceHardnessThreshold(sampling_strategy='majority')
        return resampler.fit_resample(x, y)


class NMResampler():
    """
    Dataset resampled by NearMiss
    """
    def resample(self, x, y):
        resampler = NearMiss(sampling_strategy='majority')
        return resampler.fit_resample(x, y)


class BSResampler():
    """
    Dataset resampled by BorderlineSMOTE
    """
    def resample(self, x, y):
        resampler = BorderlineSMOTE()
        return resampler.fit_resample(x, y)


class ADAResampler():
    """
    Dataset resampled by ADASYN
    """
    def resample(self, x, y):
        resampler = ADASYN(sampling_strategy='minority')
        return resampler.fit_resample(x, y)
