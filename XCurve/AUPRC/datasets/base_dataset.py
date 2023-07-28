from torch.utils.data import Dataset
import os
from torchvision import transforms
from abc import ABC, abstractmethod
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
def pil_loader(filename):
    ext = os.path.splitext(filename)[-1]
    ext = ext.lower()
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename)
        img = img.convert('RGB')
        # img = np.array(img).astype(dtype=np.uint8)
        # img = img[:,:,::-1]  #convert to BGR
        return img
    else:
        raise NotImplementedError('Unsupported file type %s'%ext)

class BaseDataset(Dataset,ABC):
    def __init__(self):
        super().__init__()

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
        return transforms.Compose([
                transforms.RandomResizedCrop(self.input_size, scale=(0.16, 1), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normal_mean, std=self.normal_std)])

    def transform_validation(self):
        return transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normal_mean, std=self.normal_std)])
