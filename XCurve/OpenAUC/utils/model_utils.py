import timm
import torch
import os.path as osp
from functools import partial


from models.classifier32 import classifier32
from utils.utils import strip_state_dict, mkdir_if_missing
from utils.config import imagenet_moco_path, places_supervised_path, places_moco_path, imagenet_supervised_path

def transform_moco_state_dict_places(obj, num_classes, supervised=False):

    """
    Transforms state dict from Places pretraining here: https://github.com/nanxuanzhao/Good_transfer
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    if supervised:

        new_model = obj
        new_model['fc.weight'] = torch.randn((num_classes, 2048))
        new_model['fc.bias'] = torch.randn((num_classes,))

    else:

        newmodel = {}
        for k, v in obj.items():

            if k.startswith("fc.2"):
                continue

            if k.startswith("fc.0"):
                k = k.replace("0.", "")
                if "weight" in k:
                    v = torch.randn((num_classes, v.size(1)))
                elif "bias" in k:
                    v = torch.randn((num_classes,))

            newmodel[k] = v

    return newmodel


def transform_moco_state_dict(obj, num_classes):

    """
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")

        if k.startswith("fc.2"):
            continue

        if k.startswith("fc.0"):
            k = k.replace("0.", "")
            if "weight" in k:
                v = torch.randn((num_classes, v.size(1)))
            elif "bias" in k:
                v = torch.randn((num_classes,))

        newmodel[k] = v

    return newmodel


def get_model(args, wrapper_class=None, evaluate=False, *args_, **kwargs):

    if args.model == 'timm_resnet50_pretrained':

         # Get model
        model = timm.create_model('resnet50', num_classes=len(args.train_classes))

        # Get function to transform state_dict and state_dict path
        if args.resnet50_pretrain == 'imagenet_moco':
            pretrain_path = imagenet_moco_path
            state_dict_transform = transform_moco_state_dict
        elif args.resnet50_pretrain == 'imagenet':
            pretrain_path = imagenet_supervised_path
            state_dict_transform = partial(transform_moco_state_dict_places, supervised=False)
        elif args.resnet50_pretrain == 'places_moco':
            pretrain_path = places_moco_path
            state_dict_transform = partial(transform_moco_state_dict_places, supervised=False)
        elif args.resnet50_pretrain == 'places':
            pretrain_path = places_supervised_path
            state_dict_transform = partial(transform_moco_state_dict_places, supervised=True)
        else:
            raise NotImplementedError

        # Load pretrain weights
        state_dict = torch.load(pretrain_path) if args.resnet50_pretrain != 'imagenet_moco' \
            else torch.load(pretrain_path)['state_dict']
        state_dict = strip_state_dict(state_dict, strip_key='module.')
        state_dict = state_dict_transform(state_dict, len(args.train_classes))

        model.load_state_dict(state_dict)

    elif args.model == 'classifier32':

        try:
            feat_dim = args.feat_dim
        except:
            feat_dim = None

        model = classifier32(num_classes=len(args.train_classes), feat_dim=feat_dim)

    elif args.model in ['wide_resnet50_2', 'efficientnet_b0', 'efficientnet_b7', 'dpn92', 'resnet50']:

        model = timm.create_model(args.model, num_classes=len(args.train_classes))

    else:

        raise NotImplementedError

    if wrapper_class is not None:
        model = wrapper_class(model, *args_, **kwargs)

    return model

def save_networks(networks, result_dir, name='', loss='', criterion=None):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        torch.save(weights, filename)