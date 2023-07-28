from .resnet import resnet18
from .resnet_s import resnet20
from .densenet import densenet121
from .mlp import mlp

def generate_net(args):
    if args.num_classes == 2:
        args.num_classes = 1
    if not args.model_type in globals().keys():
        raise NotImplementedError("there has no %s" % (args.model_type))

    return globals()[args.model_type](args)

__all__ = ['generate_net']
