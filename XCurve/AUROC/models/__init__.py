from .resnet import *
from .resnet_s import *
from .densenet import *
from .mlp import mlp

def generate_net(args):
    if args.num_classes == 2:
        args.num_classes = 1
    if not args.model_type in globals().keys():
        raise NotImplementedError("there has no %s" % (args.model_type))

    return globals()[args.model_type](args.num_classes, args.pretrained)
