import torch
import torch.nn as nn
import re

from models._resnet import _resnet50
from utils.utils import load_pretrained_model


def resnet50(args):
    model = _resnet50(
        num_classes=args.output_channels,
        pretrained=False
    )

    if args.pretrained is not None:
        state_dict = torch.load(args.pretrained)
        load_pretrained_model(model, state_dict)

    # if args.get('freeze_low_layers', False):
    #     for l in [model.conv1, model.bn1, model.layer1, model.layer2]:
    #         for p in l.parameters():
    #             p.requires_grad = False

    return model
