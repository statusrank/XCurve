import torch
import torch.nn as nn
import torchvision
import re

from .load import load_pretrained_model


def densenet121(args):
        model = torchvision.models.densenet121(pretrained=False)
        num_feats = model.classifier.in_features
        model.classifier = nn.Linear(num_feats, args.num_classes)

        if args.pretrained is not None:
            state_dict = torch.load(args.pretrained)
            pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

            load_pretrained_model(model, state_dict)

        return model
