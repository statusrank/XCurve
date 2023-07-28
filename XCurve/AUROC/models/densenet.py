import torch
import torch.nn as nn
import torchvision
import re


def densenet121(num_classes=1000, pretrained=False):
        model = torchvision.models.densenet121(pretrained=pretrained)
        num_feats = model.classifier.in_features
        model.classifier = nn.Linear(num_feats, num_classes)
        return model
