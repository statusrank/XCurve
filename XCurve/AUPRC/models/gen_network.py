import os
import importlib
import torch
from torch import nn

from .base_model import BaseModel


def generate_net(args):
    model_name = args.model
    from .retrieval_model import RetrievalModel

    all_model = [RetrievalModel]

    model = None
    for m in all_model:
        if m.__name__ == model_name:
            model = m
            break

    if model is None:
        raise NotImplementedError("there has no %s" % (model_name))

    return model(args)
