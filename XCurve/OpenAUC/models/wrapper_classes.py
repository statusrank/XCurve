import torch
from torch import nn

class TimmResNetWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, return_features=True):
        x = self.net.forward_features(x)
        embedding = self.net.global_pool(x)
        if self.net.drop_rate:
            embedding = torch.nn.functional.dropout(embedding, p=float(self.drop_rate), training=self.training)
        preds = self.net.fc(embedding)

        return embedding, preds if return_features else preds

class Classifier32Wrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def forward(self, x, return_features=True):
        return self.net(x, return_features)