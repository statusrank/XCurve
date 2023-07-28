import torch.nn.functional as F

from .base_model import BaseModel
from .resnet import resnet50


class RetrievalModel(BaseModel):
    def __init__(self, output_channels, freeze=False, freezeBN=False):
        super(RetrievalModel, self).__init__()
        self.output_channels = output_channels
        self.backbone = resnet50(output_channels)

        if freeze:
            self.freeze()

        if freezeBN:
            self.freezeBN()

    def forward(self, x):
        """ Forward

        """
        z = self.backbone(x)
        z = z.view(z.shape[0], -1)
        z = F.normalize(z, dim=1)

        return z

    def param_groups(self, lr=None, lr_fc_mul=1):
        params = list(filter(lambda x: 'fc' not in x[0] and x[1].requires_grad, self.named_parameters()))
        params = [x[1] for x in params]
        fc_params = self.backbone.fc.parameters()

        if len(params):
            if lr is not None:
                return [{'params': params, 'lr': lr}, 
                         {'params': fc_params, 'lr': lr*lr_fc_mul}]
            else:
                return [{'params': params}, {'params': fc_params}]
        else:
            return []
