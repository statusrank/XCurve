import torch.nn.functional as F

from models.base_model import BaseModel
from models.resnet import resnet50

def Euclidean_distance(x, y):
    return ((x - y)**2).sum(-1)

def cosine_distance(x, y):
    return (F.normalize(x) * F.normalize(y)).sum(-1)

class RetrievalModel(BaseModel):
    """
    """
    def __init__(self, args):
        super(RetrievalModel, self).__init__(args)

        self.backbone = resnet50(args)

        if args.get('freeze', False):
            self.freeze()

        if args.get('freeze_bn', False):
            self.freezeBN()

    def forward(self, samples, mode='test'):
        """ Forward

        """
        y = samples['label']
        z = self.backbone(samples['image'])

        batch_size = y.shape[0]
        z = z.view(batch_size, -1)
        y = y.view(batch_size)

        z = F.normalize(z, dim=1)

        out = {
            'feat': z,
            'target': y
        }

        return out

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
