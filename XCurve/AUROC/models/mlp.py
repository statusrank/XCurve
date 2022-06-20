import torch


class UnitBallClipper(object):
    def __init__(self, frequency, radius):
        self.frequency = frequency
        self.radius = radius

    def __call__(self, module):
        """
        filter the variables to get the ones you want
        """
        if hasattr(module, 'weight'):
            w = module.weight.data
            scales = torch.norm(w, 2)
            if scales > self.radius:
                w.div_(torch.norm(w, 2).expand_as(w))
                w.mul_(self.radius)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, dim=[21527, 128, 64, 12], num_classes=1):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        for i in range(1, len(dim)):
            layers += [
                torch.nn.Linear(dim[i-1], dim[i]),
                torch.nn.BatchNorm1d(dim[i]),
                torch.nn.ReLU()
            ]
        layers += [
            torch.nn.Linear(dim[-1], num_classes),
            torch.nn.Softmax()
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers(x)

def mlp(args):
    model = MultiLayerPerceptron(args.dim, args.num_classes)
    return model

# def get_model(input_dim, output_dim):
#     return torch.nn.Sequential(
#         # torch.nn.Linear(21527, 128),
#         torch.nn.Linear(input_dim, 128),
#         torch.nn.BatchNorm1d(128),
#         torch.nn.ReLU(),
#         torch.nn.Linear(128, 64),
#         torch.nn.BatchNorm1d(64),
#         torch.nn.ReLU(),
#         torch.nn.Linear(64, output_dim),
#         # torch.nn.Linear(64, 12),
#         torch.nn.Softmax()).cuda()


# def get_clipper(frequency, radius):
#     return UnitBallClipper(frequency, radius)
