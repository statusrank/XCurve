from torch import nn

class classifier32(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128):
        super(self.__class__, self).__init__()

        if feat_dim is None:
            feat_dim = 128

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    feat_dim,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(feat_dim)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.pre = nn.Sequential(
            self.dr1,
            self.conv1,
            self.bn1,
            nn.LeakyReLU(0.2),
            self.conv2,
            self.bn2,
            nn.LeakyReLU(0.2),
            self.conv3,
            self.bn3,
            nn.LeakyReLU(0.2),

            self.dr2,
            self.conv4,
            self.bn4,
            nn.LeakyReLU(0.2),
            self.conv5,
            self.bn5,
            nn.LeakyReLU(0.2),
            self.conv6,
            self.bn6,
            nn.LeakyReLU(0.2),
            self.dr3,
            self.conv7,
            self.bn7,
            nn.LeakyReLU(0.2),
            self.conv8,
            self.bn8,
            nn.LeakyReLU(0.2),

            self.conv9,
            self.bn9,
            nn.LeakyReLU(0.2),
            self.avgpool,
            nn.Flatten(1),
        )

        self.post = self.fc

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_embedding=True):
        embedding = self.pre(x)
        preds = self.post(embedding)
        return embedding, preds if return_embedding else preds

def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)