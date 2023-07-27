import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class StandardOpenAUCLoss(nn.Module):
    def __init__(self, loss_close, **options):
        super().__init__()
        self.loss_close = loss_close
        self.num_classes = options['num_classes']
        self.alpha = options['alpha']
        self.lambd = options['lambda']
    
    def loss_open(self, outputs_real, labels, outputs_fake, mask):
        pred_neg, predictions = outputs_real.max(axis=1)
        hit = (predictions.data == labels.data).float() * mask
        pred_pos, _ = outputs_fake.max(axis=1)
        
        return (hit * (pred_pos - pred_neg + 1) ** 2).sum(), hit.sum()

    def forward(self, logits, labels, f_post, manifolds):
        _, loss = self.loss_close(logits, labels)

        half_lenth = manifolds.size(0) // 2
        if 2 * half_lenth != manifolds.size(0):
            return logits, loss
        laterhalf_manifolds = manifolds[half_lenth:]
        laterhalf_labels = labels[half_lenth:]

        shuffle_ix = np.random.permutation(np.arange(half_lenth))
        shuffle_ix = torch.tensor(list(shuffle_ix)).int().cuda()
        shuffle_laterhalf_labels = torch.index_select(laterhalf_labels, 0, shuffle_ix)
        shuffle_laterhalf_manifolds = torch.index_select(laterhalf_manifolds, 0, shuffle_ix)
        mask = (shuffle_laterhalf_labels.data != laterhalf_labels.data).float()

        lam = np.random.beta(self.alpha, self.alpha)
        mixup_manifolds = lam * laterhalf_manifolds + (1 - lam) * shuffle_laterhalf_manifolds
        outputs_fake = f_post(mixup_manifolds)

        loss_fake, n = self.loss_open(logits[:half_lenth], labels[:half_lenth], outputs_fake, mask)
        loss = loss + loss_fake / n * self.lambd if n > 0 else loss

        return logits, loss
    

class Softmax(nn.Module):
    def __init__(self, **options):
        super(Softmax, self).__init__()
        self.temp = options['temp']
        self.label_smoothing = options['label_smoothing']

    def forward(self, logits, labels=None):
        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss = F.cross_entropy(logits / self.temp, labels)
        else:
            loss = smooth_cross_entropy_loss(logits / self.temp, labels=labels, smoothing=self.label_smoothing, dim=-1)

        return logits, loss


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):

    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1

    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

def smooth_cross_entropy_loss(logits, labels, smoothing, dim=-1):

    """
    :param logits: Predictions from model (before softmax) (B x C)
    :param labels: LongTensor of class indices (B,)
    :param smoothing: Float, how much label smoothing
    :param dim: Channel dimension
    :return:
    """

    # Convert labels to distributions
    labels = smooth_one_hot(true_labels=labels, smoothing=smoothing, classes=logits.size(dim))

    preds = logits.log_softmax(dim=dim)

    return torch.mean(torch.sum(-labels * preds, dim=dim))