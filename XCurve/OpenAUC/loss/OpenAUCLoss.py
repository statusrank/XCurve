import torch.nn as nn
import numpy as np
import torch

class OpenAUCLoss(nn.Module):
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