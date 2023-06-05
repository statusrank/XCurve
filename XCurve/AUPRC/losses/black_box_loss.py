import torch
from collections import deque
from blackbox_backprop.ranking import TrueRanker, rank_normalised
from abc import ABCMeta, abstractmethod

from .base_loss import BaseLoss


class BlackBoxLoss(BaseLoss):
    """ Torch module for computing recall-based loss as in 'Blackbox differentiation of Ranking-based Metrics' """
    def __init__(self,
                 num_sample_per_id,
                 lambda_val,
                 margin,
                 interclass_coef,
                 batch_memory,
                 **kwargs
                ):
        """
        :param lambda_val:  hyperparameter of black-box backprop
        :param margin: margin to be enforced between positives and negatives (alpha in the paper)
        :param interclass_coef: coefficient for interclass loss (beta in paper)
        :param batch_memory: how many batches should be in memory
        """
        super().__init__()
        self.num_sample_per_id = num_sample_per_id
        self.batch_memory = batch_memory
        self.margin = margin
        self.lambda_val = lambda_val
        self.interclass_coef = interclass_coef

        self.storage = deque()
    
    def _check_input(self, targets):
        batch_size = targets.shape[0]
        targets = targets.view(
            batch_size // self.num_sample_per_id,
            self.num_sample_per_id
        )
        diff = targets - targets[:, 0].unsqueeze(1)
        assert diff.sum() == 0
        assert (targets == targets[0]).sum() == self.num_sample_per_id

    def raw_map_computation(self, scores, targets):
        """
                :param scores: [batch_size, num_classes] predicted relevance scores
                :param targets: [batch_size, num_classes] ground truth relevances
        """
        # Compute map
        HIGH_CONSTANT = 2.0
        epsilon = 1e-5
        transposed_scores = scores.transpose(0, 1)
        transposed_targets = targets.transpose(0, 1)
        deviations = torch.abs(torch.randn_like(transposed_targets)) * (transposed_targets - 0.5)

        transposed_scores = transposed_scores - self.margin * deviations
        ranks_of_positive = TrueRanker.apply(transposed_scores, self.lambda_val)
        scores_for_ranking_positives = -ranks_of_positive + HIGH_CONSTANT * transposed_targets
        ranks_within_positive = rank_normalised(scores_for_ranking_positives)
        ranks_within_positive.requires_grad = False
        assert torch.all(ranks_within_positive * transposed_targets < ranks_of_positive * transposed_targets + epsilon)

        sum_of_precisions_at_j_per_class = ((ranks_within_positive / ranks_of_positive) * transposed_targets).sum(dim=1)
        precisions_per_class = sum_of_precisions_at_j_per_class / (transposed_targets.sum(dim=1) + epsilon)

        present_class_mask = targets.sum(axis=0) != 0
        return 1.0 - precisions_per_class[present_class_mask].mean()


    def _forward(self, output, target):

        current_storage = list(self.storage)
        long_output = torch.cat([output] + [x[0] for x in current_storage], dim=0)
        long_target = torch.cat([target] + [x[1] for x in current_storage], dim=0)

        assert long_output.shape[0] == long_target.shape[0]  # even in multi-gpu setups
        cross_batch_loss = self.raw_map_computation(long_output, long_target)

        output_flat = output.reshape((-1, 1))
        target_flat = target.reshape((-1, 1))
        interclass_loss = self.raw_map_computation(output_flat, target_flat)

        while len(self.storage) >= self.batch_memory:
            self.storage.popleft()

        self.storage.append([output.detach(), target.detach()])

        loss = (1.0 - self.interclass_coef) * cross_batch_loss + self.interclass_coef * interclass_loss
        return loss
    
    def forward(self, samples):
        feats = samples['feat']
        targets = samples['target']
        batch_size = targets.shape[0]
        nc = batch_size // self.num_sample_per_id
        ns = self.num_sample_per_id
        d = feats.shape[-1]

        self._check_input(targets)

        mask = torch.block_diag(*([torch.ones(ns, ns)]*nc)).cuda()
        sim = torch.mm(feats, feats.t())

        return self._forward(sim, mask)
