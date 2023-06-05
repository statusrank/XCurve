# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

###################### LIBRARIES #################################################
import torch
import warnings
warnings.filterwarnings("ignore")

from .base_loss import BaseLoss

"""================================================================================================="""

def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """

    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())


class SmoothAPLoss(BaseLoss):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (feats): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, num_sample_per_id, **kwargs):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        num_id : int
            the number of different classes that are represented in the batch
        """
        super(SmoothAPLoss, self).__init__()

        self.anneal = anneal
        self.num_sample_per_id = num_sample_per_id
    
    def check_input(self, targets):
        batch_size = targets.shape[0]
        targets = targets.view(
            batch_size // self.num_sample_per_id,
            self.num_sample_per_id
        )
        diff = targets - targets[:, 0].unsqueeze(1)
        assert diff.sum() == 0

    def forward(self, samples):
        """Forward pass for all input predictions: feats - (batch_size x feat_dims) """
        feats = samples['feat']
        targets = samples['target']
        ns = self.num_sample_per_id

        self.check_input(targets)
        batch_size = targets.shape[0]

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(batch_size)
        mask = mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(feats)

        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.cuda()
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = feats.view(batch_size // ns, ns, -1)
        pos_mask = 1.0 - torch.eye(ns)
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(batch_size // ns, ns, 1, 1)
        # compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, ns, 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask.cuda()
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # try:
        #     self.cnt += 1
        # except:
        #     self.cnt = 0

        # if self.cnt % 100 == 0:
        #     temp_rk_pn = sim_pos_rk.view(-1)
        #     for i in range(len(temp_rk_pn)):
        #         print(temp_rk_pn[i].item())


        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        for ind in range(batch_size // ns):
            pos_divide = torch.sum((sim_pos_rk[ind])
                 / (sim_all_rk[(ind * ns):((ind + 1) * ns), (ind * ns):((ind + 1) * ns)]))
            ap = ap + ((pos_divide / ns) / batch_size)

        return (1-ap)
