from .pair_loss import ContrastiveLoss, TripletLoss, MultiSimilarityLoss, VarianceLoss
from .smooth_ap_loss import SmoothAPLoss
from .list_wise_loss import ListWiseLoss, TieListWiseLoss
from .fast_ap_loss import FastAPLoss
from .black_box_loss import BlackBoxLoss
from .xbm_loss import XBMContrastiveLoss, XBMTripletLoss
from .auc_loss import AUCLogitLoss, AUCHuberLoss, AUCSigmoidLoss
from .soprc import SOPRC


loss_dict = {
    'black_box': BlackBoxLoss, # TODO
    'contrastive': ContrastiveLoss,
    'triplet': TripletLoss,
    'variance': VarianceLoss,
    'multi_similarity': MultiSimilarityLoss,
    'xbm_contrastive': XBMContrastiveLoss,
    'xbm_triplet': XBMTripletLoss,
    'smooth_ap': SmoothAPLoss,
    'list_wise': ListWiseLoss,
    'tie_list_wise': TieListWiseLoss, # BUG
    'fast_ap': FastAPLoss,
    'auc_lg': AUCLogitLoss,
    'auc_huber': AUCHuberLoss,
    'auc_sgm': AUCSigmoidLoss,
    'soprc': SOPRC
}


class LossWarpper(object):
    def __init__(self, args):
        self.args = args
        self.criterion = []
        for loss_name in args.keys():
            self.criterion.append(
                {
                    'loss_fn': loss_dict[loss_name](**args[loss_name]['loss_param']),
                    'loss_name': loss_name,
                    'loss_weight': args[loss_name].get('loss_weight', 1.0)
                }
            )

    def update_cnt_per_id(self, cnt_per_id):
        for i in range(len(self.criterion)):
            if isinstance(self.criterion[i]['loss_fn'], SOPRC):
                self.criterion[i]['loss_fn'].update_cnt_per_id(cnt_per_id)

    def __call__(self, samples):
        losses = {'loss': 0.0}
        for cri in self.criterion:
            loss = cri['loss_weight'] * cri['loss_fn'](samples)
            losses[cri['loss_name']] = loss.item()
            losses['loss'] += loss

        return losses

    def param_groups(self, lr=None, **kwargs):
        params = []
        for cri in self.criterion:
            params += cri['loss_fn'].param_groups(lr, **kwargs)
        return params
