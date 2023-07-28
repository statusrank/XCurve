from .pair_loss import VarianceLoss
from .soprc import SOPRC


loss_dict = {
    'variance': VarianceLoss,
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

    def __call__(self, feats, targets):
        loss = 0
        for cri in self.criterion:
            loss += cri['loss_weight'] * cri['loss_fn'](feats, targets)

        return loss

    def param_groups(self, lr=None, **kwargs):
        params = []
        for cri in self.criterion:
            params += cri['loss_fn'].param_groups(lr, **kwargs)
        return params
