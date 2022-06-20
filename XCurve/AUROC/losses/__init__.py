__copyright__ = 'Shilong Bao'
__email__ = 'baoshilong@iie.ac.cn'


import imp
from .AUROCLoss import SquareAUCLoss, ExpAUCLoss, HingeAUCLoss
from .PAUROCLoss import PAUCLoss, MinMaxPAUC


def get_losses(args):
    return eval(args.loss_type)(**(args.loss_params))
