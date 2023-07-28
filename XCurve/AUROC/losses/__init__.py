__copyright__ = 'XCurve'
__email__ = 'baoshilong@iie.ac.cn'


from .StandardAUROC import SquareAUCLoss, ExpAUCLoss, HingeAUCLoss
from .PartialAUROC import RelaxedPAUCLoss, InsRelaxedPAUCLoss, UnbiasedPAUCLoss
from .AdversarialAUROC import AdvAUROCLoss, RegAdvAUROCLoss, PGDAdversary
__all__ = ['SquareAUCLoss', 'ExpAUCLoss', 'HingeAUCLoss',
            'RelaxedPAUCLoss', 'InsRelaxedPAUCLoss', 'UnbiasedPAUCLoss',
            'AdvAUROCLoss', 'RegAdvAUROCLoss', 'PGDAdversary']
