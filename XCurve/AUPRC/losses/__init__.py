from .loss_warpper import LossWarpper

""" SOPRC

Paper:
`Exploring the Algorithm-Dependent Generalization of AUPRC Optimization with List Stability` - https://proceedings.neurips.cc/paper_files/paper/2022/file/b5dc49f44db2fadc5c4d717c57f4a424-Paper-Conference.pdf
@article{wen2022exploring,
  title={Exploring the Algorithm-Dependent Generalization of AUPRC Optimization with List Stability},
  author={Wen, Peisong and Xu, Qianqian and Yang, Zhiyong and He, Yuan and Huang, Qingming},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={28335--28349},
  year={2022}
}

Original code and weights from: https://github.com/KID-7391/SOPRC
Copyright (c) Inst. of Computing Tech., CAS.
All rights reserved.
This source code is licensed under the MIT license

"""

def ListStableAUPRC(
        tau1=0.1, 
        tau2=0.001,
        beta=0.001,
        prior_mul=0.1,
        num_sample_per_id=4,
        var_reg_weight_pos=5,
        var_reg_weight_neg=1
    ):

    """
    Args:
        tau1 (float): Control the surrogate loss of pos-neg pairs. See \tau_1 in Eq.(7).
        tau2 (float): Control the surrogate loss of pos-pos pairs. See \tau_2 in Eq.(7).
        beta (float): Control the exponential moving average. See \beta in Eq.(10).
        prior_mul (float): Imbalance ratio of the id with most positive examples.
        num_sample_per_id (int): Number of examples for each id.
        var_reg_weight_pos (float): Weight of the variance regular term w.r.t. positive examples.
        var_reg_weight_neg (float): Weight of the variance regular term w.r.t. negative examples.
    """

    args = {
        'soprc': {
            'loss_weight': 1,
            'loss_param': {
                'tau': [tau1, tau2],
                'beta': beta,
                'prior_mul': prior_mul,
                'num_sample_per_id': num_sample_per_id
            }
        },
        'variance': {
            'loss_weight': 1,
            'loss_param': {
                'thres': [var_reg_weight_pos, var_reg_weight_neg],
                'num_sample_per_id': num_sample_per_id
            }
        }
    }
    return LossWarpper(args)

__all__ = [ListStableAUPRC]
