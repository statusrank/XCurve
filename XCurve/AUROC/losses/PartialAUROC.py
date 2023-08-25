import torch
import torch.nn as nn

'''
This file includes the pytorch implementation of partial AUC optimization, including one-way and two-way partial AUC.

Note that, this code is based on our follwoing research:

"Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang. 
    When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC. ICML-2021. (Long talk)
", and

"Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang. 
    Optimizing Two-way Partial AUC with an End-to-end Framework. T-PAMI'2022.

"

'''

class BaseAUCLoss(nn.Module):
    def __init__(self, 
                gamma=1,
                E_k=0,
                weight_scheme='Poly',
                num_classes=2,
                reduction='mean',
                first_state_loss=None,
                eps=1e-6,
                **kwargs):
        super(BaseAUCLoss, self).__init__()

        '''
        args:
            gamma: safe margin in square loss (default = 1.0)
            E_k: warm-up epoch (default = 0), when epoch > E_k, the partial AUC will be conducted.
            weight_scheme: weight scheme, 'Poly' or 'Exp' 
            num_classes: only support binary classification
            reduction: loss aggregated manner (default = 'mean')
            eps: use to avoid zero gradient, users can ignore this

            first_state_loss: warmup loss (default = None), could be 'SquareAUCLoss()' 
                                or other pytorch supported loss such as CE.
        '''

        self.gamma = gamma
        self.reduction = 'mean'

        if num_classes != 2:
            raise ValueError("The current version only supports binary classification!")

        self.num_classes = num_classes
        self.reduction = reduction

        # adopt re_weight func after E_k epoch....
        self.E_k = E_k

        if weight_scheme is not None and weight_scheme not in ['Poly', 'Exp']:
            raise ValueError("weight_scheme should range in [Poly, Exp]")

        self.weight_scheme = weight_scheme

        self.eps = eps

        self.first_stage_loss = first_state_loss

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def _check_input(self, pred, target):
        assert pred.max() <= 1 and pred.min() >= 0
        assert target.min() >= 0
        assert pred.shape == target.shape

    def forward(self, pred, target, epoch=0, **kwargs):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        self._check_input(pred, target)

        if self.first_stage_loss is not None and epoch < self.E_k:
            return self.first_stage_loss(pred, target.float())

        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        n_plus, n_minus = len(pred_p), len(pred_n)

        weight = self.re_weight(pred_p, pred_n)
        if pred.is_cuda and not weight.is_cuda:
            weight = weight.cuda()

        pred_p = pred_p.unsqueeze(1).expand(n_plus, n_minus)
        pred_n = torch.reshape(pred_n, (1, n_minus))

        loss = weight * (self.gamma + pred_n - pred_p) ** 2
        
        return loss.mean() if self.reduction == 'mean' else loss.sum()

    def re_weight(self, pred_p, pred_n):
        return torch.ones(pred_p.shape[0], pred_n.shape[0])

# ICML 2021
class RelaxedPAUCLoss(BaseAUCLoss):
    def __init__(self, gamma=1.0, 
                 E_k=0,
                 weight_scheme='Poly',
                 num_classes = 2, 
                 reduction='mean', 
                 AUC_type='OP',
                 first_state_loss=None,
                 **kwargs):

        '''
        AUC_type = OP, TP (OPAUC or TPAUC optimization)
        '''
        
        super(RelaxedPAUCLoss, self).__init__(gamma, 
                                        E_k,
                                        weight_scheme,
                                        num_classes,
                                        reduction,
                                        first_state_loss,
                                        **kwargs)
        
        assert AUC_type in ['OP', 'TP'], 'AUC_type now only supports OP and TP achemes'
        
        self.AUC_type = AUC_type

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def re_weight(self, pred_p, pred_n):
        '''
        return:
            must be the (len(pred_p), len(pred_n)) matrix 
                    for element-wise multiplication
        '''

        if self.weight_scheme not in ['Poly', 'Exp']:
            raise ValueError('weight_scheme 4 TPAUC must be included in [Ploy, Exp], but weight_scheme %s' % self.weight_scheme)
        
        if self.weight_scheme == 'Poly':
            col_pred_p = torch.pow((1 - pred_p + self.eps), self.gamma)
            row_pred_n = torch.pow(pred_n + self.eps, self.gamma)
        else:
            col_pred_p = 1 - torch.exp(- self.gamma * (1 - pred_p))
            row_pred_n = 1 - torch.exp(- self.gamma * pred_n)

        if self.AUC_type == 'OP':
            col_pred_p = torch.ones_like(pred_p)

        return torch.mm(col_pred_p.unsqueeze_(1), row_pred_n.unsqueeze_(0))

# PAMI 2022
class InsRelaxedPAUCLoss(BaseAUCLoss):
    def __init__(self, 
                gamma=1,
                E_k=0,
                weight_scheme=None,
                num_classes=2,
                eps=1e-6,
                AUC_type='OP',
                first_state_loss=None,
                reg_a=0.0,
                reg_b=0.0,
                **kwargs):
        
        super(InsRelaxedPAUCLoss, self).__init__()

        '''
        args:
            reg_a and reg_b: weight of the strong convex constraint
            first_state_loss: warmup loss (default = None), could be 'SquareAUCLoss()' or other pytorch supported loss such as CE.
        '''

        self.a = torch.zeros(10, dtype=torch.float64, device="cuda", requires_grad=True)
        self.b = torch.zeros(8, dtype=torch.float64, device="cuda", requires_grad=True)

        assert self.a.requires_grad == True
        assert self.b.requires_grad == True

        self.gamma = gamma
        self.E_k = E_k

        if weight_scheme is not None and weight_scheme not in ['Poly', 'Exp']:
            raise ValueError
        self.weight_scheme = weight_scheme

        self.num_classes = num_classes

        self.AUC_type = AUC_type
        self.first_stage_loss = first_state_loss
        self.eps = eps

        self.reg_a = reg_a
        self.reg_b = reg_b

        if kwargs is not None:
            self.__dict__.update(kwargs)
        
    def function_w(self, x, y):

        assert x.shape == y.shape, "dim of x and y must be the same!"

        return 2 * x * y - torch.square(x)
    
    def function_v(self, b, a1, a2, e, f):
        return self.function_w(b, e + f) - self.function_w(a1, e) - self.function_w(a2, f)

    def forward(self, pred, target, epoch=0, **kwargs):

        if self.first_stage_loss is not None and epoch < self.E_k:
            return self.first_stage_loss(pred, target.float())
        
        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        
        if self.weight_scheme == 'Poly':
            v_plus = torch.pow(1 - pred_p + self.eps, self.gamma)
            v_minus = torch.pow(pred_n + self.eps, self.gamma)
        else:
            v_plus = 1 - torch.exp(- self.gamma * (1 - pred_p))
            v_minus = 1 - torch.exp(- self.gamma * pred_n)

        if self.AUC_type == 'OP':
            v_plus = torch.ones_like(pred_p)
        
        c_plus = v_plus.mean()
        c_minus = v_minus.mean()

        f_plus = (v_plus * pred_p).mean()
        f_minus = (v_minus * pred_n).mean()

        f_plus_sq = (v_plus * pred_p.square()).mean()
        f_minus_sq = (v_minus * pred_n.square()).mean()

        loss = 0.5 * self.function_v(self.b[0], self.a[0], self.a[1], c_plus, c_minus) - \
             self.function_v(self.a[2], self.b[1], self.b[2], c_minus, f_plus) + \
             self.function_v(self.b[3], self.a[3], self.a[4], c_plus, f_minus) + \
             0.5 * self.function_v(self.b[4], self.a[5], self.a[6], c_minus, f_plus_sq) + \
             0.5 * self.function_v(self.b[5], self.a[7], self.a[8], c_plus, f_minus_sq) - \
             self.function_v(self.a[9], self.b[6], self.b[7], f_plus, f_minus) 

        return loss.mean() + self.strong_convex_loss()
        
    def strong_convex_loss(self):
        return self.reg_a * self.a.square().sum() - self.reg_b * self.b.square().sum()
    

class UnbiasedPAUCLoss(nn.Module):
    def __init__(self, alpha, beta, device):
        super(UnbiasedPAUCLoss,self).__init__()
        # self.device = torch.device('cuda:0')
        # FPR range
        self.alpha = torch.tensor(alpha)
        self.na = alpha
        self.beta = torch.tensor(beta)
        # auxiliary variable
        self.kappa = torch.tensor(2)
        # optimzation variable
        self.a = torch.tensor(0.5).to(device)
        self.b = torch.tensor(0.5).to(device)
        self.g = torch.tensor(1.0).to(device)
        self.s_p = torch.tensor(0.5).to(device)
        self.s_n = torch.tensor(0.5).to(device)
        self.lam_b = torch.tensor(0.0).to(device)
        self.lam_a = torch.tensor(0.0).to(device)
        self.a.requires_grad = True
        self.b.requires_grad = True
        self.g.requires_grad = True
        self.s_p.requires_grad = True
        self.s_n.requires_grad = True
        self.lam_b.requires_grad = True
        self.lam_a.requires_grad = True
    def forward(self, pred, target):
        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        # one way partial auc
        if self.na == 1:
            max_val_n = torch.log(1+torch.exp(self.kappa*(torch.square(pred_n - self.b) + \
                          2 * (1 + self.g) * pred_n - self.s_n)))/self.kappa
            res = torch.mean(torch.square(pred_p - self.a) - \
                          2 * (1 + self.g) * pred_p) + (self.beta * self.s_n + \
                    torch.mean(max_val_n))/self.beta -\
                    1*self.g**2 - self.lam_b * (self.b-1-self.g)
        # two way partial auc
        else:
            max_val_p = torch.log(1+torch.exp(self.kappa*(torch.square(pred_p - self.a) - \
                          2 * (1 + self.g) * pred_p - self.s_p)))/self.kappa
            max_val_n = torch.log(1+torch.exp(self.kappa*(torch.square(pred_n - self.b) + \
                          2 * (1 + self.g) * pred_n - self.s_n)))/self.kappa
            res = self.s_p + torch.mean(max_val_p)/self.alpha + \
                  self.s_n + torch.mean(max_val_n)/self.beta -\
                    1*self.g**2 - self.lam_b * (self.b-1-self.g) + self.lam_a * (self.a+self.g)
            
        return res