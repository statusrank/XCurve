import torch
import torch.nn as nn
from abc import abstractmethod

"""
Implementation of 
        "Zhiyong Yang, Qianqian Xu, Shilong Bao, Xiaochun Cao and Qingming Huang. 
            Learning with Multiclass AUC: Theory and Algorithms. T-PAMI, 2021."
"""

class AUCLoss(nn.Module):
    
    ''' 
        args:
            num_classes: number of classes (mush include params)

            gamma: safe margin in pairwise loss (default=1.0) 

            transform: manner to compute the multi-classes AUROC Metric, either 'ovo' or 'ova' (default as 'ovo' in our paper)
    
    '''
    def __init__(self,
                 num_classes,
                 gamma=1,
                 transform='ovo', *kwargs):
        super(AUCLoss, self).__init__()

        if transform != 'ovo' and transform != 'ova':
            raise Exception("type should be either ova or ovo")
        self.num_classes = num_classes
        self.gamma = gamma
        self.transform = transform

        if kwargs is not None:
            self.__dict__.update(kwargs)
    
    def _check_input(self, pred, target):
        assert pred.max() <= 1 and pred.min() >= 0
        assert target.min() >= 0
        assert pred.shape[0] == target.shape[0]

    def forward(self, pred, target, **kwargs):
        '''
        args:
            pred: score of samples residing in [0,1]. 
            For examples, with respect to binary classification tasks, pred = torch.Sigmoid(...)
            o.w. pred = torch.Softmax(...) 

            target: index of classes. In particular, w.r.t. binary classification tasks, we regard y=1 as pos. instances.

        '''
        self._check_input(pred, target)

        if self.num_classes == 2:
            Y = target.float()
            numPos = torch.sum(Y.eq(1))
            numNeg = torch.sum(Y.eq(0))
            Di = 1.0 / numPos / numNeg
            return self.calLossPerCLass(pred.squeeze(1), Y, Di, numPos)
        else:
            if self.transform == 'ovo':
                factor = self.num_classes * (self.num_classes - 1)
            else:
                factor = 1

            Y = torch.stack(
                [target.eq(i).float() for i in range(self.num_classes)],
                1).squeeze()

            N = Y.sum(0)  
            D = 1 / N[target.squeeze().long()]  

            loss = torch.Tensor([0.]).cuda()
            if self.transform == 'ova':
                ones_vec = torch.ones_like(D).cuda()
            
            for i in range(self.num_classes):
                if self.transform == 'ovo':
                    Di = D / N[i]
                else:
                    fac = torch.tensor([1.0]).cuda() / (N[i] * (N.sum() - N[i]))
                    Di = fac * ones_vec
                Yi, predi = Y[:, i], pred[:, i]
                loss += self.calLossPerCLass(predi, Yi, Di, N[i])

            return loss / factor

    def calLossPerCLass(self, predi, Yi, Di, Ni):
        
        return self.calLossPerCLassNaive(predi, Yi, Di, Ni)

    @abstractmethod
    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        pass


class SquareAUCLoss(AUCLoss):
    def __init__(self, num_classes, gamma=1, transform='ovo', **kwargs):
        super(SquareAUCLoss, self).__init__(num_classes, gamma, transform)

        # self.num_classes = num_classes
        # self.gamma = gamma

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        diff = predi - self.gamma * Yi
        nD = Di.mul(1 - Yi)
        fac = (self.num_classes -
               1) if self.transform == 'ovo' else torch.tensor(1.0).cuda()
        S = Ni * nD + (fac * Yi / Ni)
        diff = diff.reshape((-1, ))
        S = S.reshape((-1, ))
        A = diff.mul(S).dot(diff)
        nD= nD.reshape((-1, ))
        Yi= Yi.reshape((-1, ))
        B = diff.dot(nD) * Yi.dot(diff)
        return 0.5 * A - B

class HingeAUCLoss(AUCLoss):
    def __init__(self, num_classes, gamma=1, transform='ovo', **kwargs):
        super(HingeAUCLoss, self).__init__(num_classes, gamma, transform)

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        fac = 1 if self.transform == 'ova' else (self.num_classes - 1)
        delta1 = (fac / Ni) * Yi * predi
        delta2 = Di * (1 - Yi) * predi
        return fac * self.gamma - delta1.sum() + delta2.sum()


class ExpAUCLoss(AUCLoss):
    def __init__(self, num_classes, gamma=1, transform='ovo', **kwargs):
        super(ExpAUCLoss, self).__init__(num_classes,gamma, transform)
        
        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        C1 = Yi * torch.exp(-self.gamma * predi)
        C2 = (1 - Yi) * torch.exp(self.gamma * predi)
        C2 = Di * C2
        return C1.sum() * C2.sum()