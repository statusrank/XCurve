__copyright__ = 'Shilong Bao'
__email__ = 'baoshilong@iie.ac.cn'

import torch
from torch.optim.optimizer import Optimizer 

class SGD4MinMaxPAUC(Optimizer):

    def __init__(self, 
                params, 
                a = None, 
                b = None, 
                lr = 1e-2,
                momentum=0.,
                dampening=0.,
                nesterov=False,
                weight_decay = 0., 
                clip_value = 5.0,
                epoch_to_opt=10,
                **kwargs):
        
        assert a is not None, 'Found no variable a!'
        assert b is not None, 'Found no variable b!'

        '''
        The SGD optimization for Min-Max Partial AUC, including OPAUC and TPAUC. 

        args:
            params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

            lr (float): learning rate
            weight_decay (float): regularizer weights
            clip_value (float): to clip gradient
            epoch_to_opt: 
            a (tensor): reformulation weights a
            b (tensor): reformulation weights b
        
        The other args stay the same with SGD in pytorch
        '''

        if lr <= 0.0:
            raise ValueError('Invalid Learning rate {} - should be >= 0.'.format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0:
            raise ValueError('Invalid weight_decay {} - should be >= 0.'.format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        self.eopch_to_opt = epoch_to_opt
        self.a = a
        self.b = b 

        assert self.a.requires_grad == True
        assert self.b.requires_grad == True

        defaults = dict(lr=lr,
                        weight_decay=weight_decay,
                        a = self.a, 
                        b = self.b,
                        clip_value = clip_value,
                        momentum=momentum,
                        dampening=dampening,
                        nesterov=nesterov 
                        )
        
        super(SGD4MinMaxPAUC, self).__init__(params=params, 
                                        defaults=defaults)


    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This is will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
        
        if self.a.grad is not None and self.b.grad is not None:
            self.a.grad.zero_()
            self.b.grad.zero_()
        else: 
            self.a.grad = None
            self.b.grad = None

    def __setstate__(self, state):
        super(SGD4MinMaxPAUC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    @torch.no_grad()
    def step(self, epoch=0):
        """Performs a single optimization step.
        
        args:
            epoch (int): the current epoch of optimization
        """
        for group in self.param_groups:
            
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            lr = group['lr']

            clip_value = group['clip_value']    
            
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if d_p.is_sparse:
                    raise RuntimeError('does not support sparse gradients !!!')
                
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf 
                
                # clip the gradient
                d_p = torch.clamp(d_p, -clip_value, clip_value)
                p.data.add_(d_p, alpha=-lr)
            
            if epoch >= self.eopch_to_opt:
                a, b = group['a'], group['b']
                
                if a.grad is None or b.grad is None:
                    raise ValueError('the grad of a/b is None !')
                
                d_a, d_b = a.grad, b.grad

                if d_a.is_sparse or d_b.is_sparse: 
                    raise RuntimeError('does not support sparse gradients !!!')

                if momentum != 0:
                    param_state_a = self.state[a]
                    param_state_b = self.state[b]

                    if 'momentum_buffer' not in param_state_a:
                        buf_a = param_state_a['momentum_buffer'] = torch.clone(d_a).detach()
                        buf_b = param_state_b['momentum_buffer'] = torch.clone(d_b).detach()
                    else:
                        buf_a = param_state_a['momentum_buffer']
                        buf_a.mul_(momentum).add_(d_a, alpha=1 - dampening)

                        buf_b = param_state_b['momentum_buffer']
                        buf_b.mul_(momentum).add_(d_b, alpha=1 - dampening)
                    if nesterov:
                        d_a = d_a.add(buf_a, alpha = momentum)
                        d_b = d_b.add(buf_b, alpha = momentum)
                    else:
                        d_a = buf_a 
                        d_b = buf_b 

                # clip the gradient
                d_a = torch.clamp(d_a, -clip_value, clip_value)
                d_b = torch.clamp(d_b, -clip_value, clip_value)

                a.data.add_(d_a, alpha = -lr)
                b.data.add_(d_b, alpha = lr)