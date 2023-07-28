from tokenize import group
import packaging
import torch
from torch.optim.optimizer import Optimizer, required

class AdvAUCOptimizer(Optimizer):
    def __init__(self, params, a, b, alpha, lr=required, momentum=0.9,
                 dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov,
                        a=a,
                        b=b,
                        alpha=alpha)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(AdvAUCOptimizer, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(AdvAUCOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    def step(self, closure=None):
        """Performs a single optimization step.
        
        """
        loss = None
        if closure is not None:
            loss = closure()
        clip_value = 1.0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            
            grad_alpha = torch.clamp(alpha.grad.data, -clip_value, clip_value)
            grad_a = torch.clamp(a.grad.data, -clip_value, clip_value)
            grad_b = torch.clamp(b.grad.data, -clip_value, clip_value)
            
            if weight_decay != 0:
                grad_alpha.add_(weight_decay, alpha.data)
                grad_a.add_(weight_decay, a.data)
                grad_a.add_(weight_decay, b.data)
            
            if momentum != 0:
                param_state = self.state[alpha]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(grad_alpha).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, grad_alpha)
                    
                if nesterov:
                    grad_alpha = grad_alpha.add(momentum, buf)
                else:
                    grad_alpha = buf
                
                param_state = self.state[a]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(grad_a).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, grad_a)
                
                if nesterov:
                    grad_a = grad_a.add(momentum, buf)
                else:
                    grad_a = buf
                
                param_state = self.state[b]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(grad_b).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, grad_b)
                
                if nesterov:
                    grad_b = grad_b.add(momentum, buf)
                else:
                    grad_b = buf
                
            # max
            alpha.data.add_(group['lr'], grad_alpha)
            # min
            a.data.add_(-group['lr'], grad_a)
            b.data.add_(-group['lr'], grad_b)

            # clip
            # a.data = torch.clamp(a.data, 0.0, 1.0)
            # b.data = torch.clamp(b.data, 0.0, 1.0)
            # alpha.data = torch.clamp(alpha.data, max(b.data-1, -a.data), 1)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = torch.clamp(p.grad.data , -clip_value, clip_value)
                # d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                p.data.add_(-group['lr'], d_p)
                
        return loss

class RegAdvAUCOptimizer(Optimizer):
    def __init__(self, params, a, b, alpha, lambda1, lambda2, c1=1, c2=1, gamma=1, lamda=1 ,lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = dict(lr=lr, c1=c1, c2=c2, gamma=gamma, lamda=lamda,
                        a=a,
                        b=b,
                        alpha=alpha,
                        lambda1=lambda1,
                        lambda2=lambda2)
        super(RegAdvAUCOptimizer, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(RegAdvAUCOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()
        clip_value = 1.0
        for group in self.param_groups:
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            lambda1 = group['lambda1']
            lambda2 = group['lambda2']
            # record the old alpha, lambda1, lambda2, a, b, w
            self.a_old, self.b_old, self.alpha_old = torch.clone(a.data).detach(), torch.clone(b.data).detach(), torch.clone(alpha.data).detach()
            self.lambda1_old, self.lambda2_old = torch.clone(lambda1.data).detach(), torch.clone(lambda2.data).detach()
            # get the learning rate
            # c1 = group['c1']
            # c2 = group['c2']
            lr = group['lr']
            gamma = group['gamma']
            lamda = group['lamda']
            
            # alpha
            # grad_alpha = alpha.grad.data
            param_state = self.state[alpha]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = 0
            else:
                buf = param_state['momentum_buffer']
            # what is the gamma and lambda in line 3 and line 5? 
            alpha.data.add_(lamda, buf)
            alpha.data = self.alpha_old + lr * (alpha.data - self.alpha_old)
            alpha.data = torch.clamp(alpha.data, -1.0, 1.0)
            # min
            # a
            # grad_a = a.grad.data
            param_state = self.state[a]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = 0
            else:
                buf = param_state['momentum_buffer']
            a.data.add_(-gamma, buf)
            a.data = self.a_old + lr * (a.data - self.a_old)
            a.data = torch.clamp(a.data, 0, 1.0)
            # b
            # grad_b = b.grad.data
            param_state = self.state[b]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = 0
            else:
                buf = param_state['momentum_buffer']
            b.data.add_(-gamma, buf)
            b.data = self.b_old + lr * (b.data - self.b_old)
            b.data = torch.clamp(b.data, 0, 1.0)
            # lambda1
            # grad_lambda1 = lambda1.grad.data
            param_state = self.state[lambda1]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = 0
            else:
                buf = param_state['momentum_buffer']
            lambda1.data.add_(-gamma, buf)
            # if lambda1.data<0:
            #     lambda1.data = lambda1.data-lambda1.data
            lambda1.data = torch.clamp(lambda1.data, 0.0, 5.0)
            lambda1.data = self.lambda1_old + lr * (lambda1.data - self.lambda1_old)
            
            # lambda2
            # grad_lambda2 = lambda2.grad.data
            param_state = self.state[lambda2]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = 0
            else:
                buf = param_state['momentum_buffer']
            lambda2.data.add_(-gamma, buf)
            # lambda2 = torch.clamp(b.data, 0, 1.0)
            # if lambda2.data<0:
            #     lambda1.data = lambda1.data-lambda1.data
            lambda1.data = torch.clamp(lambda1.data, 0.0, 5.0)
            lambda2.data = self.lambda2_old + lr * (lambda2.data - self.lambda2_old)
            
            # params
            # self.p_old = group['params']
            for p in group['params']:
                if p.grad is None:
                    continue
                # d_p = torch.clamp(p.grad.data , -clip_value, clip_value)
                # d_p = p.grad.data
                p_old = torch.clone(p.data).detach()
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = 0
                else:
                    buf = param_state['momentum_buffer']
                
                p.data.add_(-gamma, buf)
                p.data = p_old + lr * (p.data - p_old)
                
        return loss
    
    # before step, we need to do; which is use grad of z_t from disk
    # opt.zero_grad()
    # loss.backward()
    # opt.record_grad()
    def record_grad(self):
        for group in self.param_groups:
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            lambda1 = group['lambda1']
            lambda2 = group['lambda2']
            
            param_state = self.state[a]
            param_state['grad_buffer'] = torch.clone(a.grad.data).detach()
            param_state = self.state[b]
            param_state['grad_buffer'] = torch.clone(b.grad.data).detach()
            param_state = self.state[alpha]
            param_state['grad_buffer'] = torch.clone(alpha.grad.data).detach()
            param_state = self.state[lambda1]
            param_state['grad_buffer'] = torch.clone(lambda1.grad.data).detach()
            param_state = self.state[lambda2]
            param_state['grad_buffer'] = torch.clone(lambda2.grad.data).detach()
            
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state['grad_buffer'] = torch.clone(p.grad.data).detach()
    
    # after step()
    # opt.zero_grad()
    # loss.backward()
    # opt.updata_momentum()
    def updata_momentum(self):
        for group in self.param_groups:
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            lambda1 = group['lambda1']
            lambda2 = group['lambda2']
            
            c1 = group['c1']
            c2 = group['c2']
            lr = group['lr']
            beta1 = 1 - c1 * lr * lr
            beta2 = 1 - c2 * lr * lr
            
            param_state = self.state[a]
            param_state['momentum_buffer'] = beta1 * (param_state['momentum_buffer'] - param_state['grad_buffer']) + torch.clone(a.grad.data).detach()
            
            param_state = self.state[b]
            param_state['momentum_buffer'] = beta1 * (param_state['momentum_buffer'] - param_state['grad_buffer']) + torch.clone(b.grad.data).detach()
            
            param_state = self.state[lambda1]
            param_state['momentum_buffer'] = beta1 * (param_state['momentum_buffer'] - param_state['grad_buffer']) + torch.clone(lambda1.grad.data).detach()
            
            param_state = self.state[lambda2]
            param_state['momentum_buffer'] = beta1 * (param_state['momentum_buffer'] - param_state['grad_buffer']) + torch.clone(lambda2.grad.data).detach()
            
            param_state = self.state[alpha]
            param_state['momentum_buffer'] = beta2 * (param_state['momentum_buffer'] - param_state['grad_buffer']) + torch.clone(alpha.grad.data).detach()
            
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state['momentum_buffer'] = beta1 * (param_state['momentum_buffer'] - param_state['grad_buffer']) + torch.clone(p.grad.data).detach()
