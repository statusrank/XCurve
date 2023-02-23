import torch
from torch.optim import SGD


class ASGDA(SGD):
    def __init__(self, params=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, init_lr=0.01, hparams=None):
        super(ASGDA, self).__init__(params, momentum, dampening,
                                     weight_decay, nesterov)
        # self.device = torch.device('cuda:0')
        self.device = hparams['device']
        self.nu = torch.tensor(hparams['nu']).to(self.device)
        self.lam = torch.tensor(hparams['lam']).to(self.device)
        self.k = torch.tensor(hparams['k']).to(self.device)
        self.m = torch.tensor(hparams['m']).to(self.device)
        self.c1 = torch.tensor(hparams['c1']).to(self.device)
        self.c2 = torch.tensor(hparams['c2']).to(self.device)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old_param'] = torch.zeros(p.shape).to(self.device)
                state['old_grad'] = torch.zeros(p.shape).to(self.device)

    @torch.no_grad()
    def step(self, closure=None, pre=False, t=0):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
        """
        # t = torch.tensor(t).to(self.device)
        eta = self.k/torch.pow(self.m+t, 0.333)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            old_grad_list = []
            old_param_list = []
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    params_with_grad.append(p)
                    if pre:
                        state['old_grad'].zero_()
                        state['old_grad'].add_(p.grad, alpha=1)
                        if 'clip' in group.keys():
                            # if group['name'] == 'g':
                            #     group['clip'][0] = max(
                            #         self.param_groups[1]['params'][1] - 1, -self.param_groups[1]['params'][0])
                            #     d_p_list.append(torch.clip(
                            #         p + self.lam * state['old_param'], group['clip'][0], group['clip'][1]))
                            # else:
                            d_p_list.append(torch.clip(
                            p - self.nu * state['old_param'], group['clip'][0], group['clip'][1]))
                        else:
                            d_p_list.append(p - self.nu * state['old_param'])
                    old_grad_list.append(state['old_grad'])
                    old_param_list.append(state['old_param'])
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            if pre:
                sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    eta,
                    weight_decay=group['weight_decay'],
                    momentum=group['momentum'],
                    dampening=group['dampening'],
                    nesterov=group['nesterov'])
                # update momentum_buffers in state
                for p, momentum_buffer, o_p, o_g in zip(params_with_grad, momentum_buffer_list, old_param_list, old_grad_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
                    state['old_param'] = o_p
                    state['old_grad'] = o_g
            else:
                rho = self.c1 * torch.square(eta)
                xi = self.c2 * torch.square(eta)
                for p, momentum_buffer, o_p, o_g in zip(params_with_grad, momentum_buffer_list, old_param_list, old_grad_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
                    state['old_grad'] = o_g
                    if group['name'] == 'g':
                        all_grad = p.grad + (1 - xi) * \
                            (state['old_param'] - o_g)
                    else:
                        all_grad = p.grad + (1 - rho) * \
                            (state['old_param'] - o_g)
                    state['old_param'].zero_()
                    state['old_param'].add_(all_grad, alpha=1)

        return loss


def sgd(params,
        d_p_list,
        momentum_buffer_list,
        lr,
        weight_decay,
        momentum,
        dampening,
        nesterov):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        param.mul_(1-lr)
        param.add_(lr*d_p, alpha=1)
