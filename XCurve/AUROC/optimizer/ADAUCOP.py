import torch
from torch.optim.optimizer import Optimizer, required

class ours_opt(Optimizer):
	def __init__(self, params, a, b, alpha, lr=required, momentum=0,
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
		
		super(ours_opt, self).__init__(params, defaults)
	
	def __setstate__(self, state):
		super(ours_opt, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)
			
	def step(self, closure=None):
		"""Performs a single optimization step.
		
		"""
		loss = None
		if closure is not None:
			loss = closure()
		clip_value = 2.0
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
			# alpha.data = torch.clamp(alpha.data, -clip_value, clip_value)
			
			# min
			a.data.add_(-group['lr'], grad_a)
			b.data.add_(-group['lr'], grad_b)

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