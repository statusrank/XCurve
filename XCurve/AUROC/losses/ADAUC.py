import torch
import torch.nn as nn
from abc import abstractmethod
import numpy as np


class AUCLoss(nn.Module):
	def __init__(self, device=None, imratio=None,
				 a=None,
				 b=None,
				 alpha=None):
		super(AUCLoss, self).__init__()
		if not device:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = device
		
		self.p = imratio
		if a is not None:
			self.a = torch.tensor(a).float().to(self.device)
			self.a.requires_grad = True
		else:
			self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
		
		if b is not None:
			self.b = torch.tensor(b).float().to(self.device)
			self.b.requires_grad = True
		else:
			self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
		
		if alpha is not None:
			self.alpha = torch.tensor(alpha).float().to(self.device)
			self.alpha.requires_grad = True
		else:
			self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
	
	
	def forward(self, y_pred, y_true):
		if self.p is None:
			self.p = (y_true == 1).float().sum() / y_true.shape[0]
		
		y_pred = y_pred.reshape(-1, 1)
		y_true = y_true.reshape(-1, 1)
		loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
			   self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
			   2 * self.alpha * (self.p * (1 - self.p) +
								 torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (
											 1 == y_true).float()))) - \
			   self.p * (1 - self.p) * self.alpha ** 2
		return loss
	
	