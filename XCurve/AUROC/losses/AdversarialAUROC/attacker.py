import torch

lower_limit, upper_limit = 0.0, 1.0

def normalize(X):
    # global mu, std
    # return (X - mu)/std
    return X

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def PGDAdversary(model, X, y, criterion, epsilon=8.0/255, 
		        alpha=2.0/255, 
		        attack_iters=10, 
		        restarts=1, 
		        norm='linf'): 
	# model.eval()
	max_loss = torch.zeros(y.shape[0]).cuda() 
	max_delta = torch.zeros_like(X).cuda() 
	for _ in range(restarts): 
		delta = torch.zeros_like(X) 
		delta.uniform_(-epsilon, epsilon)
		delta = clamp(delta, lower_limit - X, upper_limit - X)
		delta.requires_grad = True
        
		for _ in range(attack_iters):
			output = model(normalize(X + delta)).view_as(y)
				
			index = slice(None, None, None)
				
			loss = criterion(output, y)
			loss.backward()
			grad = delta.grad.detach()
				
			d = delta[index, :, :, :]
			g = grad[index, :, :, :]
			x = X[index, :, :, :]

			if norm == 'linf':
				d = torch.clamp(d + alpha * torch.sign(g), 
									min=-epsilon, max=epsilon)
			else: 
				g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
				scaled_g = g/(g_norm + 1e-10)
				d = (d + scaled_g * alpha).view(d.size(0),
													-1).renorm(p=2,
							     								dim=0,
																maxnorm=epsilon).view_as(d)
			d = clamp(d, lower_limit - x, upper_limit - x)
			delta.data[index, :, :, :] = d
			delta.grad.zero_()
			all_loss = criterion(model(normalize(X + delta)).view_as(y), y)
				
		max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
		max_loss = torch.max(max_loss, all_loss)
	
	# model.train()
	return max_delta