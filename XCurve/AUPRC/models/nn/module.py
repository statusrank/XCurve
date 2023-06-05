# from .conv import *
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy
import math
from torch.nn.init import kaiming_normal_, kaiming_uniform_

class NoneLayer(nn.Module):
    def __init__(self):
        super(NoneLayer, self).__init__()
    
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
        assert len(self.shape) == 2

    def forward(self, x):
        return x.view(x.shape[0], -1, self.shape[0], self.shape[1])
    
    def inverse(self, x):
        return x.view(x.shape[0], -1)

class NoiseLayer(nn.Module):
    def __init__(self, num_feature):
        super(NoiseLayer, self).__init__()
        self.noise_weight = nn.Parameter(torch.zeros(1, num_feature, 1, 1))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = x + self.noise_weight * noise
        return x

class AdaIn(nn.Module):
    def __init__(self, n_latent, num_feature):
        super(AdaIn, self).__init__()
        self.norm = nn.InstanceNorm2d(num_feature)
        self.affine = LinearModule(n_latent, num_feature*2, act='LeakyReLU', bn=None)

    def forward(self, x, latent):

        x = self.norm(x)
        a = self.affine(latent).view(latent.shape[0], -1, 1, 1)
        s, t = a[:, :x.shape[1]], a[:, x.shape[1]:]
        return (s + 1.0) * x + t

# extending Conv2D and Deconv2D layers for equalized learning rate logic
class _equalized_conv2d(torch.nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super(_equalized_conv2d, self).__init__()

        # define the weight and bias if to be used
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv2d

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)
    
    def inverse(self, x):
        from torch.nn.functional import conv2d
        
        w_inv = torch.inverse(self.weight.squeeze()).view(self.weight.shape[0], self.weight.shape[1], 1, 1)
        if self.use_bias:
            x -= self.bias.view(1, -1, 1, 1)
        return conv2d(input=x,
                      weight=w_inv / self.scale,  # scale the weight on runtime
                      bias=None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))

class _equalized_linear(torch.nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param c_in: number of input channels
            :param c_out: number of output channels
            :param bias: whether to use bias with the linear layer
    """

    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        from numpy import sqrt

        super(_equalized_linear, self).__init__()

        self.weight = torch.nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_out, c_in)
        ))

        self.use_bias = bias

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import linear
        return linear(x, self.weight * self.scale,
                      self.bias if self.use_bias else None)


# ----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ----------------------------------------------------------------------------
class PixelwiseNorm(torch.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

class ConvModule(nn.Module):
    def __init__(self, 
            nin,
            nout,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            act='ReLU',
            bn='BatchNorm',
            equaliaeed=True,
            gdrop_param=None,
            input_size=None,
            inversable=False,
            noise=False,
            **kwargs
        ):
        super(ConvModule, self).__init__()

        self.inversable = inversable
        if inversable:
            assert nin == nout
            if equaliaeed:
                self.layers = _equalized_conv2d(nin, nout, 1, bias=bias)
            else:
                self.layers = nn.Conv2d(nin, nout, 1, bias=bias)
            return
        

        layers = []
        if gdrop_param:
            layers.append(GDropLayer(**gdrop_param))

        if equaliaeed:
            layers.append(_equalized_conv2d(nin, nout, kernel_size, stride, padding, bias))
        else:
            layers.append(nn.Conv2d(nin, nout, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))

        if noise:
            layers.append(NoiseLayer(nout))

        if act == 'ReLU':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'LeakyReLU':
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        if bn == 'BatchNorm':
            layers.append(nn.BatchNorm2d(nout))
        elif bn == 'PixelNorm':
            layers.append(PixelwiseNorm())
        elif bn == 'LayerNorm':
            layers.append(nn.LayerNorm(nout * input_size[0] * input_size[1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def inverse(self, x):
        assert self.inversable
        if isinstance(self.layers, _equalized_conv2d):
            return self.layers.inverse(x)
        else:
            raise NotImplementedError()

class Conv1dModule(nn.Module):
    def __init__(self, 
            nin,
            nout,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            act='ReLU',
            bn='BatchNorm',
            equaliaeed=False,
            gdrop_param=None,
            input_size=None,
            noise=False,
            **kwargs
        ):
        super(Conv1dModule, self).__init__()

        layers = []
        if equaliaeed:
            assert False
        else:
            layers.append(nn.Conv1d(nin, nout, kernel_size, stride=stride, padding=padding, bias=bias))

        if act == 'ReLU':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'LeakyReLU':
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        if bn == 'BatchNorm':
            layers.append(nn.BatchNorm1d(nout))
        elif bn == 'PixelNorm':
            assert False
        elif bn == 'LayerNorm':
            assert False

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def inverse(self, x):
        assert self.inversable
        if isinstance(self.layers, _equalized_conv2d):
            return self.layers.inverse(x)
        else:
            raise NotImplementedError()

class LinearModule(nn.Module):
    def __init__(self, 
            nin,
            nout,
            bias=True,
            reshape=None,
            act='ReLU',
            bn='BatchNorm',
            equaliaeed=False,
            bn_first=True,
        ):
        super(LinearModule, self).__init__()
        layers = []
        if equaliaeed:
            layers.append(_equalized_linear(nin, nout, bias))
        else:
            layers.append(nn.Linear(nin, nout, bias=bias))
            kaiming_uniform_(layers[-1].weight, a=math.sqrt(5))
            if layers[-1].bias is not None:
                layers[-1].bias.data.zero_()

        if reshape is not None:
            layers.append(Reshape(reshape))

        if bn_first:
            if bn == 'BatchNorm':
                layers.append(nn.BatchNorm1d(nout))
                nn.init.uniform_(layers[-1]._parameters['weight'], 0, 1)
            elif bn == 'PixelNorm':
                layers.append(PixelwiseNorm())

            if act == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif act == 'LeakyReLU':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            if act == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif act == 'LeakyReLU':
                layers.append(nn.LeakyReLU(0.2, inplace=True))

            if bn == 'BatchNorm':
                layers.append(nn.BatchNorm1d(nout))
                nn.init.uniform_(layers[-1]._parameters['weight'], 0, 1)
            elif bn == 'PixelNorm':
                layers.append(PixelwiseNorm())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MultiLayerPerceptron(nn.Module):
    def __init__(self,
            nin,
            nmid,
            nout,
            nlayers,
            **kwargs
        ):
        super(MultiLayerPerceptron, self).__init__()
        assert nlayers > 1
        layers = []
        layers.append(
            LinearModule(nin, nmid, **kwargs)
        )
        for i in range(nlayers - 2):
            layers.append(
                LinearModule(nmid, nmid, **kwargs)
            )
        kwargs['act'] = None
        kwargs['bn'] = None
        layers.append(
            LinearModule(nmid, nout, **kwargs)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def param_groups(self, lr=None):
        params = list(filter(lambda x:x.requires_grad, self.parameters()))
        if len(params):
            if lr is not None:
                return [{'params': params, 'lr': lr}]
            else:
                return [{'params': params}]
        else:
            return []

# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class MinibatchStddev(nn.Module):
    def __init__(self, averaging='all'):
        super(MinibatchStddev, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2,3], keepdim=True)             # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0,2,3], keepdim=True)                   # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:                                                           # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)

class GDropLayer(nn.Module):
    """
    # Generalized dropout layer. Supports arbitrary subsets of axes and different
    # modes. Mainly used to inject multiplicative Gaussian noise in the network.
    """
    def __init__(self, mode='mul', strength=0.2, axes=(0,1), normalize=False):
        super(GDropLayer, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = float(self.strength * x.size(1) ** 0.5)
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        rnd = torch.from_numpy(rnd).type(x.data.type())
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str

class NormalizeConv(nn.Module):
    def __init__(self,in_channel,normal_mean=[103.530,116.280,123.675],normal_std=[57.375,57.120,58.395],gray_mode=False,bgr_mode=True):
        super().__init__()
        if gray_mode:
            if bgr_mode:
                normal_mean = [(normal_mean[2]*299 + normal_mean[1]*587 + normal_mean[0]*114 + 500) / 1000]
            else:
                normal_mean = [(normal_mean[0]*299 + normal_mean[1]*587 + normal_mean[2]*114 + 500) / 1000]
        else:
            pass
        normal_mean = [-i/std for i,std in zip(normal_mean,normal_std)]
        normal_std = [1/std for std in normal_std]
        weight = np.array([[[normal_std]]])
        bias = np.array(normal_mean)
        weight = weight.transpose((3,0,1,2))
        self.conv = nn.Conv2d(in_channel,in_channel,1,stride=1,padding=0,groups=in_channel,bias=True)
        for i in self.conv.parameters():
            i.requires_grad = False
        self.conv.weight = nn.Parameter(torch.Tensor(weight),requires_grad=False)
        self.conv.bias = nn.Parameter(torch.Tensor(bias),requires_grad=False)

    def forward(self,x):
        out = self.conv(x)
        return out
