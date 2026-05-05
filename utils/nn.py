import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn as nn

from fluomapper.utils.run import init_with_valid_kwargs
import collections


class _Seq(nn.Module):
    def __init__(self, dim_in, dim_out, op='lin', nonlin='relu', dropout=0, bn=True, rep=1, residual=False,
                 dim_cast_on_first=True, attention=False, attention_rep=1, skip_with_bn=False, rep_with_bn=False,
                 attention_bn=True, bn_after_relu=True, *args, **kwargs):
        super(_Seq, self).__init__()
        self.residual = residual
        self.attention = attention and attention_rep > 0
        self.bn_after_relu = bn_after_relu

        self.op_dict = dict(lin=nn.Linear, conv=nn.Conv1d,) # blin=BayesianLinear)
        self.nonlin_dict = dict(relu=nn.ReLU, sigmoid=nn.Sigmoid, lrelu=nn.LeakyReLU, none=nn.Identity, prelu=nn.PReLU)

        modules = self.get_modules(dim_in, dim_out, rep, op, dim_cast_on_first=dim_cast_on_first, dropout=False,
                                   batch_norm=rep_with_bn, **kwargs)
        self.pre_modules = nn.Sequential(*modules)

        after_modules = list()
        if not self.bn_after_relu and bn:
            after_modules.append(nn.BatchNorm1d(dim_out))

        if nonlin is not None and nonlin != 'none':
            after_modules.append(init_with_valid_kwargs(self.nonlin_dict[nonlin], **kwargs))

        if self.bn_after_relu and bn:
            after_modules.append(nn.BatchNorm1d(dim_out))

        if dropout:
            after_modules.append(nn.Dropout(dropout))

        if self.residual:
            if dim_in == dim_out:
                self.skip = nn.BatchNorm1d(dim_out) if skip_with_bn else nn.Identity()

            else:
                skip = self.get_modules(dim_in, dim_out, rep=1, op=op, dropout=False,
                                        **kwargs)
                self.skip = nn.Sequential(*skip, nn.BatchNorm1d(dim_out))

        if self.attention:
            self.attention_layer = nn.Sequential(*self.get_modules(dim_in, dim_in, rep=attention_rep, op=op, nonlin='lrelu',
                                                                   dim_cast_on_first=True, dropout=False, batch_norm=attention_bn),
                                                 nn.Sigmoid())

        if len(after_modules) > 0:
            self.after_modules = nn.Sequential(*after_modules)

        else:
            self.after_modules = None

    def get_modules(self, dim_in, dim_out, rep, op, nonlin='relu', dim_cast_on_first=True,
                    dropout=0, batch_norm=False, **kwargs):

        modules = list()
        for r in range(rep):

            if dim_cast_on_first:
                _dim_in = (1 - bool(r)) * dim_in + bool(r) * dim_out
                _dim_out = dim_out
            else:
                _dim_in = dim_in
                _dim_out = (r != rep-1) * dim_in + (r == rep-1) * dim_out

            modules.append(init_with_valid_kwargs(self.op_dict[op], _dim_in, _dim_out, **kwargs))

            if not self.bn_after_relu and batch_norm:
                modules.append(nn.BatchNorm1d(_dim_out))

            if dropout:
                modules.append(nn.Dropout(dropout))

            if self.bn_after_relu and batch_norm:
                modules.append(nn.BatchNorm1d(_dim_out))

            if r != rep - 1:
                modules.append(self.nonlin_dict[nonlin]())

        return modules

    def forward(self, inp, *args, **kwargs):
        if self.attention:
            inp = self.attention_layer(inp) * inp

        x = self.pre_modules(inp)

        if self.after_modules is not None:
            x = self.after_modules(x)

        if self.residual:
            x = F.relu(x + self.skip(inp))

        return x


def conv_nonlin_bn(*args, **kwargs):
    return _Seq(op='conv', *args, **kwargs)


def lin_nonlin_bn(bayesian=False, *args, **kwargs):
    if bayesian:
        op = 'blin'

    else:
        op = 'lin'

    return _Seq(op=op, *args, **kwargs)


class MultiSequential(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self._modules.values():
            input = module(input, *args, **kwargs)
        return input


class InputNorm(nn.Module):
    def __init__(self, dim_in, n_ids=None, windows=True):
        super(InputNorm, self).__init__()
    
        self.n_ids = n_ids

        if n_ids is None:
            self.bn = nn.BatchNorm1d(dim_in, affine=False)
            if windows:
                self.bn = ApplyOnImage(model=self.bn)

        else:
            self.bns = nn.ModuleList([ApplyOnImage(model=nn.BatchNorm1d(dim_in, affine=False))
                                      if windows else
                                      nn.BatchNorm1d(dim_in, affine=False)
                                      for _ in range(n_ids)])

    def forward(self, x, **kwargs):
        if self.n_ids is None:
            x = self.bn(x)

        else:
            ids = [int(p) for p in kwargs['source_id']]
            x = torch.cat([self.bns[self.data_source_id_dict[p]](x[[i]]) 
                             for i,p in enumeratei(ids)], dim=0)

        return x

    def set_eval(self):
        if self.n_ids is None:
            self.bn.eval()

        else:
            for bn in self.bns:
                bn.eval()


def conv_out_dim(dim_in, padding, kernel_size, dilation, stride=1, **kwargs):
    return (dim_in + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1


def pooling_out_dim(dim_in, padding=0, kernel_size=3, stride=None, **kwargs):
    if stride is None:
        stride = kernel_size

    return (dim_in - kernel_size + 2 * padding) // stride + 1


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma, eps=1e-2):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.delta = nn.Parameter(torch.tensor(float(max - min) / float(bins)), requires_grad=False)
        self.centers = nn.Parameter(float(min) + self.delta * (torch.arange(bins).float() + 0.5), requires_grad=False)

        self.eps = eps

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        sc = 2.5  # sqrt(2*pi)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * sc) * self.delta

        x = x.sum(-1) + self.eps
        x /= x.sum()
        return x


class GaussianKDE(nn.Module):
    def __init__(self, bins, sigma, eps=1e-2):
        """

        :param bins: (b, ) number of bins in each dimension
        :param sigma: (b, ) sigma in each dimension
        :param eps:
        """
        super(GaussianKDE, self).__init__()
        self.bins = bins
        self.sigma = nn.Parameter(torch.stack(sigma, dim=-1).requires_grad_(False))
        self.eps = eps

    def forward(self, x):
        """

        :param x: (n, b)
        :return:
        """
        maxs = [torch.max(x, dim=dim)[0] for dim in range(1, len(x.shape))]
        mins = [torch.min(x, dim=dim)[0] for dim in range(1, len(x.shape))]

        centers = torch.meshgrid(*[torch.linspace(min_, max_, bins_)
                                   for min_, max_, bins_ in zip(mins, maxs, bins)])

        sc = 2.5  # sqrt(2*pi)
        dists = torch.einsum('ij, xy -> ijxy', )
        x = torch.einsum('nb, b -> nb', dists, 1/self.sigma)
        sq_dist = x.pow(2).sum(-1)

        x = torch.exp(-0.5 * sq_dist) / (torch.sqrt(self.sigma.pow(2).sum()) * sc) * self.delta

        x = x.sum(-1) + self.eps
        x /= x.sum()
        return x


class FixedMultiplier(nn.Module):
    def __init__(self, mult, add=0.0):
        super(FixedMultiplier, self).__init__()
        self.mult = nn.Parameter(torch.tensor(mult), requires_grad=False)
        self.add = nn.Parameter(torch.tensor(add), requires_grad=False)

    def forward(self, x):
        return x * self.mult + self.add


class ScaledSigmoid(nn.Module):
    def __init__(self, lo=0, hi=1):
        super(ScaledSigmoid, self).__init__()

        self.lo = lo
        self.hi = hi

    def forward(self, x, *args, **kwargs):
        return self.lo + torch.sigmoid(x) * (self.hi - self.lo)


class ApplyOnImage(nn.Module):
    def __init__(self, model):
        super(ApplyOnImage, self).__init__()
        self.model = model

    @staticmethod
    def flatten(x):
        if x is None:
            return None

        return x.flatten(start_dim=2).permute(0, 2, 1).flatten(start_dim=0, end_dim=1).contiguous()

    @staticmethod
    def unflatten(x, batch_dim, spatial_dim):
        if x is None:
            return None

        channel_dim = x.shape[1]
        return x.reshape(batch_dim, spatial_dim ** 2, channel_dim).permute(0, 2, 1).reshape(batch_dim, channel_dim, spatial_dim, spatial_dim).contiguous()

    def forward(self, x=None, pass_kwargs_as_args=False, mask=None, **kwargs):
        # flatten kwargs if needed
        _kwargs = dict()
        for kwarg in kwargs.keys():
            if isinstance(kwargs[kwarg], torch.Tensor) and len(kwargs[kwarg].shape) > 0\
                    and np.all(np.array(kwargs[kwarg].shape)[[0, 2, 3]] == np.array(x.shape)[[0, 2, 3]]):
                _kwargs[kwarg] = self.flatten(kwargs[kwarg])

            else:
                _kwargs[kwarg] = kwargs[kwarg]

        # apply model on flattened x
        if not pass_kwargs_as_args:
            model_out = self.model(self.flatten(x), **_kwargs)

        else:
            model_out = self.model(self.flatten(x), _kwargs)

        if isinstance(model_out, collections.abc.Mapping) and x is not None:
            for k, v in model_out.items():
                model_out[k] = self.unflatten(v, x.shape[0], x.shape[-1])

            return model_out

        elif x is not None:
            return self.unflatten(model_out, x.shape[0], x.shape[-1])  # 0: batch, 1: wvl, 2 and 3: spatial

        else:
            return model_out


def differ(dat, axis=1):
        diff = torch.diff(dat, axis=axis)
        last_column = diff.select(axis, -1).unsqueeze(axis)
        return torch.cat((diff, last_column), axis=axis)

