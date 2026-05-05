import torch

from torch import nn as nn

from fluomapper.utils.nn import FixedMultiplier, ApplyOnImage
from fluomapper.utils.run import add_model_specific_args, add_prepend

from fluomapper.nn.simulation_mlp.mlp import _MLP
from fluomapper.data._base.base import Resampler

import copy

import collections
from collections import OrderedDict

from pvlib import spectrum, irradiance, atmosphere
import numpy as np


def get_irradiance_spectra(sza, aoi=None, out_wvls=None, pressure=101300, tau500=0.1, ozone=0.31, albedo=0.2,
                           water_vapor_content=0.5, resampler=None, device=None, resample_simulation=True):

    sza = sza * 360 / 2 / 3.1415

    # adapted from https://pvlib-python.readthedocs.io/en/stable/auto_examples/plot_spectrl2_fig51A.html
    if aoi is None:
        aoi = irradiance.aoi(0, 0, sza, 0)
    relative_airmass = atmosphere.get_relative_airmass(sza, model='kasten1966')

    spectra = spectrum.spectrl2(
        apparent_zenith=sza,
        aoi=aoi,
        surface_tilt=0,
        ground_albedo=albedo,
        surface_pressure=pressure,
        relative_airmass=relative_airmass,
        precipitable_water=water_vapor_content,
        ozone=ozone,
        aerosol_turbidity_500nm=tau500,
        dayofyear=170
    )

    wvls, spectra = spectra['wavelength'], (spectra['poa_global'] * 1000).transpose()

    if resampler is None and out_wvls is not None:
        if resample_simulation:
            resampler = Resampler(wvls, new_wvls=out_wvls, fwhm=0.33)

        else:
            resampler = Resampler(out_wvls, new_wvls=wvls, fwhm=10.0)

    if resample_simulation:
        spectra = torch.from_numpy(resampler.resample(spectra)).float()

    else:
        spectra = torch.from_numpy(spectra).float()

    if device is not None:
        spectra = spectra.to(device)

    return spectra, resampler


class ImageMean(nn.Module):
    def __init__(self, keep_channel=False, axis=None):
        super(ImageMean, self).__init__()
        self.keep_channel = keep_channel
        self.axis = [axis] if axis is not None else [2, 3]

    def forward(self, x):
        if not self.keep_channel:
            self.axis += [1]
        return x.mean(dim=self.axis)


class Unsqueeze(nn.Module):
    def forward(self, x):
        return x[..., None, None]


class WeightMask(nn.Module):
    def __init__(self, weight_model, dim_in, dims, *args, **kwargs):
        super(WeightMask, self).__init__()

        self.dim_in = dim_in
        self.model = weight_model

        if self.model == 'mlp':
            kwargs['in_bn'] = False
            self.model = ApplyOnImage(nn.Sequential(_MLP(dim_in=self.dim_in, dim_out=1, dims=dims, *args, **kwargs),
                                                    nn.BatchNorm1d(1),
                                                    nn.Sigmoid()))

        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)


class Param(nn.Module):
    def __init__(self, dim_out, init=1, *args, **kwargs):
        super(Param, self).__init__()
        self.param = nn.Parameter(torch.ones(dim_out) * init, requires_grad=True)

    def forward(self, x=None, *args, **kwargs):
        if x is not None:
            n_samples = x.shape[0]
            return self.param.unsqueeze(0).expand((n_samples, len(self.param)))
        
        else:
            return self.param

class AttentionReducer(nn.Module):
    def __init__(self, forward_model, weight_model, normed_weights=True, quantile=None, encoding_to_module=True,
                 restrain_inp_to_module=None, quantile_reduction=True, axis=None, *args, **kwargs):
        super(AttentionReducer, self).__init__()

        self.model = forward_model

        if weight_model is not None and weight_model != 'mean':
            self.weight_model = weight_model

        else:
            self.weight_model = None
            self.reducer = ImageMean(keep_channel=True, axis=axis)

        self.normed_weights = normed_weights
        self.quantile = quantile
        self.encoding_to_module = encoding_to_module
        self.restrain_inp_to_module = restrain_inp_to_module
        self.quantile_reduction = quantile_reduction

        self.axis=axis

    def forward(self, inputs, eps=1e-3): 
        encoded_inp, inp = inputs[0], inputs[1]

        if self.encoding_to_module:
            x = self.model(encoded_inp)

        else:
            if self.restrain_inp_to_module is not None:
                inp_ = inp[:, :self.restrain_inp_to_module]

            else:
                inp_ = inp
            
            x = self.model(inp_)

        if type(x) in (dict, OrderedDict):
            out = OrderedDict([(k, self.reduce(x[k], inp, eps))
                                for k in x.keys()])

        else:
            out = self.reduce(x, inp, eps)

        if len(inputs) == 2:
            return out

        else:
            return tuple([out] + list(inputs[2:]))

    def reduce(self, x, inp, eps=1e-3):
        if self.weight_model is not None:
            mask = self.weight_model(inp)

            if self.quantile is not None and self.quantile_reduction:
                with torch.no_grad():
                    qs = [torch.quantile(mask[i].flatten(), self.quantile) for i in range(x.shape[0])]

                inds = [torch.where(mask[i] >= qs[i]) for i in range(x.shape[0])]

                norms = [mask[i][:, inds[i][0], inds[i][1]].sum(-1) for i in range(mask.shape[0])]

                ret = torch.stack([(
                                           x[i][:, inds[i][0], inds[i][1]] *
                                           mask[i][:, inds[i][0], inds[i][1]] / norms[i]
                                   ).sum(-1)
                                   for i in range(x.shape[0])])
                
                return ret

            else:
                if self.normed_weights:
                    normed_mask = torch.einsum('bcij, bc -> bcij', mask, 1 / (eps + mask.sum(dim=(-2, -1))))

                else:
                    normed_mask = mask

                ret = (x * normed_mask)
                return ret.sum(dim=(-2, -1))

        else:
            return self.reducer(x)

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        parser.add_argument('--normed_weights', type=int, default=1)
        parser.add_argument('--quantile', type=float, default=None)
        parser.add_argument('--quantile_reduction', type=int, default=1)

        return parser


class ParamPredictor(nn.Module):
    def __init__(self, model, param_ranges, dim_param=1, with_bn=True):
        super(ParamPredictor, self).__init__()

        self.model = model
        self.dict_mode = False

        if type(param_ranges) is dict:
            self.dict_mode = True
            self.param_keys = list(param_ranges.keys())
            self.param_ranges = list(param_ranges.items())
            lo = torch.tensor([p[0] for k, p in self.param_ranges])
            hi = torch.tensor([p[1] for k, p in self.param_ranges])

            if type(dim_param) is int:
                dim_param = [dim_param] * len(self.param_ranges)

            self.out = [[nn.BatchNorm1d(dim),
                         nn.Sigmoid(),
                         FixedMultiplier(add=lo, mult=hi - lo)] 
                         for dim, param in zip(dim_param, self.param_keys)]

            if not with_bn:
                self.out = [o[1:] for o in self.out]

            self.out = [nn.Sequential(*o) for o in self.out]

            self.out = nn.ModuleDict(zip(self.param_keys, self.out)) 


        else:
            self.param_ranges = param_ranges
            self.out = [nn.BatchNorm1d(dim_param),
                        nn.Sigmoid(),
                        FixedMultiplier(add=self.param_ranges[0],
                                        mult=self.param_ranges[1] - self.param_ranges[0])]

            if not with_bn:
                self.out = self.out[1:]

            self.out = nn.Sequential(*self.out)

    def forward(self, x=None):
        x = self.model(x)
        if self.dict_mode:
            out = OrderedDict([(param, torch.atleast_2d(self.out[param](x))) for param in self.param_keys])
            return out

        else:
            out = self.out(x)
            return out


class StackDict(nn.Module):
    def __init__(self, ordered_keys):
        super(StackDict, self).__init__()
        self.ordered_keys = ordered_keys

    def forward(self, x):
        ret = []

        for key in self.ordered_keys:
            ret.append(x[key])

        stacked = torch.cat(ret, dim=1)
        return stacked


class SepConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, bias=False):
        super(SepConv, self).__init__()
        self.depthwise = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, bias=bias)
        self.pointwise = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SharedInLayer(nn.Module):
    def __init__(self, dim_in=None, in_wvls=None, *args, **kwargs):
        super(SharedInLayer, self).__init__()
        kwargs = self.split_kwargs(kwargs, 'shared')

        dims = kwargs['dims']
        del kwargs['dims']

        with_spatial_features = kwargs['with_spatial_features']
        dims_spatial = kwargs['dims_spatial']
        
        self.in_wvls = in_wvls
        if in_wvls is not None:
            if len(self.in_wvls.shape) >= 2:
                self.in_wvls = in_wvls[0]

            else:
                self.in_wvls = in_wvls
        
        if dim_in is None:
            self.in_wvl_len = self.in_wvls.shape[-1]
        else:
            self.in_wvl_len = dim_in
        
        self.model = ApplyOnImage(_MLP(dim_in=self.in_wvl_len, dims=dims, *args, **kwargs))

        self.with_spatial_features = with_spatial_features and dims_spatial is not None
        if self.with_spatial_features:
            self.spatial_model = nn.Sequential(*nn.ModuleList([nn.Sequential(SepConv(dim_in=kwargs['dim_out'], dim_out=dims_spatial[0],
                                                                                     kernel_size=3, padding=1),
                                                                             nn.BatchNorm2d(dims_spatial[0]),
                                                                             nn.ReLU())] +
                                                               [nn.Sequential(SepConv(dim_in=dim_in, dim_out=dim_out,
                                                                                      kernel_size=3, padding=1),
                                                                              nn.BatchNorm2d(dim_out),
                                                                              nn.ReLU())
                                                                for dim_in, dim_out in zip(dims_spatial[:-1], dims_spatial[1:])]
                                                               ))

            self.out_wvls = torch.arange(dims_spatial[-1])

        else:
            self.out_wvls = torch.arange(kwargs["dim_out"])

    def forward(self, x):
        x = self.model(x)

        if self.with_spatial_features:
            x = self.spatial_model(x)

        return x

    def split_kwargs(self, dic, key):
        new_dic = dict([(k.split('.')[1], val) for k, val in dic.items()
                        if len(k.split('.')) > 1 and k.split('.')[0] == key])

        ret_dic = copy.deepcopy(dic)
        ret_dic.update(new_dic)
        return ret_dic

    @classmethod
    def add_model_specific_args(cls, parser, prepend=None, **kwargs):
        prepend = add_prepend(prepend, 'shared') 
        parser = _MLP.add_model_specific_args(parser, prepend=prepend, **kwargs)
        parser_spec = dict([
            (f'{prepend}.dim_out', dict(type=int, default=1000)),
            (f'{prepend}.dims_spatial', dict(type=int, nargs='+', default=None)),
            (f'{prepend}.with_spatial_features', dict(type=int, default=0))
        ])
        parser = add_model_specific_args(parser, parser_spec=parser_spec, **kwargs)
        return parser

class Shifter(nn.Module):
    def __init__(self, min_wvl_shift, max_wvl_shift):
        super(Shifter, self).__init__()
        self.min = min_wvl_shift
        self.max = max_wvl_shift

    def forward(self, x, wvl_shift):
        x = torch.clamp(nn.Sigmoid(x) * 1.1, min=0, max=1)
        x = (self.max - self.min) * x + self.min
class Clamp(nn.Module):
    def __init__(self, min_, max_):
        super(Clamp, self).__init__()
        self.min_ = min_
        self.max_ = max_

    def forward(self, x):
        if self.min_ is None and self.max_ is None:
            return x

        return torch.clamp(x, min=self.min_, max=self.max_)


class ConstantMultiplier(nn.Module):
    def __init__(self, const, shape):
        super(ConstantMultiplier, self).__init__()
        self.const = nn.Parameter(torch.tensor([const]).reshape(shape), requires_grad=False)

    def forward(self, x):
        return self.const


class ParametrizedMatrixMult(nn.Module):
    def __init__(self, param_dim, matrix_dim):
        super(ParametrizedMatrixMult, self).__init__()
        self.matrix_dim = matrix_dim
        self.U = nn.Parameter(torch.randn((matrix_dim[0] * matrix_dim[1], param_dim))).requires_grad_(True)

    def forward(self, x):
        x, param = x

        m = torch.einsum('ij, bj -> bi', self.U, param)
        M = torch.stack([mm.reshape(self.matrix_dim) for mm in m])

        out = torch.einsum('bij, bj -> bi', M, x)
        return out


class WeightedSum(nn.Module):
    def __init__(self, vects, dim_in, dims, **kwargs):
        super(WeightedSum, self).__init__()

        self.vects = nn.Parameter(vects.clone(), requires_grad=False)
        self.nn = nn.Sequential(_MLP(dim_in=dim_in, dims=dims, 
                                     dim_out=self.vects.shape[0], **kwargs),
                                nn.BatchNorm1d(self.vects.shape[0]),
                                nn.Sigmoid())

    def forward(self, x):
        weights = self.nn(x)
        weights = torch.einsum('bi, b -> bi', weights, 1 / weights.sum(-1))
        return torch.einsum('ij, bi -> bj', self.vects, weights)
