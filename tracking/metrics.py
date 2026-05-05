from fluomapper.utils.run import init_with_valid_kwargs, make_parser_soft, update_namespace, add_model_specific_args, \
    eval_with_valid_kwargs
from fluomapper.utils.data import search_spectral_window, permute_channels, select

import pytorch_lightning as pl
import torchmetrics
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError
import torch
import torch.nn as nn
from types import FunctionType
import inspect
import numpy as np

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import re, copy, pandas as pd, ast
from itertools import chain


def mean(x):
    return x.mean()


def median(x):
    return torch.median(x)


class EvidentialLogger(torchmetrics.Metric):
    DO_FLATTEN = False
    EXPECTED_DIM = 4

    def __init__(self):
        super(EvidentialLogger, self).__init__()

        self.add_state('mse', default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('mean_epistemic_variance', default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('mean_aleatoric_variance', default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('mean_beta', default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('mean_alpha', default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('mean_nu', default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('counter', default=torch.tensor(0), dist_reduce_fx='sum')

    def _split_preds(self, preds):
        gamma = preds[:, 0]
        alpha = preds[:, 1]
        beta = preds[:, 2]
        nu = preds[:, 3]

        machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(gamma.device)
        alpha = torch.max(machine_epsilon + 1, alpha)
        beta = torch.max(machine_epsilon, beta)
        nu = torch.max(machine_epsilon, nu)

        return gamma, alpha, beta, nu

    def update(self, preds, target):
        preds = preds.detach()
        target = target.detach()

        gamma, alpha, beta, nu = self._split_preds(preds)

        self.counter += preds.shape[0]
        self.mse += ((gamma - target.squeeze()) ** 2).sum()
        self.mean_epistemic_variance += torch.clamp(beta / (alpha - 1) / nu, max=5).sum()
        self.mean_aleatoric_variance += torch.clamp(beta / (alpha - 1), max=5).sum()

        self.mean_epistemic_variance += torch.clamp(beta / (alpha - 1) / nu, max=5).sum()
        self.mean_aleatoric_variance += torch.clamp(beta / (alpha - 1), max=5).sum()

        self.mean_alpha += alpha.sum()
        self.mean_beta += beta.sum()
        self.mean_nu += nu.sum()

    def compute(self):
        return [('mse', self.mse / self.counter),
                ('mean_epistemic_variance', self.mean_epistemic_variance / self.counter),
                ('mean_aleatoric_variance', self.mean_aleatoric_variance / self.counter),
                ('mean_alpha', self.mean_alpha / self.counter),
                ('mean_beta', self.mean_beta / self.counter),
                ('mean_nu', self.mean_nu / self.counter)]


class StratifiedValidation(torchmetrics.Metric):

    def __init__(self, stratify_along_vars, stratify_bins, stratify_lo, stratify_hi, out_wvls, in_wvls,
                 plot=False, reducers=None):
        super(StratifiedValidation, self).__init__()
        self.agg_funcs = dict(mean=('mean', self.agg_for_mean),
                              var=('var', self.agg_for_var))

        self.vars = [dict(list(zip(('name', 'bins', 'lo', 'hi'), tup)) + [('state', ["mean", "var"])])
                     for tup in zip(stratify_along_vars, stratify_bins, stratify_lo, stratify_hi)]

        self.vars = self._check_vars(self.vars)

        self.edges = dict([])
        self.bins = dict([])

        self.plot = plot
        self.reducers = reducers

        self.out_wvls = out_wvls
        self.in_wvls = in_wvls

        self.device = self.out_wvls.device

        self._init()

    def _check_vars(self, vars):
        for var in self.vars:
            assert getattr(self, var['name'], None) is not None

            if not 'state' in var:
                var['state'] = 'mean'

            if type(var['state']) is str:
                var['state'] = [var['state']]

        return vars

    def _get_state_name(self, var, state):
        return '%s_%s' % (var['name'], state)

    def _get_counter_name(self, var):
        return '%s_counter' % var['name']

    def _create_bins(self, var):
        edges = np.arange(var['lo'], var['hi'], var['bins'])
        bins = ['%.2f - %.2f' % (edge_lo, edge_hi) for edge_lo, edge_hi in zip(edges[:-1], edges[1:])]

        self.edges[var['name']] = edges
        self.bins[var['name']] = bins

        return edges, bins

    def _init(self):
        for var in self.vars:
            edges, bins = self._create_bins(var)

            for state in var['state']:
                self.add_state(self._get_state_name(var, state), default=torch.zeros(len(bins), dtype=torch.float),
                               dist_reduce_fx='sum')

            self.add_state(self._get_counter_name(var), default=torch.zeros(len(bins), dtype=torch.int),
                           dist_reduce_fx='sum')

            var.update(dict(labels=bins))

    def update(self, preds, target, inp):
        for var in self.vars:
            inp, y, ypred = inp.cpu().numpy(), target.cpu().numpy(), preds.cpu().numpy()

            feature = getattr(self, var['name'])(x=inp, y=y, ypred=ypred)

            df = pd.DataFrame(dict(err=np.abs(y - ypred), feature=feature))

            binned_feature = pd.cut(df.feature, bins=self.edges[var['name']])
            df = df.assign(binned_feature=binned_feature.values).groupby('binned_feature')

            agg_funcs = [self.agg_funcs[state] for state in var['state']] + ['count']
            df = df.err.agg(agg_funcs)

            # replace nan with 0 for missing categories
            df = df.fillna(0)

            for state in var['state']:
                device = self.__dict__[self._get_state_name(var, state)].device
                self.__dict__[self._get_state_name(var, state)] += torch.from_numpy(df.loc[:, state].values).to(device)

            self.__dict__[self._get_counter_name(var)] += torch.from_numpy(df.loc[:, 'count'].values).to(device)

    def agg_for_var(self, series):
        return (series ** 2).sum()

    def agg_for_mean(self, series):
        return series.sum()

    def compute(self):
        outs = []
        for var in self.vars:
            for state in var['state']:
                state_name = self._get_state_name(var, state)

                if state == 'mean':
                    var_state_out = self.__dict__[state_name] / self.__dict__[self._get_counter_name(var)]

                elif state == 'var':
                    mean_state_name = self._get_state_name(var, 'mean')
                    var_state_out = (self.__dict__[state_name] / self.__dict__[self._get_counter_name(var)]) - \
                                    (self.__dict__[mean_state_name] / self.__dict__[self._get_counter_name(var)]) ** 2

                else:
                    var_state_out = self.__dict__[state_name]

                outs.append((state_name, var_state_out))

        rets = []
        for out in outs:
            var_name, state = out[0].split('_')
            var = [var for var in self.vars if var['name'] == var_name][0]

            if self.plot:
                fig = plt.figure(figsize=(12, 7))
                plt.plot(var['labels'], out[1].cpu().numpy())
                plt.title(self._get_state_name(var, state))
                plt.xticks(rotation=90)

                plt.xlabel(var_name)
                plt.ylabel('$%s(|y - \hat y|)$ in %s bins' % (state, var_name))

                plt.tight_layout()

                rets.append((out[0], fig))

        return rets

    def ndvi(self, x=None, y=None, ypred=None):
        MIR_wvls = search_spectral_window(770, 810, where=self.in_wvls[0])
        NIR_wvls = search_spectral_window(650, 680, where=self.in_wvls[0])

        MIR = np.mean(select(x, MIR_wvls, axis=-1), axis=-1)
        NIR = np.mean(select(x, NIR_wvls, axis=-1), axis=-1)

        return (MIR - NIR + 1e-5) / (MIR + NIR + 1e-5)

    def pred(self, x=None, y=None, ypred=None):
        return ypred

    def label(self, x=None, y=None, ypred=None):
        return y

    @classmethod
    def add_logger_specific_args(cls, parser, prepend=''):
        parser_spec = dict(stratify_along_vars=dict(type=str, default=["ndvi"], nargs="*"),
                           stratify_bins=dict(type=float, default=[0.1], nargs="*"),
                           stratify_lo=dict(type=float, default=[0], nargs="*"),
                           stratify_hi=dict(type=float, default=[1.01], nargs="*"),
                           plot=dict(type=int, default=1))
        parser = add_model_specific_args(parser=parser, parser_spec=parser_spec, prepend=prepend)
        return parser


class QuantizedConfusionMatrix(torchmetrics.Metric):
    def __init__(self, hist_lo, hist_hi, hist_nbins, lo_y=None, hi_y=None, nbins_y=None, density_ax=None,
                 log=True, *args, **kwargs):
        super(QuantizedConfusionMatrix, self).__init__()

        self.log = log
        self.density_ax = density_ax

        self.rangey = (lo_y, hi_y)
        self.rangex = (hist_lo, hist_hi)
        if lo_y is None or hi_y is None:
            self.rangey = self.rangex

        self.nbins_y = nbins_y
        self.nbins_x = hist_nbins
        if nbins_y is None:
            self.nbins_y = self.nbins_x

        self.bins = (self.nbins_x, self.nbins_y)

        self.add_state('hist', default=torch.zeros((self.nbins_x, self.nbins_y)), dist_reduce_fx='sum')
        self.xedges = None
        self.yedges = None

    def update(self, preds, target):

        # TODO: compute histogram as a tensor so no casting and moving is necessary
        hist, yedges, xedges = np.histogram2d(preds.cpu().numpy().flatten(), target.cpu().numpy().flatten(),
                                              bins=self.bins, range=[self.rangex, self.rangey])
        
        self.xedges_all, self.yedges_all = xedges, yedges
        self.xedges, self.yedges = [(rang[0], rang[-1])
                                    for rang in (xedges, yedges)]

        self.hist += torch.tensor(hist, device=self.hist.device)

    def plot(self, vmin=None, vmax=None, fig=None, ax=None, fig_close=True, plt_kwargs=None, 
             return_im=False, colorbar=True, subplot_ind=0, contour=False, contourf=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(**kwargs)

            if type(ax) is np.ndarray:
                ax = ax.flatten()
                ax = ax[subplot_ind]

        if plt_kwargs is None:
            plt_kwargs = dict()

        h = self.hist.cpu().numpy()
        if self.density_ax is not None:
            norm = np.sum(h, axis=self.density_ax)

            if self.density_ax == 0:
                h = h / norm[None, :]

            else:
                h = h / norm[:, None]

        if self.log:
            h = np.log10(h)
        
        extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]
        Xs, Ys = np.meshgrid(self.xedges_all[:-1], self.yedges_all[:-1])

        if contour:
            im = plt.contour(Xs, Ys, h, **plt_kwargs)

        elif contourf:
            im = plt.contourf(Xs, Ys, h, **plt_kwargs)

        else:
            im = ax.imshow(h, interpolation='nearest', origin='lower',
                           extent=extent, vmin=vmin, vmax=vmax, **plt_kwargs)
        
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        plt.tight_layout()

        if fig_close:
            plt.close(fig)
        
        if return_im:
            return fig, im

        return fig

    def compute(self, *args, **kwargs):
        fig = self.plot(*args, **kwargs)
        return fig

    def reset(self):
        super(QuantizedConfusionMatrix, self).reset()
        self.xedges = None
        self.yedges = None

    @classmethod
    def add_logger_specific_args(cls, parser, prepend=''):
        parser_spec = dict(hist_nbins=dict(type=int, default=500),
                           hist_lo=dict(type=float, default=0),
                           hist_hi=dict(type=float, default=1))
        parser = add_model_specific_args(parser=parser, parser_spec=parser_spec, prepend=prepend)
        return parser
