import ast
import copy
import inspect
import matplotlib as mpl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import re
import torch
import torch.nn as nn
import torchmetrics
from itertools import chain
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError
from types import FunctionType

from tracking.metrics import mean, median, QuantizedConfusionMatrix, EvidentialLogger, StratifiedValidation
from fluomapper.utils.data import search_spectral_window, permute_channels, select
from fluomapper.utils.run import init_with_valid_kwargs, make_parser_soft, update_namespace, add_model_specific_args, \
    eval_with_valid_kwargs

__METRICS__ = dict(r2=R2Score,
                   mse=MeanSquaredError,
                   mae=MeanAbsoluteError,
                   hist=QuantizedConfusionMatrix,
                   evidential=EvidentialLogger,
                   stratify=StratifiedValidation)

__REDUCERS__ = dict(mean=mean,
                    median=median)


class _Logger(object):
    def __init__(self, val_metrics=None, in_wvls=None, out_wvls=None, set_up=None, name_prepend='', label=None,
                 *args, **kwargs):
        super(_Logger, self).__init__()

        self.labels = label if label is not None else ['']

        if set_up is None:
            self.val_metrics, self.val_wvls = self.parse_val_metrics(val_metrics)

        else:
            self.val_metrics, self.val_wvls = set_up

        self.name_prepend = name_prepend

        self.out_wvls = out_wvls
        self.in_wvls = in_wvls

        self.indices = None

        self.args = args
        self.kwargs = kwargs

        self.metrics = None
        self._idx_modular_metrics = None

    def set_up(self):
        # get indices for wvls
        uniq_wvls = list(chain(*[wvls for wvls in self.val_wvls if wvls is not None]))
        indices = search_spectral_window(*list(chain(*self.val_wvls)), where=self.out_wvls[0], pairs=False)
        self.indices = dict(zip(uniq_wvls, indices))

        # flatten out metrics
        metrics = []
        for _ in self.labels:
            for metric, wvls in zip(self.val_metrics, self.val_wvls):
                if len(wvls) == 0:
                    metrics.append((metric, None))

                else:
                    for wvl in wvls:
                        metrics.append((metric, wvl))

        if len(self.val_metrics) > 0:
            self.val_metrics, self.val_wvls = list(zip(*metrics))

        self.metrics = self._setup_metrics(*self.args, **self.kwargs)

        self._idx_modular_metrics = [i for i, (metric_name, metric) in enumerate(self.metrics.items())
                                     if issubclass(type(metric[0]), nn.Module)]

        return self

    @classmethod
    def parse_val_metrics(cls, val_metrics):
        if type(val_metrics) is list:
            val_metrics = ' '.join(val_metrics)

        if val_metrics is None:
            return [], []

        pattern = '[\w\.]+(?:\([\s\w,]+\))?'
        groups = re.findall(pattern, val_metrics)

        val_metrics, val_wvls = [], []

        pattern = '[\w\.]+'
        for group in groups:
            parts = re.findall(pattern, group)

            val_metrics.append(parts[0])
            if len(parts) > 0:
                val_wvls.append(list(map(float, parts[1:])))

            else:
                val_wvls.append(None)

        return val_metrics, val_wvls

    @property
    def modular_metrics(self):
        return dict([(metric_name, metric[0]) for i, (metric_name, metric) in enumerate(self.metrics.items())
                     if i in self._idx_modular_metrics])

    def _setup_metrics(self, default_reducer='mean', *args, **kwargs):
        parsed_metrics = dict([])

        # add wvls info for individual metrics
        kwargs.update(in_wvls=self.in_wvls, out_wvls=self.out_wvls)

        if self.val_metrics is None or len(self.val_metrics) == 0:
            return parsed_metrics

        for i, (metric, wvl) in enumerate(zip(self.val_metrics, self.val_wvls)):
            for label in self.labels:
                if type(metric) is str:
                    tup, str_tup = self._get_spec((metric, default_reducer), *args, **kwargs)

                    if wvl is not None:
                        str_tup.append(wvl)

                elif type(metric) in (tuple, list):
                    tup, str_tup = self._get_spec(metric, *args, **kwargs)

                else:
                    raise ValueError('Metric argument %d must be string or tuple of strings or functions.')

                name = '%s_%s_%s' % tuple(str_tup) if len(str_tup) == 3 \
                    else '%s_%s' % tuple(str_tup) if len(str_tup) == 2 \
                    else tuple(str_tup)[0]

                if label is not None:
                    name += f'__{label}'

                name = name.replace('.', '')
                parsed_metrics[name] = tup

        return parsed_metrics

    def _is_valid_metric(self, metric_cls):
        valid_metric_base_cls = [torchmetrics.Metric]
        return np.any([issubclass(metric_cls, m) for m in valid_metric_base_cls])

    def _get_spec(self, tup, *args, **kwargs):
        spec = []
        spec_str = []

        for i, (obj, lib) in enumerate(zip(tup, (__METRICS__, __REDUCERS__))):

            name = None

            if type(obj) is str:
                name = obj
                obj = lib[obj]

            # force setting reducer to metric itself if metric is a torchmetrics metric
            if i == 1 and self._is_valid_metric(type(spec[0])):
                obj = spec[0]
                name = None

            elif type(obj) is FunctionType:
                name = obj.__name__

            # if obj is an torchmetrics.Metric object
            elif i == 0 and self._is_valid_metric(type(obj)):
                name = obj.__class__.__name__ if name is None else name

            # if obj is an torchmetrics.Metric class
            # make one for each val_wvl
            elif i == 0 and inspect.isclass(obj) and self._is_valid_metric(obj):
                obj = init_with_valid_kwargs(obj, *args, **kwargs)
                name = obj.__name__ if name is None else name

            else:
                raise ValueError('function %s must be string in %s or of type function' % (str(obj), str(lib.keys())))

            spec += [obj]
            if name is not None:
                spec_str += [name]

        return tuple(spec), spec_str

    @staticmethod
    def _get_step_name(metric_name):
        return '%s_%s' % (metric_name, 'step')

    def possibly_flatten(self, metric, pred, true):
        # if it's multiple outputs
        if pred.shape[1] > 1 and not hasattr(metric, 'EXPECTED_DIM'):
            pred = pred[:, 0]
            true = true[:, 0]

        elif pred.shape[0] > 1 and hasattr(metric, 'EXPECTED_DIM'):
            pred = pred[:, :metric.EXPECTED_DIM]
            true = true[:, :metric.EXPECTED_DIM]

        if len(pred.shape) > 2:
            # flatten by default
            if not hasattr(metric, 'DO_FLATTEN') or metric.DO_FLATTEN:
                return pred.flatten(), true.flatten()

        else:
            if not hasattr(metric, 'DO_FLATTEN') or metric.DO_FLATTEN:
                return pred.flatten(), true.flatten()

        return pred.squeeze(), true.squeeze()

    def add_label_dim(self, v):
        if len(v.shape) == 1 or (len(v.shape) == 3 and v.shape[-1] == v.shape[-2]):
            v = v[:, None]

        return v

    def log_validation_step(self, pred, true, inp, images=False):
        # TODO do this in parallel

        pred = self.add_label_dim(pred)
        true = self.add_label_dim(true)

        val_metrics_step = dict([])
        if len(self.metrics) == 0:
            return val_metrics_step

        if images:
            pred, true = permute_channels(pred, reverse=True), permute_channels(true, reverse=True)

        # get only step metrics (no reducers)
        metrics = dict([(n, m[0]) for n, m in self.metrics.items()])

        for i, (metric_name, metric) in enumerate(metrics.items()):

            if self.labels is not None:
                label = metric_name.split('__')[-1]
                label_ind = self.labels.index(label)

            else:
                label_ind = 0

            if self._is_valid_metric(type(metric)):

                # check if have to update for multiple wvls
                if self.val_wvls[i] is not None:
                    wvl = self.val_wvls[i]
                    eval_with_valid_kwargs(metric, 'update',
                                           *self.possibly_flatten(metric, pred[..., self.indices[wvl]],
                                                                  true[..., self.indices[wvl]]),
                                           inp=inp)

                else:
                    eval_with_valid_kwargs(metric, 'update',
                                           *self.possibly_flatten(metric, pred[:, [label_ind]], true[:, [label_ind]]),
                                           inp=inp)

            elif type(metric) is FunctionType:
                val_metrics_step[self._get_step_name(metric_name)] = metric(pred[:, [label_ind]].flatten(),
                                                                            true[:, [label_ind]].flatten())

            else:
                raise NotImplementedError

        return val_metrics_step

    def log_validation_epoch_end(self, outputs, model):
        # TODO do this in parallel
        val_metrics = dict([])

        # get only reducers (no step metrics)
        metrics = dict([(n, m[1]) for n, m in self.metrics.items()])
        for metric_name, metric in metrics.items():

            if self._is_valid_metric(type(metric)):
                val_metrics[metric_name] = metric.compute()
                metric.reset()

            elif type(metric) is FunctionType:
                val_metrics[metric_name] = metric(torch.stack([x[self._get_step_name(metric_name)] for x in outputs]))

            else:
                raise NotImplementedError

            if type(val_metrics[metric_name]) is not list:
                self._log(to_be_logged=val_metrics[metric_name], name=self.name_prepend + metric_name, model=model,
                          run_id=model.logger.run_id, current_epoch=model.current_epoch)

            else:
                for name, m in val_metrics[metric_name]:
                    self._log(to_be_logged=m, name=self.name_prepend + metric_name + '_%s' % name, model=model,
                              run_id=model.logger.run_id, current_epoch=model.current_epoch)

        return val_metrics

    def _log(self, to_be_logged, model, name, run_id, current_epoch):
        if type(to_be_logged) is mpl.figure.Figure:
            model.logger.experiment.log_figure(run_id=run_id,
                                               figure=to_be_logged,
                                               artifact_file='epoch=%d_val_%s.png' % (current_epoch, name))

        else:
            model.log(name, to_be_logged.cpu().detach(), sync_dist=True)

    @classmethod
    def add_logger_specific_args(cls, parser, args, namespace=None):
        parser.add_argument('--val_metrics', nargs='*', type=str, default=None)
        # copy to not return a soft parser
        tmp_parser = copy.deepcopy(parser)
        make_parser_soft(tmp_parser)

        args, _ = tmp_parser.parse_known_args(args=args)
        args = update_namespace(namespace, args, ignore_none=True)

        if 'val_metrics' in args and args.val_metrics is not None:
            val_metrics, _ = cls.parse_val_metrics(args.val_metrics)

            for metric_name in val_metrics:
                splits = metric_name.split('.')

                if len(splits) == 1:
                    logger_name, metric_name = '', splits[0]

                else:
                    logger_name, metric_name = splits

                metric = __METRICS__[metric_name]

                if hasattr(metric, 'add_logger_specific_args'):
                    parser = metric.add_logger_specific_args(parser, prepend=logger_name)

        return parser


class Logger(_Logger):
    def __init__(self, labels=None, *args, **kwargs):
        super(Logger, self).__init__(*args, **kwargs)

        # find out whether we have logging by name
        self.split_names = [(v.split('.'), wvl) for v, wvl in zip(self.val_metrics, self.val_wvls)]
        self.logger_names = np.unique([v[0] for v, _ in self.split_names if len(v) > 1])

        self.is_logging_per_wvl = len(self.logger_names) > 0

        self.set_up_args = args
        self.set_up_kwargs = kwargs
        self.reset()

    def reset(self):
        if self.is_logging_per_wvl:

            metrics_sorted_by_name = [(v[0], '.'.join(v[1:]), wvl)  # name, metric_type, wvl
                                      for i, (v, wvl) in enumerate(self.split_names) if len(v) > 1]

            metrics_sorted_by_name = dict([(name, [item[1:] for item in metrics_sorted_by_name if item[0] == name])
                                           for name in self.logger_names])

            self.val_metrics = dict([(name, _Logger(set_up=list(zip(*metrics_sorted_by_name[name])), *self.set_up_args,
                                                    name_prepend='%s_' % name,
                                                    **self.split(self.set_up_kwargs, name)).set_up())
                                     for name in self.logger_names])

        else:
            self.logger_names = ['main']
            self.val_metrics = dict(main=_Logger(*self.set_up_args, **self.set_up_kwargs).set_up())

        self._updated_val_metrics = set([])

    def split(self, dic, key):
        new_dic = dict([(k.split('.')[1], val) if len(k.split('.')) > 1 and k.split('.')[0] == key
                        else (k, val) for k, val in dic.items()])

        return new_dic

    @_Logger.modular_metrics.getter
    def modular_metrics(self):
        return dict([('%s_%s' % (name, metric_name), metric[0])
                     for name in self.logger_names
                     for metric_name, metric in self.val_metrics[name].metrics.items()])

    def log_validation_epoch_end(self, *args, logger_name=None, **kwargs):
        if logger_name is None:
            logger_name = list(self._updated_val_metrics)

        if type(logger_name) is str:
            logger_name = [logger_name]

        for ln in logger_name:
            self.val_metrics[ln].log_validation_epoch_end(*args, **kwargs)

        self._updated_val_metrics = set([])

    def log_validation_step(self, *args, logger_name=None, **kwargs):
        if logger_name is None:
            logger_name = 'main'

        if logger_name in self.val_metrics:
            self._updated_val_metrics.add(logger_name)
            return self.val_metrics[logger_name].log_validation_step(*args, **kwargs)

        else:
            return dict([])

    @classmethod
    def add_logger_specific_args(cls, parser, prepend='', *args, **kwargs):
        parser = super(Logger, cls).add_logger_specific_args(parser, *args, **kwargs)
        parser_spec = dict([('val_channel', dict(type=int, default=None))])

        parser = add_model_specific_args(parser=parser, parser_spec=parser_spec, prepend=prepend)
        return parser
