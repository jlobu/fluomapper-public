from argparse import ArgumentError, ArgumentParser

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d

import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from test_tube.argparse_hopt import OptArg, TTNamespace

from pytorch_lightning.loggers import MLFlowLogger as MLFlowLogger_
from pytorch_lightning.utilities import rank_zero_only

from tracking.loggers import _Logger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
import re
from copy import deepcopy

from fluomapper.config import get_avail_heads, get_avail_losses, NoneAction
from fluomapper.utils.run import add_model_specific_args, eval_with_valid_kwargs, init_with_valid_kwargs
from fluomapper.utils.nn import InputNorm

import numpy as np
from functools import reduce


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler, Dataset

from typing import Optional, Iterator, TypeVar
T_co = TypeVar("T_co", covariant=True)


class RepeatSampler(object):
    """ Sampler that repeats forever.
    from here https://github.com/pytorch/pytorch/issues/15849

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


#@variational_estimator
class _Module(pl.LightningModule):
    _avail_losses = get_avail_losses()

    BAYESIAN_CHOICES = ('backprop', 'dropout', None, 'none', 'None')

    def __init__(self, bayesian_mode=None, bayesian_n_samples_eval=100, bayesian_n_samples_elbo=10,
                 loss='mse', learning_rate=1e-3, lr_scheduler_factor=0, weight_decay=1e-3, min_lr=5e-5, lr_patience=1, 
                 lr_monitor_loss='val_loss', freeze_layers=None, input_norm_nr_train_epochs=2, 
                 freeze_input_norm=False, freeze_bn=True, train_only_id=False, 
                 freeze_all_but=None, ignore_ckpt_keys=None, load_closest_ids=False, *args, **kwargs):
        super(_Module, self).__init__()
        self.save_hyperparameters()

        self._val_logger = None

        self.ignore_ckpt_keys = [] if ignore_ckpt_keys is None else ignore_ckpt_keys
        self.load_closest_ids = load_closest_ids

        self.bayesian_mode = bayesian_mode
        assert self.bayesian_mode in self.BAYESIAN_CHOICES
        self.bayesian_n_samples_eval = bayesian_n_samples_eval
        self.bayesian_n_samples_elbo = bayesian_n_samples_elbo

        # self.initialize()
        self.learning_rate = learning_rate
        self.lr_scheduler_factor = lr_scheduler_factor
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay

        self.lr_monitor_loss = lr_monitor_loss

        self.freeze_layers = freeze_layers
        self.freeze_all_but = freeze_all_but
        self.freeze_bn = freeze_bn
        self.train_only_id = train_only_id

        self._loss = self._avail_losses[loss]
        if isinstance(self._loss, type):
            self._loss = init_with_valid_kwargs(self._loss, **kwargs)

        self.input_norm_nr_train_epochs = input_norm_nr_train_epochs if not freeze_input_norm else 0

        self._optimizer_prefixes = [None]

        self.automatic_optimization = True

        self.VAL_LOGS = []

    @property
    def val_logger(self):
        return self._val_logger

    def add_validation_logger(self, val_logger: _Logger):
        self._val_logger = val_logger

        # register any modular metrics
        for metric_name, metric in self._val_logger.modular_metrics.items():
            #metric = metric.to(device)
            self.add_module(metric_name, metric)

    def _validation_step(self, *args, **kwargs):
        if self.val_logger is None:
            return

        else:
            return self.val_logger.log_validation_step(*args, **kwargs)

    def _validation_epoch_end(self, outputs: List[Any]):
        if self.val_logger is None:
            return

        else:
            return self.val_logger.log_validation_epoch_end(outputs, model=self)

    def _init_weights(self, modules=None, mode='xavier'):
        if modules is None:
            modules = self.modules()

        for m in modules:
            if mode == 'xavier':
                if type(m) in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)

            else:
                raise NotImplementedError

    def set_dropout_to_train(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # ignore weight size differences
        # from here: https://github.com/PyTorchLightning/pytorch-lightning/issues/4690
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        del_keys = []
        not_present = []

        if self.load_closest_ids:
            m_dids = [(i, k) for i, k in model_state_dict.items()
                                               if 'data_source_ids' == i.split('.')[-1]]
            s_dids = [(i, k)  for i, k in state_dict.items()
                                               if 'data_source_ids' == i.split('.')[-1]]

            s_ids =  [(i, k)  for i, k in state_dict.items()
                                              if 'ids' == i.split('.')[-1]]

            m_ids = [(i, k) for i, k in model_state_dict.items()
                                              if 'ids' == i.split('.')[-1]]

            if len(m_dids) > 0 and len(s_dids) > 0:
                model_data_source_ids = m_dids[0][1].cpu().numpy()
                state_data_source_ids = s_dids[0][1].cpu().numpy()

                s_ids_k, s_ids = s_ids[0]
                m_ids_k, m_ids = m_ids[0]

                for n, data_source_id in enumerate(model_data_source_ids):
                    closest_id = np.argmin(np.abs(state_data_source_ids - data_source_id))
                    m_ids[n] = s_ids[closest_id]

                state_dict[s_ids_k] = m_ids

        for k in state_dict:
            if k in model_state_dict:
                if np.any([k.startswith(ig) for ig in self.ignore_ckpt_keys]):
                    state_dict[k] = model_state_dict[k]
                    print(f"Skip loading parameter: {k} was set to be ignored.")

                elif state_dict[k].shape != model_state_dict[k].shape:
                    #if np.any([k.startswith(ig) for ig in self.ignore_ckpt_keys]):
                    #    print(f"Skip loading parameter: {k}, "
                    #          f"required shape: {model_state_dict[k].shape}, "
                    #          f"loaded shape: {state_dict[k].shape}")
                    #    continue

                    #else:
                    print(f"Skip loading parameter: {k}, "
                          f"required shape: {model_state_dict[k].shape}, "
                          f"loaded shape: {state_dict[k].shape}")

                    state_dict[k] = model_state_dict[k]
                    is_changed = True

            else:
                not_present.append(k)
                is_changed = True

            #if 'ids' in k and self.model.init_ids_from is not None:
            #    del_keys.append(k)
        
        for k in del_keys:
            state_dict.pop(k)

        if is_changed:
            print('WARNING: loaded ckpt is different to architecture. '
                  'These keys are present in the saved state_dict but they are not in the model', not_present)

            checkpoint.pop("optimizer_states", None)

    def on_validation_model_eval(self):
        super(_Module, self).on_validation_model_eval()

        if self.bayesian_mode == 'dropout':
            self.set_dropout_to_train()

    def on_test_model_eval(self):
        super(_Module, self).on_test_model_eval()

        if self.bayesian_mode == 'dropout':
            self.set_dropout_to_train()

    def on_train_epoch_start(self):
        self.set_inp_norm_eval()

    def set_inp_norm_eval(self, force=True):
        if self.current_epoch >= self.input_norm_nr_train_epochs or force:
            for name, layer in self.named_modules():
                if isinstance(layer, InputNorm):
                    layer.set_eval()

    def bayesian_eval(self, x, y, std_multiplier=2):
        preds = torch.stack([self.forward(x) for _ in range(self.bayesian_n_samples_eval)])

        means = preds.mean(dim=0)
        stds = preds.std(dim=0)

        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)

        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        ic_acc = ic_acc.float().mean()

        return means, stds, ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

    def loss(self, *args, **kwargs):
        if isinstance(self._loss, type):
            return eval_with_valid_kwargs(self._loss, "forward", *args, **kwargs)
        else:
            return eval_with_valid_kwargs(None, self._loss, *args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        x, y = self.prepare_batch(batch)[:2]

        if self.bayesian_mode == 'backprop':
            loss = self.sample_elbo(inputs=x, labels=y,
                                    criterion=F.mse_loss,
                                    sample_nbr=self.bayesian_n_samples_elbo)
        else:
            ypred = self.forward(x)
            loss = self.loss(ypred, y, model=self, inp=x, mode='train', **kwargs, **batch[0])

        self.log('loss', loss.cpu().detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        input = batch[0]
        x, y = self.prepare_batch(batch)[:2]

        logs = dict([])
        if self.bayesian_mode is not None:
            ypred_mean, ypred_std, ic_acc, _, _ = self.bayesian_eval(x, y)

            logs.update(dict(val_step_stds=ypred_std.mean().cpu().detach(), val_step_ic_acc=ic_acc))

            # TODO: add bayesian specific validation metrics, do not reduce with mean
            ypred = ypred_mean
            if self.val_logger is not None:
                logs.update(self.val_logger.log_validation_step(ypred, y, inp=input))

        else:
            ypred = self.forward(x)
            if self.val_logger is not None:
                logs.update(self.val_logger.log_validation_step(ypred, y, inp=input))

        val_step_mse = self.loss(ypred, y, model=self, inp=x, mode="val", **kwargs, **batch[0]).cpu().detach()
        logs.update({'val_step_loss': val_step_mse})

        # if len(ypred.shape) <= 2:
        val_step_frmse = torch.sqrt((((y - ypred) / (y + 1e-3)) ** 2).median()).cpu().detach()
        logs.update(dict(val_step_frmse=val_step_frmse))

        self.VAL_LOGS.append(logs)

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def on_validation_epoch_end(self):
        outputs = self.VAL_LOGS

        if self.val_logger is not None:
            _ = self.val_logger.log_validation_epoch_end(outputs=outputs, model=self)

        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log('val_loss', avg_loss.cpu().detach(), sync_dist=True)

        if "val_step_frmse" in outputs[0]:
            avg_loss = torch.stack([x["val_step_frmse"] for x in outputs]).mean()
            self.log('frmse', avg_loss.cpu().detach(), sync_dist=True)

        if self.bayesian_mode is not None:
            avg_loss = torch.stack([x["val_step_stds"] for x in outputs]).mean()
            self.log('mean_stds', avg_loss.cpu().detach(), sync_dist=True)

            avg_loss = torch.stack([x["val_step_ic_acc"] for x in outputs]).mean()
            self.log('mean_ic_acc', avg_loss.cpu().detach(), sync_dist=True)

        self.VAL_LOGS.clear()

    def on_test_epoch_end(self, *args, **kwargs):
        return self.on_validation_epoch_end(*args, **kwargs)

    def get_trainable_parameters(self, set_requires_grad=True, freeze_bn=True, train_only_id=False, return_names=False):
        all_parameters = list(self.named_parameters())

        if not train_only_id and self.freeze_all_but is None:
            frozen_parameters = self.get_parameters_by_name(prefix=self.freeze_layers,
                                                            exclude_bn=not freeze_bn)

            if set_requires_grad:
                frozen_parameters = dict([(name, p.requires_grad_(False)) for name, p in frozen_parameters])
            
            parameters = dict([(name, p) for name, p in all_parameters if name not in frozen_parameters])

        else:
            frozen_parameters = all_parameters
            frozen_parameters = [(name, p.requires_grad_(False)) for name, p in frozen_parameters]

            if freeze_bn:
                for name, layer in all_parameters:
                    if isinstance(layer, BatchNorm1d):
                        layer.eval()
            
            if train_only_id:
                parameters = dict([(name, p.requires_grad_(True)) for name, p in all_parameters 
                              if ('ids' in name or name.split('.')[-1] == 'U') 
                              and torch.is_floating_point(p)])

            elif self.freeze_all_but is not None:
                parameters = dict([(name, p.requires_grad_(True)) for name, p in all_parameters 
                                    if np.any([name.startswith(s) for s in self.freeze_all_but])])

        if not return_names:
                parameters = [p for n, p in parameters.items()]


        return parameters
    
    def get_module_by_name(self, access_string, get_parent=False):
        names = access_string.split(sep='.')
        if get_parent:
            names = names[:-1]

        return reduce(getattr, names, self)

    def get_parameters_by_name(self, prefix=None, exclude_bn=False):
        if prefix is None:
            return []

        if type(prefix) is str:
            prefix = [prefix]

        if 'none' in prefix:
            return []

        all_prefix = [(name, param) for name, param in self.named_parameters()
                      if np.any([name.startswith(pre) for pre in prefix])]
        
        if exclude_bn:
            return [(name, param) for name, param in all_prefix 
                    if not isinstance(self.get_module_by_name(name, get_parent=True), nn.BatchNorm1d)]

        return all_prefix

    def configure_optimizers(self):
        weight_decay = self.weight_decay

        if self.bayesian_mode == "backprop":
            weight_decay = 0

        if not hasattr(self, '_optimizer_prefixes'):
            self._optimizer_prefixes = [None]

        assert np.all([np.all([not pre in op for op in list(set(self._optimizer_prefixes) - set([pre])) if op is not None])
                       for pre in self._optimizer_prefixes if pre is not None])

        params = self.get_trainable_parameters(set_requires_grad=True,
                                               freeze_bn=self.freeze_bn,
                                               train_only_id=self.train_only_id, 
                                               return_names=True)

        opts = []
        for prefix in self._optimizer_prefixes:
            other_prefixes = list(set(self._optimizer_prefixes) - set([prefix]))

            if prefix is None:
                _params = [p for n, p in params.items() if np.all([not pre in n for pre in other_prefixes])]

            else:
                _params = [p for n, p in params.items() if prefix in n]

            opt = torch.optim.Adam(_params, lr=self.learning_rate, weight_decay=weight_decay)
            if self.lr_scheduler_factor and prefix is None:
                opt = dict(optimizer=opt,
                           lr_scheduler=dict(
                                        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=self.lr_scheduler_factor,
                                                                                   min_lr=self.min_lr,
                                                                                   patience=self.lr_patience),
                                        monitor=self.lr_monitor_loss)
                           )

            elif self.lr_scheduler_factor and prefix is not None:
                opt = dict(optimizer=opt,
                           lr_scheduler=torch.optim.lr_scheduler.LambdaLR(opt, 
                                        lr_lambda=lambda epoch: max(self.lr_scheduler_factor ** epoch, self.min_lr / self.learning_rate)))

            opts.append(opt)

        #if len(opts) == 1:
        #    return opts[0]

        return opts

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        avail_losses = get_avail_losses()
        parser_spec = dict([
            ('learning_rate', dict(type=float, default=1e-4)),
            ('lr_scheduler_factor', dict(type=float, default=0)),
            ('min_lr', dict(type=float, default=5e-5)),
            ('lr_patience', dict(type=int, default=1)),
            ('loss', dict(type=str, default='mse')),
            ('weight_decay', dict(type=float, default=1e-3)),
            ('model_spectral_window_obs_wvl', dict(type=float, default=None, nargs='*', action=NoneAction)),
            ('bayesian_mode', dict(type=str, default=None, choices=cls.BAYESIAN_CHOICES, action=NoneAction)),
            ('bayesian_n_samples_elbo', dict(type=int, default=10)),
            ('bayesian_n_samples_eval', dict(type=int, default=10)),
            ('lr_monitor_loss', dict(type=str, default='val_loss')),
            ('freeze_layers', dict(type=str, default=None, nargs='+')),
            ('freeze_all_but', dict(type=str, default=None, nargs='+')),
            ('freeze_input_norm', dict(type=int, default=0)),
            ('freeze_bn', dict(type=int, default=1)),
            ('train_only_id', dict(type=int, default=0)),
            ('ignore_ckpt_keys', dict(type=str, nargs='*', default=None)),
            ('load_closest_ids', dict(type=int, default=False)),
        ])
        parser = add_model_specific_args(parser, parser_spec=parser_spec, prepend=prepend, ignore_overrides=False, **kwargs)

        for loss in avail_losses.values():
            if isinstance(loss, type) and hasattr(loss, 'add_model_specific_args'):
                parser = loss.add_model_specific_args(parser, prepend=prepend, **kwargs)

        return parser


class MLFlowLogger(MLFlowLogger_):
    def __init__(self, *args, **kwargs):
        super(MLFlowLogger, self).__init__(*args, **kwargs)
        self._add_run_id_tag()

    @rank_zero_only
    def _add_run_id_tag(self):
        run_id = self.run_id
        print('RUN_ID', run_id)
        self._mlflow_client.set_tag(run_id, 'run_id', run_id)


class HyperOptArgumentParser_(HyperOptArgumentParser):
    def __init__(self, *args, **kwargs):
        super(HyperOptArgumentParser_, self).__init__(*args, **kwargs)

    def opt_list(self, *args, **kwargs):
        options = kwargs.pop("options", None)
        tunable = kwargs.pop("tunable", False)

        # TODO: find better way to check if argument already registered
        try:
            self.add_argument(*args, **kwargs)
        except ArgumentError:
            pass

        for i in range(len(args)):
            arg_name = args[i]
            self.opt_args[arg_name] = OptArg(obj_id=arg_name, opt_values=options, tunable=tunable)

    def parse_known_args(self, args, namespace=None):
        # this function is a copy of HyperOptArgumentParser.parse_args with the exception of this line
        # allowing to bypass all unknown args
        results, argv = super(HyperOptArgumentParser, self).parse_known_args(args, namespace=namespace)

        # add hpc_exp_number if not passed in so we can never get None,
        # this is from HyperOptArgumentParser.__whitelist_args
        if HyperOptArgumentParser.SLURM_EXP_CMD not in args:
            results.__setattr__(HyperOptArgumentParser.SLURM_EXP_CMD, None)

        # extract vals
        old_args = vars(results)

        # override with json args if given
        if self.json_config_arg_name and old_args[self.json_config_arg_name]:
            for arg, v in self.__read_json_config(old_args[self.json_config_arg_name]).items():
                old_args[arg] = v

        # track args
        self.parsed_args = deepcopy(old_args)
        # attach optimization fx
        old_args['trials'] = self.opt_trials
        old_args['optimize_parallel'] = self.optimize_parallel
        old_args['optimize_parallel_gpu'] = self.optimize_parallel_gpu
        old_args['optimize_parallel_cpu'] = self.optimize_parallel_cpu
        old_args['generate_trials'] = self.generate_trials
        old_args['optimize_trials_parallel_gpu'] = self.optimize_trials_parallel_gpu

        return TTNamespace(**old_args), argv

    def opt_range(
            self,
            *args,
            **kwargs
    ):
        """
        Add new opt_range even if argument already exists.

        :param args:
        :param kwargs:
        :return:
        """
        low = kwargs.pop("low", None)
        high = kwargs.pop("high", None)
        arg_type = kwargs["type"]
        nb_samples = kwargs.pop("nb_samples", 10)
        tunable = kwargs.pop("tunable", False)
        log_base = kwargs.pop("log_base", None)

        # TODO: find better way to check if argument already registered
        try:
            self.add_argument(*args, **kwargs)
        except ArgumentError:
            pass

        arg_name = args[-1]
        self.opt_args[arg_name] = OptArg(
            obj_id=arg_name,
            opt_values=[low, high],
            arg_type=arg_type,
            nb_samples=nb_samples,
            tunable=tunable,
            log_base=log_base,
        )

    def _HyperOptArgumentParser__whitelist_cluster_commands(self, args, argv):
        """
        Patch super().__whitelist_cluster_commands in order to allow for unshashable parser values
        :param args:
        :param argv:
        :return:
        """
        parsed = {}

        # build a dict where key = arg, value = value of the arg or None if just a flag
        for i, arg_candidate in enumerate(argv):
            arg = None
            value = None

            # only look at --keys
            if '--' not in arg_candidate:
                continue

            # skip items not on the white list
            if arg_candidate[2:] not in HyperOptArgumentParser.CMD_MAP:
                continue

            arg = arg_candidate[2:]
            # pull out the value of the argument if given
            if i + 1 <= len(argv) - 1:
                if '--' not in argv[i + 1]:
                    value = argv[i + 1]

                if arg is not None:
                    parsed[arg] = value
            else:
                if arg is not None:
                    parsed[arg] = value

        # add the whitelist cmds to the args
        # NOTE : here set was changed to list in order to allow for list (or other unhashable) arguments
        all_values = list()
        for k, v in args.__dict__.items():
            all_values.append(k)
            all_values.append(v)

        for arg, v in parsed.items():
            v_parsed = self._HyperOptArgumentParser__parse_primitive_arg_val(v)
            all_values.append(v)
            all_values.append(arg)
            args.__setattr__(arg, v_parsed)

        # make list with only the unknown args
        unk_args = []
        for arg in argv:
            arg_candidate = re.sub('--', '', arg)
            is_bool = arg_candidate == 'True' or arg_candidate == 'False'
            if is_bool: continue

            if arg_candidate not in all_values:
                unk_args.append(arg)

        # when no bad args are left, return none to be consistent with super api
        if len(unk_args) == 0:
            unk_args = None

        # add hpc_exp_number if not passed in so we can never get None
        if HyperOptArgumentParser.SLURM_EXP_CMD not in args:
            args.__setattr__(HyperOptArgumentParser.SLURM_EXP_CMD, None)

        return args, unk_args


def cast_to_hyperopt_parser(parser, strategy='grid_search'):
    hopt_parser = HyperOptArgumentParser_(strategy=strategy, allow_abbrev=False)
    hopt_parser.__dict__.update(parser.__dict__)

    return hopt_parser


class CustomDistributedSampler(DistributedSampler):
    def __init__(
            self,
            dataset: Dataset,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = False,
            same_sequence: bool = True,
            num_samples: Optional[int] = None,
            **kwargs
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # Assume every GPU loads a disjoint subset of the dataset and that all subsets have the same size
        self.len_dset = len(self.dataset)
        self.num_samples = self.len_dset if num_samples is None else num_samples

        # self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        # Bool to control whether to have the same sequence of batches for every GPU or not
        self.same_sequence = same_sequence

    def __iter__(self) -> Iterator[T_co]:
        """Generate and send the same sequence of indices (batches) to every GPU.
        If we want to send a different sequence to every GPU, I think it is sufficient to
        add the rank to the seed (the bool 'same_sequence' controls this two behaviours).
        """
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()

            # Same sequence for every GPU
            if self.same_sequence:
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(self.num_samples, generator=g).tolist()  # type: ignore[arg-type]

            # Different sequence for every GPU
            else:
                g.manual_seed(self.seed + self.epoch + self.rank)
                indices = torch.randperm(self.len_dset, generator=g).tolist()[:self.num_samples]  # type: ignore[arg-type]

        else:
            indices = list(range(self.num_samples))  # type: ignore[arg-type]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

