import ast
import copy
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from os.path import join as pjoin

import mlflow
import pytorch_lightning as pl
import torch
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from test_tube import HyperOptArgumentParser

from fluomapper.config import get_avail_models, get_default_parser, get_avail_data_sources, ExpandPathAction
from fluomapper.utils.lightning_base import HyperOptArgumentParser_, cast_to_hyperopt_parser
from fluomapper.utils.run import make_parser_soft, update_namespace
from tracking.loggers import Logger

from pytorch_lightning.strategies import DDPStrategy

import logging
logging.basicConfig(level=logging.DEBUG)


def get_model_parser(parser, args, namespace=None):
    Model = get_avail_models()[namespace.model]
    parser = Model.add_model_specific_args(parser, args=args, namespace=namespace)
    return parser


def get_ckpt(reload_args):
    # if a path is provided
    if os.path.isfile(reload_args.reload_ckpt) or  reload_args.reload_ckpt == 'none' :
        return reload_args.reload_ckpt

    # if no path is provided reconstruct it from args in reload_args
    else:
        run = get_run(reload_args.run_id, reload_args.mlflow_backend_uri)
        artifact_path_splits = run.info.artifact_uri.replace('\\', '/').split('/')

        ind = artifact_path_splits.index(reload_args.run_id)
        artifact_path = pjoin(reload_args.mlflow_backend_uri, '/'.join(artifact_path_splits[ind -1:]))

        checkpoints_path = pjoin(os.path.dirname(artifact_path), 'checkpoints')
        # ckpt_path = os.path.normpath(pjoin(checkpoints_path, reload_args.reload_ckpt))

        ckpts = os.listdir(checkpoints_path)

        if reload_args.reload_ckpt == 'last':
            creation_times = [os.path.getmtime(pjoin(checkpoints_path, path)) for path in ckpts]

            ckpt = ckpts[np.argmax(creation_times)]

        elif reload_args.reload_ckpt == 'first':
            creation_times = [os.path.getmtime(pjoin(checkpoints_path, path)) for path in ckpts]

            ckpt = ckpts[np.argmin(creation_times)]

        elif reload_args.reload_ckpt == 'best':
            pattern = 'epoch=(\d+)-.*=([\d\.\w]+)\.ckpt'

            val_losses = []
            for ckpt in ckpts:
                match = re.search(pattern, ckpt)
                if match is not None and match[2] is not None:
                    val_losses.append(float(match[2]))
                else:
                    val_losses.append(np.inf)

            ckpt = ckpts[np.argmin(np.array(val_losses))]

        elif reload_args.reload_ckpt.isdigit():
            creation_times = [os.path.getmtime(pjoin(checkpoints_path, path)) for path in ckpts]

            order = np.argsort(creation_times)
            ckpts = np.array(ckpts)[order]
            ckpt = ckpts[int(reload_args.reload_ckpt)]

        else:
            raise NotImplementedError("reload_ckpt must be in [first, last, best]. "
                                      "You provided %s" % reload_args.reload_ckpt)

        ckpt_path = pjoin(checkpoints_path, ckpt)
        return ckpt_path


def load_ckpt(model_class, hparams=None, reload_args=None):
    if reload_args is None:
        return model

    ckpt_path = get_ckpt(reload_args)

    print('ABOUT TO LOAD CKPT', ckpt_path)

    # prevent wrong path when working on different machine
    if not os.path.exists(ckpt_path):
        path_split = ckpt_path.split('/')
        mlruns_sub_path = '/'.join(path_split[path_split.index('mlruns') + 1:])

        ckpt_path = pjoin(reload_args.mlflow_backend_uri, mlruns_sub_path)

    if os.path.exists(ckpt_path):
        hparams = vars(hparams) if hparams is not None else {}

        print('Loading ckpt %s ' % os.path.basename(ckpt_path))
        model = model_class.load_from_checkpoint(ckpt_path, **hparams, reload_args=reload_args, strict=False, weights_only=False)
        model.loaded_checkpoint = ckpt_path
    else:
        raise Exception('ckpt file does not exist. You searched path %s' % ckpt_path)

    return model


def get_model(hparams, reload_args=None):
    Model = get_avail_models()[hparams.model]
    
    if reload_args is not None and reload_args.reload_ckpt is not None and reload_args.reload_ckpt != 'none' :
        print('LOADING MODEL with these reload_args', reload_args)
        model = load_ckpt(Model, copy.deepcopy(hparams), reload_args)

    else:
        model = Model(**vars(copy.deepcopy(hparams)), reload_args=reload_args)

    return model, hparams


def set_experiment_tags(hparams, reload_args=None):
    # TAGGING
    user_specified_tags = dict([])  # your own tags here
    user_specified_tags[MLFLOW_RUN_NAME] = hparams.name
    # user_specified_tags[MLFLOW_GIT_COMMIT] = str(sp.check_output(
    #                                                   'git -C %s rev-parse HEAD' % os.path.dirname(__file__),
    #                                                    shell=True).strip().decode('utf-8'))

    if 'SLURM_JOB_ID' in os.environ:
        user_specified_tags['job_id'] = os.environ['SLURM_JOB_ID']

    if reload_args is not None:
        if 'run_id' in reload_args:
            user_specified_tags['parent_run_id'] = str(reload_args.run_id)

        if 'reload_ckpt' in reload_args:
            user_specified_tags['parent_ckpt'] = str(reload_args.reload_ckpt)

    tags = context_registry.resolve_tags(user_specified_tags)
    return tags


def namespace_to_list(spec):
    if not type(spec) is dict:
        spec = vars(spec)

    st = []
    for key, item in spec.items():
        if item is None:
            continue

        # if item is False
        if type(item) is bool and not item:
            item = '0'

        # if item is True
        if type(item) is bool and item:
            item = '1'

        if type(item) is str:
            item = item.split(' ')

        if type(item) in (list, tuple):
            item_string = list(map(str, item))
        else:
            item_string = str(item)
            item_string = [item_string]

        st.append('--%s' % str(key))
        st += item_string
    return st


def clean_args(args=None):
    if args is None:
        args = sys.argv

    def _clean(arg):

        try:
            new_arg = ast.literal_eval(arg)
        except (SyntaxError, ValueError):
            new_arg = arg

        if type(new_arg) not in (list, tuple):
            new_arg = [str(new_arg)]

        else:
            new_arg = list(map(str, new_arg))

        return new_arg

    if type(args) is list:
        new_args = []
        for i in range(len(args)):
            new_args += _clean(args[i])

    elif type(args) is dict:
        new_args = {}
        for key, value in args.items():
            try:
                clean_value = ast.literal_eval(value)

            except(SyntaxError, ValueError):
                clean_value = value

            new_args[key] = clean_value

    else:
        raise NotImplementedError

    return new_args


def construct_parser(args=None, soft=False, preloaded_args=None):

    default_parser = get_default_parser()
    if soft:
        make_parser_soft(default_parser)

    pargs, _ = default_parser.parse_known_args(args=args)
    pargs = update_namespace(preloaded_args, pargs, ignore_none=soft)

    # load data source, get parser for data source
    parser = get_avail_data_sources()[pargs.dset].add_argparse_args(default_parser)
    if soft:
        make_parser_soft(parser)
    parser.allow_abbrev = False

    pargs, _ = parser.parse_known_args(args=args)

    # add loaded args if any
    pargs = update_namespace(preloaded_args, pargs, ignore_none=soft)

    # get model specific parser
    parser = get_model_parser(parser, args=args, namespace=pargs)

    # get logger specific parser
    parser = Logger.add_logger_specific_args(parser, args=args, namespace=pargs)
    if soft:
        make_parser_soft(parser)

    parser.allow_abbrev = False
    return parser


def process_args(args, reload_args=None):
    ret = {}

    # ARGs PROCESSING
    limit_val_batches = args.limit_val_batches_perc \
        if args.limit_val_batches_perc is not None \
        else args.limit_val_batches_n \
        if args.limit_val_batches_n is not None \
        else 1.0

    ret['limit_val_batches'] = limit_val_batches

    limit_train_batches = args.limit_train_batches_perc \
        if args.limit_train_batches_perc is not None \
        else args.limit_train_batches_n \
        if args.limit_train_batches_n is not None \
        else 1.0

    ret['limit_train_batches'] = limit_train_batches

    # TRAINER instantiation
    # mlflow.pytorch.autolog()
    accelerator = args.accelerator if args.accelerator != 'none' else None
    ret['accelerator'] = accelerator

    if torch.cuda.is_available():
        gpus = args.n_gpus if args.n_gpus is not None and not args.exact_gpus_specified else args.gpus

    else:
        gpus = None

    ret['gpus'] = gpus

    if reload_args is not None and 'reload_ckpt' in reload_args and reload_args.reload_ckpt is not None:
        reload_args.reload_ckpt = get_ckpt(reload_args)

    if reload_args is not None and 'run_id' in reload_args and reload_args.run_id is not None \
            and reload_args.reload_hparams:
        ret.update(dict(run_id=reload_args.run_id,
                        mlflow_backend_uri=reload_args.mlflow_backend_uri,
                        reload_trainer=reload_args.reload_trainer))

    if reload_args is not None and 'reload_ckpt' in reload_args and reload_args.reload_ckpt is not None:
        ret.update(dict(reload_ckpt=reload_args.reload_ckpt,
                        run_id=reload_args.run_id,
                        mlflow_backend_uri=reload_args.mlflow_backend_uri),
                        reload_trainer=reload_args.reload_trainer)

        ret.update(reload_exclude=reload_args.reload_exclude)

    if 'val_patch_on_init' in ret and ret['val_patch_on_init'] and isinstance(ret['limit_val_batches'], float):
        # in this case make trainer go over all samples but restrain dataset in reading
        ret['limit_val_batches'] = 1.0

    if 'trn_patch_on_init' in ret and ret['trn_patch_on_init'] and isinstance(ret['limit_train_batches'], float):
        # in this case make trainer go over all samples but restrain dataset in reading
        ret['limit_train_batches'] = 1.0

    return ret


def get_reload_args(args=None):
    if args is None:
        args = sys.argv

    parser = ArgumentParser()
    parser.add_argument('--reload_hparams', type=int, default=False)
    parser.add_argument('--reload_trainer', type=int, default=False)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--mlflow_backend_uri', type=str, default=None, action=ExpandPathAction)
    parser.add_argument('--reload_ckpt', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--reload_exclude', type=str, default=None)

    parsed_args, _ = parser.parse_known_args(args)

    return parsed_args


def get_run(run_id, mlflow_backend_uri=None):
    if mlflow_backend_uri is not None:
        mlflow.set_tracking_uri("file:%s" % mlflow_backend_uri)

    run = mlflow.get_run(run_id)
    return run


def get_hparams(run_id, mlflow_backend_uri=None, **kwargs):
    run = get_run(run_id, mlflow_backend_uri)
    args = Namespace(**clean_args(args=run.data.params))
    return args


def parse(args=None, parser=None, process=True, do_reload=True):
    if args is None:
        args = sys.argv

    if type(args) is str:
        args = args.split()

    if type(args) in (Namespace, dict):
        args = namespace_to_list(args)

    # do this separately from other args
    # this way reload args are not logged along with other args
    reload_args = get_reload_args(args)

    loaded_args = dict([])
    if reload_args.reload_hparams and do_reload:
        loaded_args = get_hparams(**vars(reload_args))

    args = clean_args(args)

    # construct default parser, pass in the args from loaded_args to construct the model parser correctly
    if parser is None:
        parser = construct_parser(args=args, soft=reload_args.reload_hparams, preloaded_args=loaded_args)

    # get only known args
    args, _ = parser.parse_known_args(args=args)

    # override loaded_args with args
    if do_reload:
        args = update_namespace(loaded_args, args, ignore_none=reload_args.reload_hparams)

    if process:
            processed_args = process_args(args, reload_args=reload_args)
            args = update_namespace(args, processed_args)

            # do this manually as this is not done in HyperOptArgumentParser
            if type(parser) in (HyperOptArgumentParser, HyperOptArgumentParser_):
                parser.parsed_args.update(processed_args)

    return args, parser, reload_args


def get_data_module(hparams):
    data_module = get_avail_data_sources()[hparams.dset](**vars(hparams))

    data_module, new_hparams = val_dset_info(data_module, hparams)

    hparams = new_hparams if new_hparams is not None else hparams
    return data_module, hparams


#@rank_zero_only
def val_dset_info(data_module, hparams):
    data_module.prepare_data()

    _kwargs = dict(**data_module.dset_kwargs, **data_module.val_data)
    _kwargs['reader_batch_size'] = 1
    _kwargs['batch_size'] = 1
    _kwargs['patch_on_init'] = 0
    _kwargs['shard_along'] = None
    _kwargs['load'] = False
    val_dset = data_module.get_dset(**_kwargs)

    batch_raw = val_dset.__getitem__(0, return_wvl=True)
    example_batch = [[data_module.collate_fn((bb, ))  # add batch dimension
                     for bb in b] if b is not None else None
                     for b in batch_raw]

    batch_shape = tuple([tuple(b[0].shape)
                         if b is not None and type(b[0]) is not dict
                         else tuple(b[0]['obs'].shape) if b is not None
                         else None
                         for b in example_batch])

    # if label is provide
    if batch_shape[-1] is not None:
        in_wvls, out_wvls = [b[1] if b is not None else None
                             for b in [example_batch[0], example_batch[-1]]]

    # else assume out_wvls == in_wvls
    else:
        in_wvls = example_batch[0][1] if example_batch[0][1] is not None else None
        out_wvls = torch.as_tensor(_kwargs['spectral_window_label_wvl'])
    
    # update hparams
    hparams = vars(hparams)
    hparams.update(dict(batch_shape=batch_shape, 
                        in_wvls=in_wvls, 
                        out_wvls=out_wvls,
                        data_source_ids=val_dset.path_ids if hasattr(val_dset, 'path_ids') else None, 
                        data_sources=val_dset.paths if hasattr(val_dset, 'paths') else None
                        ))
    hparams = Namespace(**hparams)
    
    # update data module with info from loaded dataset
    try:
        # recalculate length of reader0
        data_module.num_samples_dloader_per_source = min(val_dset.readers[0].source_lengths // data_module.dset_kwargs['reader_batch_size'])

    except AttributeError:
        print('Found no readers in data set.')
        pass

    return data_module, hparams


def parse_hypopt(args=None, parser=None, opt_lists=None, opt_ranges=None, strategy='random_search', process=True):
    # adapted from https://pytorch-lightning.readthedocs.io/en/stable/clouds/slurm.html
    if parser is None:
        args, parser, _ = parse(args=args, process=True)
        args = namespace_to_list(args)

    parser = cast_to_hyperopt_parser(parser, strategy=strategy)
    do_hyper_opt_search = strategy is not None

    if do_hyper_opt_search:

        if opt_lists is None:
            opt_lists = dict([])

        if opt_ranges is None:
            opt_ranges = dict([])

        for kword, optlist in opt_lists.items():
            parser.opt_list('--%s' % kword, options=optlist, tunable=True)

        for kword, optrange_dict in opt_ranges.items():
            # prepare options
            opts = dict(low=optrange_dict['low'],
                        high=optrange_dict['high'],
                        type=optrange_dict['type'],
                        nb_samples=optrange_dict['nb_samples'],
                        tunable=True,
                        log_base=optrange_dict['log_base'])

            parser.opt_range('--%s ' % kword, **opts)

    # parse again, this time with opts
    args, parser, reload_args = parse(args=args, parser=parser, process=process, do_reload=False)
    return args, parser, reload_args


def get_trainer(hparams, reload_args=None):
    tracking_uri = "file:%s" % hparams.tracking_uri

    # LOGGER
    tags = set_experiment_tags(hparams, reload_args=reload_args)
    logger = MLFlowLogger(experiment_name=hparams.experiment,
                          tags=tags,
                          tracking_uri=tracking_uri,
                          )

    #run = mlflow.active_run()
    #run.set_tag('run_id', logger.run_id)

    # store ckpts to logging directory
    checkpoint_dirname = pjoin(tracking_uri[len('file:'):], str(logger.experiment_id), str(logger.run_id), 'checkpoints')

    # CALLBACKS
    callbacks = []

    if hparams.top_k_ckpts:
        checkpointer = ModelCheckpoint(
            monitor=hparams.log_monitor_loss,
            dirpath=checkpoint_dirname,
            filename='{epoch:02d}-{%s:.2f}' % hparams.log_monitor_loss,
            save_top_k=hparams.top_k_ckpts,
            mode='min',
            every_n_epochs=hparams.check_val_every_n_epoch,
            save_last=True
        )
        callbacks += [checkpointer]

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    profiler = hparams.profiler
    if False:  #hparams.profiler == 'pytorch':
        profiler_dirname = pjoin(tracking_uri[len('file:'):], str(logger.experiment_id), str(logger.run_id))
        profiler = PyTorchProfiler(dirpath=profiler_dirname, file_name='pytorch_profile.txt',
                                   profile_memory=True)


    # Explicitly specify the process group backend if you choose to
    ddp = DDPStrategy(process_group_backend="nccl", 
                      find_unused_parameters=True,
                      static_graph=False)
    
    if hparams.early_stopping:
        early_stopping = EarlyStopping(monitor="val_loss", mode="min")
        callbacks.append(early_stopping)

    # fix seeds
    torch.random.seed()
    trainer = pl.Trainer(logger=logger,
                         strategy=ddp,
                         devices=hparams.gpus,
                         accelerator='gpu',
                         num_nodes=hparams.n_nodes,
                         val_check_interval=hparams.val_check_interval,
                         fast_dev_run=hparams.fast_dev_run,
                         limit_val_batches=hparams.limit_val_batches,
                         limit_train_batches=hparams.limit_train_batches,
                         check_val_every_n_epoch=hparams.check_val_every_n_epoch,
                         overfit_batches=hparams.overfit_batches,
                         max_epochs=hparams.max_epochs,
                         profiler=profiler,
                         callbacks=callbacks,
                         default_root_dir=checkpoint_dirname,
                         precision=hparams.precision,
                         sync_batchnorm=hparams.sync_batchnorm,
                         enable_progress_bar=True,
                         reload_dataloaders_every_n_epochs=hparams.reload_dataloaders_every_n_epochs,
                         #automatic_optimization=~np.any([hparams.encoding_unlearn_xtrack,
                         #                                hparams.params_unlearn_xtrack]),
                         use_distributed_sampler=getattr(hparams, 'shard_along', None) is None or \
                                                 getattr(hparams, 'shard_along', None) == 'none'
                         )
    return trainer


def prepare_val_logger(hparams, trainer, model):
    # VALIDATION logging
    # batch_size = hparams.batch_shape[0][0]
    #val_epoch_size = int(len(val_dloader) * trainer.limit_val_batches) \
    #                    if type(hparams.limit_val_batches) is float else hparams.limit_val_batches
    
    val_logger = Logger(**vars(hparams))  #, val_epoch_size=val_epoch_size)
    model.add_validation_logger(val_logger)

    return val_logger
