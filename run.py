import torch
import pytorch_lightning as pl

from ui import get_model, parse, get_data_module, parse_hypopt, get_trainer, prepare_val_logger
from test_tube.hpc import SlurmCluster

import numpy as np
import pickle as pkl
from functools import partial

from fluomapper.utils.run import update_namespace
import multiprocessing as mp


def train_model(trainer, model, data_module, hparams):
    trainer.fit(model, data_module, 
                ckpt_path=hparams.reload_ckpt if 'reload_ckpt' in hparams and hparams.reload_ckpt != 'none'
                          and hparams.reload_trainer else None)


def eval_model(trainer, model, dloader, validate=True):
    trainer.test(model, test_dataloaders=dloader)


# @profile
def run_directly(args=None, **kwargs):
    args, parser, reload_args = parse(args=args)
    return run_(args, reload_args)


def reload_data_module(args=None, parser=None):
    args, parser, reload_args = parse(args, parser)
    data_module, args = get_data_module(args)

    return data_module, args, reload_args


def reload_model(args=None, parser=None, with_dm=True):
    if with_dm:
        data_module, args, reload_args = reload_data_module(args=args, parser=parser)
        model, args = get_model(args, reload_args=reload_args)
        return data_module, model

    else:
        args, parser, reload_args = parse(args, parser, do_reload=True)
        model, args = get_model(args, reload_args=reload_args)
        return model


# @profile
def prepare_run(args, reload_args=None):
    data_module, args = get_data_module(args)
    model, args = get_model(args, reload_args=reload_args)
    trainer = get_trainer(args, reload_args=reload_args)
    
    if args.auto_lr_find:
        trainer.tune(model, datamodule=data_module)
    
    val_logger = prepare_val_logger(args, #val_dloader=data_module.val_dataloader(),
                                    model=model, trainer=trainer)

    return data_module, model, trainer, val_logger, args


# @profile
def run_(args, reload_args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    pl.seed_everything(42)

    data_module, model, trainer, val_logger, hparams = prepare_run(args, reload_args=reload_args)

    if args.train:
        train_model(trainer=trainer, model=model, data_module=data_module, hparams=hparams)

    elif args.validate:
        eval_model(trainer, model, dloader=data_module.val_dataloader(), validate=True)

    elif args.test:
        eval_model(model, model, dloader=data_module.test_dataloader())

    elif args.predict:
        pass


def run_with_hypopt(args=None, opt_lists=None, opt_ranges=None, strategy='random_search', nb_trials=1):
    args, parser, reload_args = parse_hypopt(args=args, opt_lists=opt_lists, opt_ranges=opt_ranges,
                                             strategy=strategy)

    if nb_trials == 'all':
        nb_trials = None

    for trial in args.trials(nb_trials):
        run_(args=trial, reload_args=reload_args)


def run_crossval(args, paths_file, n_splits=10, launch_mode=None, mode='random', val_frac=0.3, tst_frac=0.1,
                 exclude_file=None, *other_args, **kwargs):
    with open(paths_file, 'rb') as f:
        paths = pkl.load(f)

    if exclude_file is None:
        with open(exclude_file, 'rb') as f:
            exclude = pkl.load(f)

        paths = [path for path in paths if (path[0] not in exclude and path[1] not in exclude)]

    nr_val_paths = int(len(paths) * val_frac)
    assert nr_val_paths > 0

    if mode == 'random':
        inds = np.arange(len(paths))
        val_files_permutations = []

        finished = False
        while not finished:
            np.random.shuffle(inds)
            permutation = tuple(inds[:nr_val_paths])
            if permutation not in val_files_permutations:
                val_files_permutations.append(permutation)

            if len(val_files_permutations) == n_splits:
                finished = True

    else:
        raise NotImplementedError

    args, parser, reload_args = parse(args=args)
    if launch_mode != 'slurm':
        for val_files in val_files_permutations:
            with mp.get_context('spawn') as ctx:
                args_ = update_namespace(args, dict(paths_file=paths_file, shuffle_data_module=False, val_files=val_files))
                p = ctx.Process(target=run_, kwargs=dict(args=args_, reload_args=reload_args))
                p.start()
                p.join()

    else:
        for val_files in val_files_permutations:
            args_ = update_namespace(args, dict(paths_file=paths_file, shuffle_data_module=False, val_files=val_files, val_frac=val_frac))
            run_with_slurm(args=args_,  *other_args, **kwargs)


def run_with_slurm(args=None, opt_lists=None, opt_ranges=None, strategy='random_search', nb_trials=1, **kwargs):
    args, parser, reload_args = parse_hypopt(args=args, opt_lists=opt_lists, opt_ranges=opt_ranges,
                                             strategy=strategy, process=True)
    job_name = '%s_%s' % (args.experiment, args.name)
    cluster = SlurmCluster(
        hyperparam_optimizer=args,
        # stest_tube_exp_name=job_name,
        log_path=args.slurm_out_path,
    )

    # Change script path to call this file directly
    cluster.script_name = __file__

    # see the output of the NCCL connection process
    cluster.add_command('source .../venv/bin/activate')

    # on JURECA by default CUDA_VISIBLE_DEVICES is restricted to one per process
    # however the PL framework needs to see all devices to build th AccelerarorConnector
    cluster.add_command('export CUDA_VISIBLE_DEVICES=%s' % ','.join(map(str, range(min(int(args.n_gpus) * int(args.n_nodes), 4)))))
    cluster.add_command('export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}')

    # ************** DON'T FORGET THIS ***************
    # MUST load the latest NCCL version
    # cluster.load_modules(['NCCL/2.4.7-1-cuda.10.0'])

    # configure cluster
    cluster.per_experiment_nb_nodes = args.n_nodes
    cluster.per_experiment_nb_gpus = args.n_gpus
    cluster.per_experiment_nb_cpus = args.n_cpus
    cluster.job_time = args.job_time

    cluster.add_slurm_cmd(cmd="ntasks-per-node",
                          value=str(cluster.per_experiment_nb_gpus),
                          comment="1 task per gpu, for ddp")

    cluster.add_slurm_cmd(cmd='account', value='dummy', comment='')
    cluster.add_slurm_cmd(cmd='partition', value='dummy', comment='')

    cluster.optimize_parallel_cluster_gpu(
        partial(run_, reload_args=reload_args),
        nb_trials=nb_trials,  # how many permutations of the grid search / random sampling to run
        job_name=job_name
    )


if __name__ == '__main__':
    run_directly()

