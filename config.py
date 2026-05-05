import os
from argparse import Action


def get_avail_batch_samplers():
    from fluomapper.utils.lightning_base import CustomDistributedSampler

    return dict(custom_distributed_batch_sampler=CustomDistributedSampler, none=None)


def get_avail_heads():
    from fluomapper.nn._base import heads

    return dict(poly=heads.PolynomialHead, two_gaussians=heads.TwoGaussians, evidential=heads.Evidential, none=None)


def get_avail_models():
    from fluomapper.nn.SFMNN.SFMNN import SFMNN
    from fluomapper.nn.FWDNN.FWDNN import FWDNN
    from fluomapper.nn.simulation_mlp.mlp import MLP
    from fluomapper.nn.combined import CombinedFWDNN
    
    avail_models = dict(mlp=MLP, 
                        sfmnn=SFMNN, explicit2=SFMNN,
                        fwdnn=FWDNN, 
                        combined_fwdnn=CombinedFWDNN)

    return avail_models


def get_avail_data_sources():
    import fluomapper.data.hyplant.hyplant as hyplant
    import fluomapper.data.desis.desis as desis
    import fluomapper.data.simulations.simulations as sim
    import fluomapper.data.flex.flex as flex

    avail_data_sources = dict(hyplant=hyplant.DataModule,
                              desis=desis.DESISModule,
                              hyplant_meta_var=hyplant.MetaVarDataModule,
                              desis_meta_var=desis.DESISMetaVarDataModule,
                              fluomap=sim.SimulationDmodule,
                              flex=flex.FLEXDmodule)

    return avail_data_sources


def get_avail_losses():
    from losses import losses
    import torch.nn.functional as F

    avail_losses = dict(mse=F.mse_loss, evidential=losses.EvidentialLoss,
                        var_weighted_mse=losses.var_weighted_mse, l1=F.l1_loss)
    return avail_losses


def get_default_parser():
    from argparse import ArgumentParser
    from os.path import join as pjoin
    FLUOMAP_DIR = os.path.dirname(os.path.abspath(''))

    parser = ArgumentParser(add_help=False, allow_abbrev=False)

    group_runspec = parser.add_argument_group('run_spec')
    group_runspec.add_argument('--train', type=int, default=1)
    group_runspec.add_argument('--validate', type=int, default=1)
    group_runspec.add_argument('--test', type=int, default=1)
    group_runspec.add_argument('--predict', type=int, default=1)
    group_runspec.add_argument('--dset', type=str)

    group_runspec.add_argument('--fast_dev_run', type=int, required=False, default=0)
    group_runspec.add_argument('--overfit_batches', type=int, required=False, default=0)
    group_runspec.add_argument('--auto_lr_find', type=int, default=None)

    group_general = parser.add_argument_group('general')
    group_general.add_argument('--name', type=str, required=True)
    group_general.add_argument('--experiment', type=str, required=True)
    group_general.add_argument('--tracking_uri', type=str, required=False,
                               default=pjoin(FLUOMAP_DIR, 'mlruns'), action=ExpandPathAction)
    group_general.add_argument('--slurm_out_path', type=str, required=False,
                               default=pjoin(FLUOMAP_DIR, 'slurm'), action=ExpandPathAction)
    group_general.add_argument('--accelerator', type=str, required=False, default='ddp',
                               choices=['ddp', 'dp', 'ddp2', 'ddp_spawn', 'horovod', 'none', 'None'], action=NoneAction)
    group_general.add_argument('--n_gpus', type=int, default=1)
    group_general.add_argument('--exact_gpus_specified', type=int, default=0)
    group_general.add_argument('--gpus', type=int, default=None, nargs='+')
    group_general.add_argument('--n_nodes', type=int, default=1)
    group_general.add_argument('--n_cpus', type=int, default=1)
    group_general.add_argument('--job_time', type=str, default='00:01:00')
    group_general.add_argument('--precision', type=int, default=32)
    group_general.add_argument('--sync_batchnorm', type=int, default=0)

    group_general.add_argument('--val_check_interval', type=float, required=False, default=1.0)
    group_general.add_argument('--limit_val_batches_perc', type=float, required=False, default=None)
    group_general.add_argument('--limit_val_batches_n', type=int, required=False, default=None)
    group_general.add_argument('--check_val_every_n_epoch', type=int, required=False, default=1)
    group_general.add_argument('--max_epochs', type=int, required=False, default=50)
    group_general.add_argument('--profiler', type=str, required=False, default=None, action=NoneAction)
    group_general.add_argument('--log_monitor_loss', type=str, default='val_loss')
    group_general.add_argument('--early_stopping', type=int, default=0)

    group_general.add_argument('--limit_train_batches_perc', type=float, required=False, default=None)
    group_general.add_argument('--limit_train_batches_n', type=int, required=False, default=None)

    group_general.add_argument('--top_k_ckpts', type=int, required=False, default=0)

    group_general.add_argument('--reload_dataloaders_every_n_epochs', type=int, default=0)

    group_networks = parser.add_argument_group('networks')
    group_networks.add_argument('--model', type=str, choices=list(get_avail_models().keys()), required=True)

    return parser


class ExpandPathAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) in (list, tuple):
            setattr(namespace, self.dest, [os.path.expandvars(v) for v in values])

        else:
            setattr(namespace, self.dest, os.path.expandvars(values))


class NoneAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values in ('none', 'None'):
            setattr(namespace, self.dest, None)

        else:
            setattr(namespace, self.dest, values)
