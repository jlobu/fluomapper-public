import sys
sys.path.append('.')

from fluomapper.scripts.emsfmnn_prediction.prediction_utils \
        import RUN_IDS, get_models, run_prediction
from fluomapper.scripts.emsfmnn_prediction.geometry_utils import create_geometry_files
from fluomapper.scripts.emsfmnn_prediction.prediction_utils import load_paths_file
from fluomapper.run import run_directly, run_with_hypopt, run_with_slurm

import os
from os.path import join as pjoin
import pickle as pkl

import argparse
import yaml
import shutil
import glob

import numpy as np


def run(cfg):
    paths_file = cfg['paths_file']
    meta_vars_dir = cfg['meta_vars_dir']
    data_volume = cfg['data_volume']
    base_ft_cfg = cfg['base_finetune_config']
    preload_to = cfg['preload_to']

    geom_cfg = cfg['geometry_files']
    dem_file = geom_cfg['dem_file']
    num_workers = geom_cfg['num_workers']
    do_create_geometry_files = geom_cfg['create_geometry_files']
    overwrite = geom_cfg['overwrite']
    use_parallel = geom_cfg['use_parallel']

    ft = cfg['finetune']
    ft['paths_file'] = paths_file
    
    files, _ = load_paths_file(paths_file) 
    if ft['files_range'] is not None:
        file_ids = list(range(*ft['files_range']))

    else:
        file_ids = list(range(len(files)))

    files = np.asarray(files)[file_ids]
    if preload_to is not None:
        for fil in files:
            file = pjoin(data_volume, fil)
            new_data_volume = preload_to  

            all_this_file = glob.glob(file[:-len('_radiance.dat')] + '*')

            for f in all_this_file:
                if f.endswith('raw') or 'GLT' in f or 'rect' in f or 'deconv' in f:
                    continue

                new_file = pjoin(new_data_volume, *f.split(os.sep)[len(data_volume.split(os.sep)):])
                os.makedirs(os.path.dirname(new_file), exist_ok=True)
                if not os.path.exists(new_file): 
                    shutil.copy(f, new_file)

        orig_data_volume = data_volume
        data_volume = new_data_volume

    else:
        orig_data_volume = data_volume

    # run geometry files creation
    if do_create_geometry_files:
        print(f'############ CREATING GEOMETRY FILES in {meta_vars_dir}')
        files = list(files)
        create_geometry_files(files, dem_file, meta_vars_dir,
                              data_path=orig_data_volume,
                              use_parallel=use_parallel,
                              num_workers=num_workers,
                              overwrite=overwrite)

    # create config
    meta_vars_volume_, meta_vars_dir_ = os.path.dirname(meta_vars_dir), os.path.basename(meta_vars_dir)
    
    if base_ft_cfg is not None:
        ft_ = load_config(base_ft_cfg)
    else:
        ft_ = dict()

    ft_.update(ft)
    ft_['meta_vars_dir'] = meta_vars_dir_
    ft_['meta_vars_volume'] = meta_vars_volume_
    ft_['volume'] = data_volume

    if 'train_files' in ft:
        ft_['train_files'] = ft['train_files']
        ft_['val_files'] = ft['val_files']

    else:
        ft_['train_files'] = file_ids
        ft_['val_files'] = file_ids
    
    tracking_uri = ft_['tracking_uri']
    print(f'############ STARTING TRAINING RUN, LOGGING TO {tracking_uri}')

    run_directly(ft_)


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()

