import sys


sys.path.append('.')

from fluomapper.scripts.emsfmnn_prediction.prediction_utils \
        import RUN_IDS, get_models, create_dm, run_prediction
from fluomapper.scripts.emsfmnn_prediction.geometry_utils import create_geometry_files
from fluomapper.scripts.emsfmnn_prediction.prediction_utils import load_paths_file

import os
from os.path import join as pjoin
import pickle as pkl

import argparse
import yaml, glob, shutil


def run(cfg):
    model_ids = cfg['model']['ids']
    mlflow_backend_uri = cfg['model']['mlflow_backend_uri']
    fluomap = cfg['model']['fluomap']

    fwd_model_file = fluomap['fwd_model_file']
    fwd_sensitivity_file = fluomap['fwd_sensitivity_file']

    paths_file = cfg['input']['paths_file']
    meta_vars_dir = cfg['input']['meta_vars_dir']
    data_volume = cfg['input']['data_volume']
    dem_file = cfg['input']['dem_file']

    save_dir = cfg['output']['save_dir']
    dirname = cfg['output']['dirname']

    run_cfg = cfg['run']
    device = run_cfg['device']
    return_output = run_cfg['return_output']
    write = run_cfg['write']
    overwrite = run_cfg['overwrite']
    overwrite_geometry = run_cfg['overwrite_geometry']
    model_verbose = run_cfg['model_verbose']
    do_create_geometry_files = run_cfg['create_geometry_files']
    use_parallel = run_cfg['use_parallel']
    num_workers = run_cfg['num_workers']
    verbose = run_cfg['verbose']
    load = getattr(run_cfg, 'load', True)

    # prediction setup
    files, _ = load_paths_file(paths_file, remove=False)
    run_ids = [RUN_IDS[model_id] if model_id in RUN_IDS else model_id for model_id in model_ids]

    preload_to = cfg['input'].get('preload_to', None)
    if preload_to is not None:
        print('PRELOADING TO:', preload_to)
        all_copied_files = []
        for fil in files:
            file = pjoin(data_volume, fil)
            new_data_volume = preload_to

            all_this_file = glob.glob(file[:-len('_radiance.dat')] + '*')

            copied_files = []
            for f in all_this_file:
                if f.endswith('raw') or 'GLT' in f or 'rect' in f or 'deconv' in f:
                    continue

                new_file = pjoin(new_data_volume, *f.split(os.sep)[len(data_volume.split(os.sep)):])
                os.makedirs(os.path.dirname(new_file), exist_ok=True)
                if not os.path.exists(new_file):
                    shutil.copy(f, new_file)
                copied_files.append(new_file)

            all_copied_files.append(copied_files)

        orig_data_volume = data_volume
        data_volume = new_data_volume

    else:
        orig_data_volume = data_volume
        all_copied_files = None

    # run geometry files creation
    if do_create_geometry_files:
        print(f'############ CREATING GEOMETRY FILES in {meta_vars_dir}')
        create_geometry_files(files, dem_file, meta_vars_dir, data_path=orig_data_volume, use_parallel=use_parallel,
                              num_workers=num_workers, overwrite=overwrite_geometry)

    # prediction setup
    print(f'############ SETTING UP READERS AND MODELS')
    meta_vars_volume_, meta_vars_dir_ = os.path.dirname(meta_vars_dir), os.path.basename(meta_vars_dir)
    models = get_models(run_ids, paths_file, mlflow_backend_uri,
                        fwd_model_file, fwd_sensitivity_file,
                        meta_vars_volume=meta_vars_volume_, meta_vars_dir=meta_vars_dir_,
                        data_volume=data_volume, verbose=verbose)
    dms = [create_dm(paths_file, id_, meta_vars_volume=meta_vars_volume_, meta_vars_dir=meta_vars_dir_,
                     data_volume=data_volume) for id_ in range(len(files))]

    # run prediction
    print(f'############ RUNNING MODEL INFERENCE')
    run_prediction(models, dms, overwrite=overwrite,
                   device=device, save_dir=save_dir,
                   dirname=dirname, return_output=return_output,
                   write=write, model_verbose_name=model_verbose, 
                   load=load, delete_preloaded=all_copied_files, )


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





