import sys

from fluomapper.data.simulations.simulations import SimulationDmodule
from fluomapper.data.hyplant.hyplant import DataModule, MetaVarDataModule

from fluomapper.utils.func.interp1d import Interpolate
from fluomapper.utils.gaussian import Resampler
from fluomapper.utils.data import search_spectral_window, select, load_mat_paths

from fluomapper.tracking.loggers import QuantizedConfusionMatrix
from fluomapper.run import reload_model
from fluomapper.ui import get_hparams, get_data_module, get_run

import pvlib, pytz, datetime
from pvlib import solarposition, irradiance
import geopandas as gpd

import os, re, tqdm, glob, xml
from os.path import join as pjoin
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import xarray as xr
import pandas as pd
import h5py, torch
import rasterio as rio

import pytorch_lightning as pl

from collections import OrderedDict
from scipy.interpolate import interp1d

import pickle as pkl
import subprocess as sp

torch.multiprocessing.set_sharing_strategy('file_system')

from rasterio import logging
log = logging.getLogger()
log.setLevel(logging.ERROR)

import shutil

RUN_IDS = dict(CKA350='6e19c632c7e04f1695ca9c4193d19c94', # CKA2020-600-EmSFMNN
               CKA600='f730b9582bdd46efb57af69ab1274776',  # CKA2020-350-EmSFMNN
               WST1500='a6853bf715294a5392ba4071b2397d33', # WST-1500-EmSFMNN
               BASE='8407f8645add41b499620ad280d34786', # Backbone
               TOPO='b5f9611399184e17a85ae9034b708bca'  # Topo
               )


def write_tif(out_path, arr, dtype=rio.uint8, band_names=None, **kwargs):
    if len(arr.shape) == 2:
        arr = arr[None]

    profile = dict(transform=None)
    profile.update({
        'driver': 'GTiff',
        'dtype': dtype,
        'height': arr.shape[1],
        'width': arr.shape[2],
        'count': arr.shape[0],
        'interleave': 'band'})
    profile.update(kwargs)

    with rio.open(out_path, 'w', **profile) as dst:
        for i, band in enumerate(arr, start=1):
            dst.write(band, i)

            if band_names is not None:
                dst.set_band_description(i, band_names[i-1])


def bayesian_eval(model, N=10, *args, **kwargs):
    model.eval()
    model.set_dropout_to_train()

    preds, pred_keys = predict(model, *args, **kwargs)

    preds = [preds['pred']] + [predict(model, *args, **kwargs)[0]['pred'] for _ in range(N - 1)]

    for pred in preds:
        if 'pred_fwd_scale' in pred:
            del pred['pred_fwd_scale']

    keys = preds[0].keys()
    preds = dict([(key, torch.stack([p[key] for p in preds], dim=0).float()) for key in keys])

    pred_mean = dict([(p, val.mean(0)) for p, val in preds.items()])
    pred_std = dict([(p + '_std', val.std(0)) for p, val in preds.items()])

    # x, obs, meta = model(batch)[1:]

    model.eval()
    return pred_mean, pred_std, pred_keys


def predict(model, batch, y, sid, scalar_vars, dset_vars, obs_window, device='cpu', **kwargs):
    add_vars = []

    inp = batch['obs'].clone()

    if type(obs_window) in (tuple, list):
        obs_window = slice(*obs_window)

    batch['obs'] = batch['obs'][:, obs_window]

    if sid is not None:
        batch['source_id'] = torch.ones(batch['source_id'].shape).long() * sid

    sc_vars = dict()
    for key in scalar_vars:
        sc_vars[key] = torch.ones((batch['obs'].shape[0],
                                   1,
                                   batch['obs'].shape[2],
                                   batch['obs'].shape[3])) * scalar_vars[key]

    batch.update(sc_vars)

    # if model.model.with_ids and p in model.model.data_sources:
    #    idx = model.model.data_sources.index(p)
    #    source_id = model.model.data_source_id_dict[idx]

    new_batch, x_, y = model.prepare_batch((batch, y))
    new_batch = new_batch.float().to(device)

    for key in y.keys():
        if model.model.pass_meta_vars_to_fwd:
            if key == 'source_id':
                y[key] = y[key].to(device)

            elif y[key] is not None:
                y[key] = y[key].float().to(device)

            else:
                y[key] = torch.ones(y[list(y.keys())[0]].shape, device=device)

    ypred = model(new_batch, **y)
    ats, _, _ = model.model.simulate_ats(ypred['pred'], **y)

    residuals = torch.abs(ats - select(inp, model.pred_window, axis=1).to(device=device))
    mae = residuals.mean(dim=1)

    rmae = (residuals / ats)
    rmae_fwindow = select(rmae, windows=model.sif_focus_window, axis=1).mean(dim=1)
    rmae = rmae.mean(axis=1)

    ypred['pred'].update(mae=mae, sif=y['sif'].mean(dim=1), ats=ats, rmae=rmae, rmae_fwindow=rmae_fwindow,
                         mask=ypred['mask'])
    add_vars.append('mae')
    add_vars.append('sif')
    add_vars.append('ats')
    add_vars.append('rmae')
    add_vars.append('rmae_fwindow')
    add_vars.append('mask')

    for var in dset_vars:
        ypred['pred'].update({var: y[var]})

    del new_batch
    torch.cuda.empty_cache()

    ypred['pred'].update(dict([(key, val.cpu()) for key, val in y.items() if key not in ypred['pred']]))

    for key in ypred.keys():
        if type(ypred) is torch.Tensor:
            ypred[key] = ypred[key].to('cpu')

    pred_keys = model.model.fwd_inputs + add_vars + dset_vars

    if 'pred_fwd_scale' in ypred:
        del ypred['pred_fwd_scale']

    return ypred, pred_keys


def assemble(model, pred, pred_keys, shape, orig_shape, keep_reconstructed=False, assemble_non_constant_shifts=False):
    shape = tuple([len(pred_keys)] + list(shape))
    image = np.zeros(shape)

    fwhm_var = None
    cw_var = None
    rec = None

    batch_size = None
    for k, key in enumerate(pred_keys):

        if key.startswith('fwhm') and not model.model.constant_fwhm_shift and assemble_non_constant_shifts:
            shape_ = list(shape)
            shape_[0] = pred[0][key].shape[1]
            shape_ = tuple(shape_)

            fwhm_var = np.ones(shape_)

        if key.startswith('CW') and not model.model.constant_cw_shift and assemble_non_constant_shifts:
            shape_ = list(shape)
            shape_[0] = pred[0][key].shape[1]
            shape_ = tuple(shape_)

            cw_var = np.ones(shape_)

        if key.startswith('ats') and keep_reconstructed:
            shape_ = list(shape)
            shape_[0] = pred[0]['ats'].shape[1]
            shape_ = tuple(shape_)

            rec = np.ones(shape_)

        for i, b in enumerate(pred):
            b = b[key].squeeze(1)

            if batch_size is None:
                batch_size = b.shape[0]

            window_shape = b.shape[1:]
            for j, s in enumerate(b):
                s = s.cpu().numpy()

                try:
                    nr_windows_per_line = int(image.shape[1] / window_shape[-2])

                except Exception as e:
                    print(window_shape, key)
                    raise e

                x = int(((i * batch_size + j) % nr_windows_per_line) * window_shape[-2])
                y = int(((i * batch_size + j) // nr_windows_per_line) * window_shape[-1])

                if ((key.startswith('fwhm') and not model.model.constant_fwhm_shift) or (
                        key.startswith('CW') and not model.model.constant_cw_shift)) and assemble_non_constant_shifts:
                    arr = fwhm_var if key == 'fwhm' else cw_var
                    arr[:, x: x + win.shape[0], y: y + win.shape[1]] = s[:, :window_shape[-1] - max(0, x + window_shape[
                        -1] - image.shape[1]),
                                                                       :window_shape[-2] - max(0, y + window_shape[-2] -
                                                                                               image.shape[2])]

                if key.startswith('ats') and keep_reconstructed:
                    rec[:, x: x + win.shape[0], y: y + win.shape[1]] = s[:, :window_shape[-1] - max(0, x + window_shape[
                        -1] - image.shape[1]),
                                                                       :window_shape[-2] - max(0, y + window_shape[-2] -
                                                                                               image.shape[2])]

                if len(s.shape) == 3:
                    s = s[s.shape[0] // 2]

                win = s[:window_shape[-1] - max(0, x + window_shape[-1] - image.shape[1]),
                      :window_shape[-2] - max(0, y + window_shape[-2] - image.shape[2])]
                image[k, x: x + win.shape[0], y: y + win.shape[1]] = win

                # else:
                #    break

    image = xr.DataArray(image,
                         coords={'x': np.arange(image.shape[1]),
                                 'y': np.arange(image.shape[2]),
                                 'variable': pred_keys},
                         dims=["variable", "x", "y"])

    image = image[:, :orig_shape[0], :orig_shape[1]]
    image = image.transpose('variable', 'y', 'x')

    if rec is not None:
        rec = rec[:, :orig_shape[0], :orig_shape[1]]
        rec = rec.transpose(0, 2, 1)

    return image, rec, fwhm_var, cw_var


def iterate(models, dms, device='cpu',
            write=True, base_path=None, dirname=None, scalar_vars=None, source_id=None, overwrite=False,
            dset_vars=[], keep_reconstructed=False, assemble_non_constant_shifts=True, do_mcdropout=False,
            N=10, save_dir_w_epoch=True, out_bands=None, return_output=True, obs_window=(0, 1023), 
            model_verbose_name=None, load=False, delete_preloaded=None,):
    out_dict = dict()

    for i, (run_id, model) in enumerate(models.items()):
        model.to(device)
        out_dict[run_id] = dict(image=[], fwhm_var=[], cw_var=[], name=[], rec_image=[])

        source_id = [source_id] * len(dms) if type(source_id) in (int, str) else source_id
        for d, dm in enumerate(dms):
            if delete_preloaded is not None:
                to_be_deleted = delete_preloaded[d]

            else:
                to_be_deleted = None

            if save_dir_w_epoch:
                verbose = os.path.basename(model.loaded_checkpoint).split('-')[0]
                if model_verbose_name is not None:
                    verbose = f'{model_verbose_name[i]}_{verbose}'
                verbose = f'__{verbose}'

            else:
                if model_verbose_name is not None:
                    verbose = f'__{model_verbose_name[i]}'
                
                else:
                    verbose = ''

            save_path = pjoin(base_path, dirname, run_id + verbose)
            name = os.path.basename(dm.paths[dm.val_files[0]][0])[:-4]
            save_path = pjoin(save_path, name + '_EmSFMNN.tif')

            print('PROCESSING ', save_path)
            if os.path.exists(save_path) and not overwrite:
                continue

            dl = get_dl(dm, load=load)

            shape = dl.dataset.readers[0].sources[0].shape[:-1]
            orig_shape = dl.dataset.readers[0].sources_meta[0].attrs['orig_shape'][:2]

            sid = source_id[d] if source_id is not None else dl.dataset.readers[0].source_ids[0]

            if hasattr(model.model, 'data_sources') and hasattr(model.model, 'ids'):

                ckpt = torch.load(model.loaded_checkpoint, map_location=model.device, weights_only=False)
                try:
                    ind = list(ckpt['state_dict'][f'model.data_source_ids']).index(sid)
                    id_ = ckpt['state_dict'][f'model.ids'][[ind]]
                    print('Loading ID ', ind)

                except Exception as e:
                    dids = ckpt['state_dict'][f'model.data_source_ids'].cpu().numpy()
                    closest_id = np.argmin(np.abs(dids - sid))

                    sid = dids[closest_id].item()
                    id_ = ckpt['state_dict'][f'model.ids'][[closest_id]]
                    print(f'Could not load ID', e, f'Using {sid} instead')

                    # continue

                model.model.ids = torch.nn.Parameter(ckpt['state_dict'][f'model.ids'].to(model.device),
                                                     requires_grad=False)
                model.model.data_source_id_dict = dict(
                    [(ckpt['state_dict'][f'model.data_source_ids'][i].item(), i) for i in
                     range(model.model.ids.shape[0])])

            p = os.path.basename(dl.dataset.paths[0][0])

            # get scalar_vars
            scalar_vars_ = dict()
            if scalar_vars is not None:
                for key, var in scalar_vars.items():
                    if type(var) in (float, int):
                        scalar_vars_[key] = var

                    else:
                        scalar_vars_[key] = var(p)

            if not do_mcdropout:
                pred = []
                for batch, y in tqdm.tqdm(iter(dl)):
                    ypred, pred_keys = predict(model, batch=batch, y=y, sid=sid, scalar_vars=scalar_vars_,
                                               dset_vars=dset_vars, obs_window=obs_window, device=device)
                    pred.append(ypred['pred'])

                image, rec, fwhm_var, cw_var = assemble(model, pred, pred_keys, shape, orig_shape,
                                                        keep_reconstructed=keep_reconstructed,
                                                        assemble_non_constant_shifts=assemble_non_constant_shifts)

            else:
                pred_mean = []
                pred_std = []

                for batch, y in tqdm.tqdm(iter(dl)):
                    ypred_mean, ypred_std, pred_keys = bayesian_eval(model, batch=batch, y=y, sid=sid,
                                                                     scalar_vars=scalar_vars_, dset_vars=dset_vars, 
                                                                     obs_window=obs_window, device=device, N=N)
                    pred_mean.append(ypred_mean)
                    pred_std.append(ypred_std)

                image_mean, rec, fwhm_var, cw_var = assemble(model, pred_mean, pred_keys, shape=shape,
                                                             orig_shape=orig_shape,
                                                             keep_reconstructed=keep_reconstructed,
                                                             assemble_non_constant_shifts=assemble_non_constant_shifts)

                image_std, rec_std, fwhm_std, cw_std = assemble(pred_std, [p + '_std' for p in pred_keys],
                                                                shape=shape, orig_shape=orig_shape,
                                                                keep_reconstructed=False,
                                                                assemble_non_constant_shifts=False)

                image = xr.concat([image_mean, image_std], dim='variable')
                if rec is not None:
                    rec = np.stack([rec, rec_std], axis=-1)
                if fwhm_var is not None:
                    fwhm_var = np.stack([fwhm_var, fwhm_std], axis=-1)
                if cw_var is not None:
                    cw_var = np.stack([cw_var, cw_std], axis=-1)

            if out_bands is not None:
                assert np.all([band in pred_keys for band in out_bands]), f'No all bands {out_bands} are in {pred_keys}'
                image = image.loc[out_bands]
                pred_keys = out_bands

            if return_output:
                out_dict[run_id]['image'].append(image)
                out_dict[run_id]['fwhm_var'].append(fwhm_var)
                out_dict[run_id]['cw_var'].append(cw_var)
                out_dict[run_id]['name'].append(os.path.basename(save_path))
                out_dict[run_id]['rec_image'].append(rec)

            if write:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # image.to_netcdf(pjoin(save_path, 'out.nc'))
                write_tif(save_path, image.astype(np.float32), dtype=rio.float32, band_names=pred_keys)

                if cw_var is not None:
                    np.save(pjoin(save_path[:-len('_EmSFMNN.tif')] + '_cw_var.npy'), cw_var)

                if fwhm_var is not None:
                    np.save(pjoin(save_path[:-len('_EmSFMNN.tif')] + 'fwhm_var.npy'), fwhm_var)

                if rec is not None:
                    np.save(pjoin(save_path[:-len('_EmSFMNN.tif')] + 'rec_image.npy'), rec)

            if to_be_deleted is not None:
                for file in to_be_deleted:
                    if not os.path.ismount(file):
                        os.remove(file)
                        print('NOW removing', file)
    if return_output:
        return out_dict


def create_dm(paths_file, val_file_id, meta_vars_volume, meta_vars_dir, data_volume, dm_kwargs=None):
    args = dict(paths_file=paths_file,
                val_files=[val_file_id],
                train_files=[val_file_id],
                batch_size=1,
                num_workers=10,
                reader_batch_size=1,
                out_shape=60,
                out_stride=60,
                out_mode='windows',
                load_val=True,
                load_train=False,
                load=False,
                spectral_window_obs_wvl=None,
                spectral_window_label_wvl=(760,),
                shuffle=False,
                shuffle_reading=False,
                path_ids=[val_file_id],
                edge_mode='mirror',
                )

    args.update(meta_vars=['dem_sensor', 'scene_incidence_sensor', 'parm1_sensor'],
                meta_info=['sza', 'off_nadir', 'alt', 'dist', 'ndvi'],
                meta_vars_volume=meta_vars_volume,
                meta_vars_dir=meta_vars_dir,
                volume=data_volume,
                path_ids=['datetime'],
                reader_batch_size=1, batch_size=1,
                )

    if dm_kwargs is not None:
        args.update(dm_kwargs)

    dm = MetaVarDataModule(**args)
    return dm


def run_prediction(models, dls, device, save_dir, dirname, model_verbose_name=None, 
                   write=True, overwrite=False, return_output=True, load=False, delete_preloaded=None):
    out_dict = iterate(models, dls,
                       device=device,
                       write=write, base_path=save_dir, dirname=dirname,
                       overwrite=overwrite, keep_reconstructed=False,
                       assemble_non_constant_shifts=False,
                       save_dir_w_epoch=False,
                       dset_vars=['ndvi', 'off_nadir'],
                       out_bands=['tilt', 'parm1', 'parm2', 'h1alt', 'h2alt', 'neg_AOT', 'h2ostr', 'rho', 'rho_slope',
                                  'e', 'f', 'mae', 'rmae', 'rmae_fwindow', 'ndvi'],
                       return_output=return_output, 
                       obs_window=(0, 1023), 
                       model_verbose_name=model_verbose_name, 
                       load=load,
                       delete_preloaded=delete_preloaded
                       )

    return out_dict


def list_to_txt(path, items):
    """
    Writes a list of strings to a text file, one item per line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(str(item) + "\n")


@contextmanager
def suppress_stdout(verbose=False):
    if verbose:
        yield

    else:
        saved_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stdout = saved_stdout


def get_model(run_id, dm_kwargs_add, reload_kwargs_add, verbose=False):
    reload_kwargs = dict(run_id=run_id,  
                     reload_hparams=True,
                     reload_ckpt="last", 
                     load=False,
                     rho_l2a_distort=None,
                     kneighbours_on_setup=False)
    
    reload_kwargs.update(reload_kwargs_add)
    
    dm_kwargs = dict([])
    dm_kwargs.update(dm_kwargs_add)
    
    with suppress_stdout(verbose=verbose):
        model = reload(reload_kwargs, dm_kwargs)
        model = model.eval()

    return model
    

def get_models(run_ids, paths_file, mlflow_backend_uri, fwd_model_file, fwd_sensitivity_file,
               meta_vars_volume, meta_vars_dir, data_volume, verbose=False):
    models = dict()
    
    for run_id in run_ids:
        for ckpt_id in ('best',):
            reload_kwargs_add = dict(paths_file=paths_file,
                                     mlflow_backend_uri=mlflow_backend_uri,
                                     train_files=[0], #this is just for loading, arbitrary
                                     val_files=[0], # this is just for loading, arbitrary
                                     reload_ckpt=str(ckpt_id), 
                                     fwd_model_file=fwd_model_file,
                                     fwd_corrector_model_file='none',
                                     fwd_sensitivity_corrector_model_file=fwd_sensitivity_file,    
                                    )
            
            dm_kwargs_add = dict(meta_vars_volume=meta_vars_volume,
                                 meta_vars_dir=meta_vars_dir,
                                 spectral_window_label_wvl=760,
                                 load_ids_from_ckpt=False,
                                 volume=data_volume,
                                )
            models[run_id] = get_model(run_id, reload_kwargs_add=reload_kwargs_add, dm_kwargs_add=dm_kwargs_add, 
                                       verbose=verbose)

    return models


def reload(reload_kwargs, dm_kwargs):
    torch.set_grad_enabled(False)
    pl.seed_everything(42)

    hparams = get_hparams(**reload_kwargs)
    hparams = vars(hparams)

    hparams.update(dm_kwargs)
    hparams.update(reload_kwargs)

    _, model = reload_model(args=hparams, with_dm=True)
    return model


def get_dl(dm, load=False):
    dm.load_val = load
    dm.prepare_data()
    dm.setup(load=load)
    dl = dm.val_dataloader()

    return dl


def load_paths_file(paths_file, new_paths_file=None, data_volume=None, remove=False):
    if paths_file.endswith('pkl'):
        with open(paths_file, 'rb') as f:
            files = pkl.load(f)

    elif paths_file.endswith(".txt"):
        with open(paths_file, "r") as f:
            files = [line.strip() for line in f if line.strip()]

    else:
        raise NotImplementedException()

    if type(files[0]) is str:
        if files[0].startswith('processed') and remove:
            n = 1
        else:
            n = 0

        files = [pjoin(*p.split(os.sep)[n:]) for p in files]

    else:
        if files[0][0].startswith('processed') and remove:
            n = 1
        else:
            n = 0

        files = [pjoin(*p[0].split(os.sep)[n:]) for p in files]


    if data_volume is not None:
        if type(files[0]) is str:
            files = [pjoin(f'{data_volume}', p) for p in files]
        else:
            files = [pjoin(f'{data_volume}', p[0]) for p in files]
    
    if new_paths_file is not None:
        list_to_txt(new_paths_file, files)

    return files, new_paths_file


