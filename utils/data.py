import numpy as np
import os
from os.path import join as pjoin
import rasterio

import rioxarray as rx
from rasterio.transform import Affine

import torch
import copy

from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator

from fluomapper.utils.run import chunk_list

from rasterio import logging
log = logging.getLogger()
log.setLevel(logging.ERROR)

import warnings
warnings.simplefilter("ignore")

import struct
from types import SimpleNamespace


DATA_TO_FFS_NAMES = dict(sza=('parm2', 1),
                         rel_azimuth=('parm1', 1), parm1_sensor=('parm1', 1),
                         aoi_sensor=('parm2', 1), aoi=('parm2', 1),
                         dem_sensor=("h2alt", 1e-3), dem=("h2alt", 1e-3), demadapted_sensor=("h2alt", 1e-3),
                         h1alt=('h1alt', 1e-3), alt=('h1alt', 1e-3),
                         tilt_sensor=('tilt', 1), scene_incidence_sensor=('tilt', 1),
                         scene_incidence_adapted_sensor=('tilt', 1)
                         )


def to_ffs_names(data, remove_unknown=False):
    default = lambda param, dv=None: param if not remove_unknown or not type(param) is str else dv
    get_ffs_name = lambda param, dv=None: DATA_TO_FFS_NAMES.get(param, [default(param, dv)])[0]
    get_ffs_norm = lambda param: DATA_TO_FFS_NAMES.get(param, [1, 1])[1]

    if type(data) in (list, tuple):
        ret = [get_ffs_name(param) for param in data]
        return [r for r in ret if r is not None]

    elif type(data) is dict:
        ret = dict([(get_ffs_name(param, '__'), get_ffs_norm(param) * val) if val is not None
                     else (get_ffs_name(param, '__'), None) 
                     for param, val in data.items()
                    ])
        return dict([(key, val) for key, val in ret.items() if key != '__'])

    else:
        return get_ffs_name(data)


def desis_to_world_coords(arr, geolayer_path, rect_path=None, res=30, return_transform=False):
    
    if len(arr.shape) == 2:
        arr = arr[None]
        
    lon, lat, alt = read_bil(geolayer_path)[0]

    x = lon.flatten()
    y = lat.flatten()
    z = arr.flatten()

    if rect_path is not None:
        rect = rx.open_rasterio(rect_path) 
        new_shape = [arr.shape[0]] + list(rect.shape[1:])

        new_x = rect.coords['x'].data
        new_y = rect.coords['y'].data

        new_x_, new_y_ = np.meshgrid(new_x, new_y)

    else:
        new_x = np.arange(np.min(lon), np.max(lon), res)
        new_y = np.arange(np.min(lat), np.max(lat), res)
        new_x_, new_y_ = np.meshgrid(new_x, new_y)

        new_shape = [arr.shape[0]] + list(new_x_.shape)
    
    interpolator = LinearNDInterpolator(list(zip(x, y)), z)
    interpolated = interpolator(new_x_, new_y_).reshape(new_shape)

    if return_transform:
        xres = np.diff(new_x[:2])
        yres = np.diff(new_y[:2])
        transform = Affine.translation(new_x[0] - xres / 2, new_y[0] - yres / 2) * Affine.scale(xres, yres)
        return interpolated, transform

    return interpolated


def zero_nonfinite(x, _isfinite=None, **kwargs):
    if _isfinite is None:
        isfinite_patch = torch.isfinite(torch.mean(x, dim=[1, 2, 3]))
        isfinite = torch.isfinite(x)
        isfinite[~isfinite_patch] = False

    else:
        isfinite, isfinite_patch = _isfinite

    with torch.no_grad():    
        xmean = torch.nanmean(x, dim=0)
        xmean = xmean if torch.all(torch.isfinite(xmean)) else 0

        x[~isfinite_patch] = xmean

    for key, val in kwargs.items():
        if torch.is_tensor(val) and len(val.shape) == 4 and torch.is_floating_point(val):
            kwargs[key] = zero_nonfinite(val, _isfinite=(isfinite, isfinite_patch), kwargs=None)[0]

    return x, torch.any(~isfinite, axis=1), kwargs


def load_mat_paths(base_path):
    if type(base_path) is list:
        paths = base_path
    else:
        paths = [os.path.join(base_path, p) for p in os.listdir(os.path.join(base_path))]

    params = {}
    for fil in paths:
        if fil.endswith('mat'):
            mat = loadmat(fil, mat_dtype=False)
            keys = [k for k in mat.keys() if not k.startswith('__')]

            for k in keys:
                inarr = mat[k]
                if inarr.dtype.names is not None:
                    inarr = convert_structured_array_to_dict(inarr)
                else:
                    inarr = inarr.astype(np.float64)
                params.update({k.lower(): inarr})

    return params


def convert_structured_array_to_dict(sarr):
    dic = {}
    for name in sarr.dtype.names:
        dic[name] = sarr[name][0][0]
    return dic


def to_pairs(it):
    it = chunk_list(it, 2)

    # remove empty seqs
    new_it = []
    for pair in it:
        if len(pair) == 2 and pair[0] != pair[-1]:
            new_it.append(pair)

        elif len(pair) == 1:
            new_it.append(pair)

        else:
            pass

    it = new_it
    return it


def search_spectral_window(*wvls, where, invert=False, pairs=True, smallest=1):
    where = where.squeeze()

    if torch.is_tensor(where):
        where = where.cpu().numpy()

    assert len(where.shape) <= 1
    if smallest == 1:
        pos = [np.argmin(np.abs(where - float(wvl))) for wvl in wvls]

    else:
        pos = [np.argpartition(np.abs(where - float(wvl)), k=smallest) for wvl in wvls]

    if pairs:
        assert smallest == 1
        if invert:
            pos = np.r_[0, pos, len(where) - 1]

        pos = to_pairs(pos)

    return pos


def select(signals, windows, axis=-1, inclusive=True):
    
    if windows is None or windows[0] is None:
        return signals

    if type(signals) is not torch.Tensor:
        if type(windows[0]) is slice:
            return np.concatenate([np.take(signals, indices=np.arange(signals.shape[axis])[win], axis=axis)
                                   for win in windows], axis=axis)

        return np.concatenate([np.take(signals, indices=range(slic[0], slic[-1] + inclusive), axis=axis)
                               for slic in windows], axis=axis)

    else:
        axis = np.arange(len(signals.shape))[axis]

        inds = [[slice(0, signals.shape[i]) if i != axis else slice(slic[0], slic[-1] + inclusive)
                 for i in range(len(signals.shape))]
                for slic in windows]
        return torch.cat([signals[ind] for ind in inds], dim=axis)


def permute_channels(out, reverse=False):
    """
    Move last dimension to dim=1. With reverse, move dim=1 to last dimension

    :param out:
    :return:
    """
    permute = list(range(len(out.shape)))

    if not reverse:
        permute.insert(-3, permute[-1])
        del permute[-1]
        out = out.permute(*permute)

    else:
        permute.append(1)
        del permute[1]
        out = out.permute(*permute)

    return out


class XarrayMeta(object):
    def __init__(self, arr):
        self.coords = copy.deepcopy(arr.coords)
        self.attrs = copy.deepcopy(arr.attrs)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_rasterio(path, load=False, window=None, band_indices=None):
    if window is None:
        with warnings.catch_warnings():
            source =  rx.open_rasterio(path, cache=True, chunks='auto')
            source = source.assign_attrs(path=path)
            source_meta = XarrayMeta(source)

            arr = source

            if load:
                arr = arr.to_numpy()
                source.close()

        return arr, source_meta

    else:
        with rasterio.open(path) as src:
            win = src.read(
                indexes=band_indices,  # bands
                window=window  # spatial slice
            )
        return win

def logical_or(*args):
    res = args[0]
    for arg in args[1:]:
        res = torch.logical_or(res, arg)

    return res


def read_bil(path, rows=None, cols=1024, channels=3, nodata=-9999, get_channel=None, min_=None, max_=None, 
             move_to_non_neg=False, format_string='<%df', **kwargs):

    with open(path, 'rb') as bil_f:
        bil_data = bil_f.read()
        
        if rows is None:
            rows = len(bil_data) // 4 // cols // channels

        # Unpack binary data into a flat tuple z
        s = format_string % (int(cols * rows * channels),)
        z = struct.unpack(s, bil_data)

        arr = np.zeros((channels, rows, cols), dtype=np.float32)
        for r in range(0, rows):
            for b in range(0, channels):
                for c in range(0, cols):
                    arr[b, r, c] = float(z[r * channels * cols + b * cols + c])

        arr[arr == nodata] = np.nan

    if get_channel is not None:
        arr = arr[[get_channel]]

    if move_to_non_neg:
        for channel in range(arr.shape[0]):
            arr -= min(0, np.min(arr))

    if min_ is not None:
        arr = np.clip(arr, min_, None)

    if max_ is not None:
        arr = np.clip(arr, None, max_)

    return arr, SimpleNamespace(attrs=dict(path=path), coords=dict([]))


def open_source(path, driver='rasterio', **kwargs):
    if driver == 'bil':
        return read_bil(path, **kwargs)
    
    elif driver == 'rasterio':
        return read_rasterio(path, **kwargs)
    
    else:
        print(f'Driver {driver} not known.')


def load_sensor_types(sensor_type, fwhm=None):
    raise NotImplementedError()
