import torch

import numpy as np

import glob, os, itertools, copy

from fluomapper.utils.gaussian import Resampler
from fluomapper.utils.func.interp1d import Interpolate

try:
    import rasterio as rio
except ModuleNotFoundError as e:
    pass

import pickle as pkl

from os.path import join as pjoin
import fluomapper.utils.data as da

from torch import nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import get_worker_info

from fluomapper.utils.nn import pooling_out_dim

try:
    from rasterio.windows import Window
except ModuleNotFoundError as e:
    pass

import pytorch_lightning as pl
from fluomapper.config import ExpandPathAction

from rasterio import logging
log = logging.getLogger()
log.setLevel(logging.ERROR)

import warnings

from collections import OrderedDict
import xarray as xr
import re

from fluomapper.config import get_avail_batch_samplers


def flatten_window(win):
    # win shape expected : batch_samples, xtrack, atrack, wvls
    # out win shape : px (batch_samples * xtrack * atrack), wvls

    # return win.flatten(start_dim=-3, end_dim=-2).permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
    if len(win.shape) > 2:
        if type(win) == torch.Tensor:
           return win.flatten(start_dim=0, end_dim=-2)

        else:
            win = win.transpose(3, 0, 1, 2)
            win = win.reshape(win.shape[0], -1).transpose()

            return win

    else:
        return win


class _Reader(Dataset):
    
    def __init__(self, base_path=None, paths=None, out_mode='single_spectra',  path_ids=None,
                 out_shape=1, out_stride=1, shuffle_reading=True, return_wvl=False, load=False,
                 spectral_window=None, spectral_window_wvl=None, read_sources=True, resample=False, fwhm=None,
                 sensor_type='hyplant', resample_sensor_type=None, reader_batch_size=1, meta_info=None,
                 wavelengths=None, constrain_to_xtrack_px=None, edge_mode=None, limit_patches=1.0, patch_on_init=False,
                 keep_frac=None, max_samples=None, exact_wvls=None, norm=1, return_source_ids=False, read_specs=None,
                 constrain_to_atrack_px=None, **run_spec):

        super(_Reader, self).__init__()
        
        if return_source_ids:
            self.META_INFO_KEYS = dict(source_id=self.get_source_id)

        else:
            self.META_INFO_KEYS = {}

        self.base_path = base_path
        self.paths = paths

        os.environ['GDAL_CACHEMAX'] = "0"
        os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "FALSE"
        os.environ["VSI_CACHE"] = "FALSE"
        os.environ["HDF5_USE_FILE_LOCKING"] = "0"
        os.environ["RAW_CHECK_FILE_SIZE"] = "NO"

        self.sources = None
        self.sources_meta = None
        self._sources_initialized = False
        self._worker_id = None

        self.spectral_window = spectral_window
        self.spectral_window_wvl = spectral_window_wvl
        self.spectral_windows = None

        self.out_shape = out_shape
        self.out_mode = out_mode
        self.out_stride = out_stride
        
        self.shuffle_reading = shuffle_reading
        self.reader_batch_size = reader_batch_size

        self.resample = resample
        self.fwhm = fwhm
        self.sensor_type = sensor_type

        self.constrain_to_xtrack_px = constrain_to_xtrack_px
        self.constrain_to_atrack_px = constrain_to_atrack_px
        self.edge_mode = edge_mode

        self.load = load
        self.read_specs = read_specs if read_specs is not None else {} 

        self.norm = norm

        self.wavelengths = wavelengths
        self._wavelengths = copy.deepcopy(self.wavelengths)
        if self.wavelengths is not None:
            self.wavelengths = nn.Parameter(self.wavelengths.float(), requires_grad=False)
            self._wavelenths = self.wavelengths.clone()
 
        self.resample_sensor_type = resample_sensor_type
        if self.resample_sensor_type is None:
            self.resample_sensor_type = self.sensor_type

        self.patch_on_init = patch_on_init
        self.limit_patches = limit_patches

        self.exact_wvls = exact_wvls
        self.keep_frac = keep_frac
        self.max_samples = max_samples

        if self.keep_frac is not None:
            assert patch_on_init

        self.read_sources()
        
        self.run_spec = run_spec
        self.gen_spec = dict(return_wvl=return_wvl)
        self.meta_info = meta_info 
         
        if self.exact_wvls is not None:
            self.spectral_window = None
            self.spectral_window_wvl = None

        self.source_ids = path_ids

    def read_sources(self):
        self.sources, self.sources_meta, self.wavelengths, self.spectral_windows, \
                self.out_wvls, self.resampler = [None] * 6

        if len(self.paths) == 0:
            self._len = 0
            return

        self.sources, self.sources_meta = self._read_sources(self.paths, self.read_specs)

        # fake lightweight sources for shape computations
        #self.sources = [np.zeros((s.shape[0], s.shape[1], s.shape[2]))
        #                for s in sources]

        self.sources_are_patched = False

        self._specify_spectral_domain()
        self._specify_spatial_domain()

        if self.keep_frac is not None:
            sl = slice(int(len(self.inds) * self.keep_frac[0]),
                       int(len(self.inds) * self.keep_frac[1]))
            self.inds = self.inds[sl]

        if self.max_samples:
            self.inds = self.inds[:self.max_samples]

        self._len = len(self.inds) // self.reader_batch_size

        if self.load and self.patch_on_init:
            self._patch_sources()
            self.sources_are_patched = True

            self._specify_spatial_domain()

        #elif not self.load:
        #    self.sources = None
        #    self.wavelengths = None

    def _ensure_open(self):
        if not self.load and self.sources is None:
            sources, meta = self._read_sources(self.paths, self.read_specs)
            self.sources = list(sources)
            self.sources_meta = list(meta)

    def get_source_id(self, source_index, indices, shape, win, **kwargs):
        return self.source_ids[source_index] * torch.ones(indices.shape[0]).long()
        
    def _specify_spectral_domain(self):
        if 'wavelength' in self.sources_meta[0].coords and self.wavelengths is None:
            self.wavelengths = [np.array(meta.coords['wavelength']).astype(np.float32) for meta in self.sources_meta]

        #elif self.wavelengths is None:
        #    self.wavelengths = [None for _ in self.sources]

        elif self.wavelengths is not None and not type(self.wavelengths) == list:
            self.wavelengths = [self.wavelengths] * len(self.sources)

        # create windows for each source
        if self.wavelengths is not None and self.spectral_window_wvl is not None:
            self.spectral_windows = [da.search_spectral_window(*self.spectral_window_wvl,
                                                                where=self.wavelengths[i])
                                     if self.spectral_window_wvl is not None and self.wavelengths[i] is not None
                                     else
                                     da.search_spectral_window(*self.spectral_window,
                                                               where=np.arange(source.shape[2]))
                                     if self.spectral_window is not None
                                     else
                                     [(0, source.shape[2])]
                                     for i, source in enumerate(self.sources)]
            self.spectral_windows = np.array([np.array(list(
                                                    itertools.chain(*[list(range(sp_win[0], sp_win[-1])) if len(sp_win) > 1 else [sp_win[0]]
                                                                      for sp_win in windows
                                                                      ]
                                                                    )
                                                    )
                                              )
                                     for windows in self.spectral_windows])

        elif self.wavelengths is not None and self.spectral_window is not None:
            self.spectral_windows = [np.arange(min(self.spectral_window[0], source.shape[-1] - 1), 
                                               min(self.spectral_window[-1], source.shape[-1])) 
                                     for source in self.sources]

        else:
            self.spectral_windows = [slice(None, None) for _ in self.sources]

        if self.exact_wvls is not None:
            self.exact_wvls = torch.from_numpy(self.exact_wvls) if not torch.is_tensor(self.exact_wvls) else self.exact_wvls
            self.interpolator = [Interpolate(x=torch.from_numpy(self.wavelengths[i]), xnew=self.exact_wvls) for i in range(len(self.sources))]
            self.wavelengths = [self.exact_wvls]
            self.out_wvls = self.exact_wvls
            
        # find out_wvls
        elif self.wavelengths is not None:
            in_sensor_wvls = self.wavelengths[0]
            self.out_wvls = da.select(in_sensor_wvls, [self.spectral_windows[0]], inclusive=True)#.astype(np.float32)
        
        else:
            self.out_wvls = torch.arange(self.sources[0].shape[0]).float()

        # set up resampler TODO: atm single resampler for all sources
        if self.resample:
            assert(self.fwhm is not None, 'Need to set fwhm to a float if resample is True')

            # change out_wvls if resampling is active
            in_wvls = self.out_wvls
            out_sensor_wvls, out_fwhm = da.load_sensor_types(sensor_type=self.resample_sensor_type)

            if self.spectral_window_wvl is not None:
                windows = da.search_spectral_window(*self.spectral_window_wvl, where=out_sensor_wvls)
            elif self.spectral_window is not None:
                windows = da.search_spectral_window(*self.spectral_window, where=np.arange(len(out_sensor_wvls)))
            else:
                windows = [(0, len(out_sensor_wvls))]

            self.out_wvls = da.select(out_sensor_wvls, windows, inclusive=False)

            # define resampler
            if self.fwhm is None:
                self.fwhm = out_fwhm

            self.resampler = Resampler(wvl=in_wvls, new_wvls=self.out_wvls, fwhm=out_fwhm)

        else:
            self.resampler = None

    def _specify_spatial_domain(self):
        if not self.sources_are_patched:
            self.source_shapes = [s.shape[:2] for s in self.sources]

            # set up reading indices
            # figure out length of each source
            self.source_sampling_dims = np.array([[pooling_out_dim(dim, padding=0, kernel_size=self.out_shape,
                                                                   stride=self.out_stride)
                                                   for dim in source_shape]
                                                  for source_shape in self.source_shapes])

            self.source_lengths = np.array(list(map(np.prod, self.source_sampling_dims)))

        else:
            self.source_lengths = [s.shape[0] for s in self.sources]

        self.inds = np.array([(i, j) for i in range(len(self.sources)) for j in range(self.source_lengths[i])])
        self._shuffle_reading_inds()

    def _shuffle_reading_inds(self, seed=0):
        if self.shuffle_reading:
            np.random.seed(seed)
            self.inds = np.random.permutation(self.inds)

    def synchronize(self, reader):
        self.inds = reader.inds
        self.source_sampling_dims = reader.source_sampling_dims

    def unit_transform(self, data):
        return data * self.norm
    
    def calibrate(self, source_ind, data):
        return data

    def resample_(self, data):
        if self.resampler:
            data = self.resampler.resample(data)
            data = torch.from_numpy(data)

        return data

    def _get_win_from_source(self, source_ind, spatial_index=None, absolute_index=None, tot_spectrum=False):
        if not tot_spectrum:
            spectral_window = self.spectral_windows[source_ind]

        else:
            spectral_window = slice(None, None)

        if not self.sources_are_patched:
            index = spatial_index
            if not self.load:
                win = [self._read_win_lazily_from_source(source_ind=source_ind,
                                                        spatial_index=i,
                                                        spectral_window=spectral_window)
                       for i in index]

                win = [w for w in win if not 0 in w.shape]
                if len(win) == 0:
                    return None
                else:
                    win = np.stack(win)

            else:
                source = self.sources[source_ind]
                win = [source[i[0]: i[0] + self.out_shape,
                                    i[1]: i[1] + self.out_shape,
                                    spectral_window]
                                    for i in index]
                
                win = [w for w in win if not 0 in w.shape]
                if len(win) == 0:
                    return None
                else:
                    win = torch.stack([w.expand(win[0].shape) for w in win])
        else:
            index = absolute_index
            win = source[index]

        return win

    def get_win_from_source(self, source_ind, spatial_index=None, absolute_index=None, flatten_to_px=False,
                            to_tensor=True, to_float=True, tot_spectrum=False):

        win = self._get_win_from_source(source_ind=source_ind,
                                        spatial_index=spatial_index,
                                        absolute_index=absolute_index,
                                        tot_spectrum=tot_spectrum)
        if win is None:
            return None
        
        if to_tensor and 'torch' not in str(win.dtype):
            win = self._load_from_numpy(win)

        if to_float:
            win.float()

        # now shape is (len(index), x, y, wvls, var), permute to (len(index), var, x, y, out_wvls)
        # win = win.transpose(0, 1, 2, 3)

        if flatten_to_px:
            win = flatten_window(win)

        return win
        
    def __len__(self):
        return self._len

    def _check_pattern(self, pattern):
        return pattern

    def __getitem__(self, index, channels=None, source_ind=None, **kwargs):
        self._ensure_open()

        gen_spec = self.gen_spec.copy()
        gen_spec.update(kwargs)

        if hasattr(index, '__len__'):
            index = list(index)

        elif type(index) is int:
            index = [index]

        else:
            raise NotImplementedError

        ret = self._read(index, source_ind=source_ind, flatten_to_px=self.out_mode == 'single_spectra', **gen_spec)

        return ret

    def _transform_sources(self, sources):
        if type(sources[0]) == xr.DataArray:
            return [s.transpose('x', 'y', 'band') for s in sources]  # band, y, x -> x, y, band

        elif torch.is_tensor(sources[0]):
            return [s.permute(2, 1, 0) for s in sources]  # band, y, x -> x, y, band

        else:
            return [s.transpose(2, 1, 0) for s in sources]

    def _complete_sources(self, sources, sources_meta):
        return sources, sources_meta

    def _cut_sources(self, sources):
        if self.constrain_to_xtrack_px is not None and len(self.constrain_to_xtrack_px) > 0:
            sources = [source[slice(*self.constrain_to_xtrack_px)] for source in sources]

        if self.constrain_to_atrack_px is not None and len(self.constrain_to_atrack_px) > 0:
            sources = [source[:, slice(*self.constrain_to_atrack_px)] for source in sources]

        return sources

    def _edge_mode(self, sources, sources_meta=None):
        _ret_meta = True
        if sources_meta is None:
            sources_meta = [None] * len(sources)
            _ret_meta = False

        for source, meta in zip(sources, sources_meta):
            if meta is not None:
                meta.attrs['orig_shape'] = source.shape

        if self.edge_mode == 'mirror':
            if type(sources[0]) == xr.DataArray: 
                sources = [s.pad(dict(x=(0, self.out_shape), y=(0, self.out_shape)), mode='symmetric') for s in sources]
            else:
                sources = [np.pad(s.numpy(), ((0, self.out_shape), (0, self.out_shape), (0, 0)), mode='symmetric') for s in sources]
                sources = [torch.from_numpy(s) for s in sources]

        if _ret_meta:
            return sources, sources_meta

        return sources

    def _read_sources(self, source_paths, read_specs):
        sources, sources_meta = zip(*[da.open_source(s, load=self.load, **read_specs) for s in source_paths])

        # to tensor if sources are loaded
        is_xr = type(sources[0]) is xr.DataArray
        if (self.load and is_xr) or not is_xr :
            sources, sources_meta = zip(*[self._create_tensor(s, m)
                                          for s, m in zip(sources, sources_meta)])

        sources = self._transform_sources(sources)
        sources, sources_meta = self._complete_sources(sources, sources_meta)
        sources = self._cut_sources(sources)
        sources, sources_meta = self._edge_mode(sources, sources_meta)
        return sources, sources_meta

    def get_rasterio_window(self, spatial_index, spectral_window):
        col = spatial_index[0]
        row = spatial_index[1]
        size = self.out_shape

        # define spatial window
        window = Window(col_off=col, row_off=row, width=size, height=size)

        # define bands
        if isinstance(spectral_window, slice) and spectral_window.start is not None \
                and spectral_window.stop is not None:
            band_indices = list(range(
                spectral_window.start + 1,
                spectral_window.stop + 1
            ))
        elif isinstance(spectral_window, (list, np.ndarray)):
            band_indices = [b + 1 for b in spectral_window]
        else:
            band_indices = None  # all bands

        return window, band_indices

    def _prep_raw_data(self, win):
        win = self._transform_sources([win])[0]
        win = self._edge_mode([win])[0]
        return win

    def _read_win_lazily_from_source(self, source_ind, spatial_index, spectral_window):
        s = self.paths[source_ind]
        window, band_indices = self.get_rasterio_window(spatial_index=spatial_index, spectral_window=spectral_window)
        win = da.open_source(s, window=window, band_indices=band_indices, driver='rasterio')
        win = self._prep_raw_data(torch.from_numpy(win))
        return win

    def _patch_sources(self):
        new_sources = []
        for i in range(len(self.sources)):
            indices_per_batch = self.inds[np.where(self.inds[:, 0] == i)][:, 1]  # np.arange(0, len(self.inds))
            new_sources.append(self._read(source_ind=i, index=indices_per_batch, patching=True))

        del self.sources
        self.sources = new_sources

    def _load_from_numpy(self, arr):
        dtype = dict(uint8=np.int8,
                     int16=np.int16,
                     int32=np.int32,
                     uint16=np.int16,
                     float=np.float32,
                     float64=np.float32,
                     float32=np.float32)[str(arr.dtype)]

        return torch.from_numpy(arr.astype(dtype))

    def _create_tensor(self, source, source_meta):
        with warnings.catch_warnings():
            return self._load_from_numpy(source), source_meta

    def _get_2d_index(self, ind, source_ind):
        return (self.out_stride * (ind % self.source_sampling_dims[source_ind][0]),
                self.out_stride * (ind // self.source_sampling_dims[source_ind][0]))
    
    def _get_reading_indices(self, index, source_ind=None):
        if source_ind is None:
            index = np.concatenate([self.inds[ind * self.reader_batch_size: ind * self.reader_batch_size
                                                                            + self.reader_batch_size]
                                    for ind in index], axis=0)

        else:
            index = np.asarray([[source_ind, i] for i in index])

        source_inds = np.unique(np.atleast_2d(index)[:, 0])

        # get corresponding 2d indices
        spatial_indices = [[source_ind] + list(self._get_2d_index(ind, source_ind))
                           for source_ind, ind in index]

        spatial_indices = dict([(source_ind, np.array([tup[1:] for tup in spatial_indices if tup[0] == source_ind]))
                                for source_ind in source_inds])

        absolute_indices = dict([(source_ind, np.array([tup[1] for tup in index if tup[0] == source_ind]))
                                 for source_ind in source_inds])

        return spatial_indices, absolute_indices

    def _assemble_meta_info(self, *args, **kwargs):
        meta = OrderedDict()
        for var in self.meta_info:
            if var in self.META_INFO_KEYS:
                meta[var] = self.META_INFO_KEYS[var](*args, **kwargs)
                
        return meta

    def _get_meta_info(self, meta, source_index, indices, shape=None, flatten_to_px=False, win=None):
        def to_tensor(arr):
            if type(arr) is np.ndarray:
                if arr.dtype in (np.float16, np.float32, np.float64):
                    arr = torch.from_numpy(arr).float()

                else:
                    arr = torch.from_numpy(arr).long()

            elif torch.is_floating_point(arr):
                arr = arr.float()

            else:
                arr = arr.long()

            return arr

        update_meta = self._assemble_meta_info(source_index=source_index, 
                                               indices=indices, shape=shape, 
                                               win=win)

        for var_name, var in update_meta.items():
           
            if not var_name in meta:
                meta[var_name] = []

            var = to_tensor(var)

            if flatten_to_px:
                var = flatten_window(var)

            meta[var_name].append(var)

    def _read(self, index, return_wvl=False, source_ind=None, flatten_to_px=False, patching=False):
        spatial_indices, absolute_indices = self._get_reading_indices(index, source_ind=source_ind)

        out = []
        meta = dict([])
        for source_index, abs_ind in absolute_indices.items():
            sp_ind = spatial_indices[source_index] if spatial_indices is not None else None

            out_per_source = self.get_win_from_source(source_index,
                                                      spatial_index=sp_ind,
                                                      absolute_index=abs_ind,
                                                      flatten_to_px=False,
                                                      to_float=not patching,
                                                      tot_spectrum=True)
            if out_per_source is None:
                continue

            if not patching:
                out_per_source = self.post_process(source_index, out_per_source)
                if self.meta_info is not None:
                    self._get_meta_info(meta,
                                        source_index, 
                                        indices=sp_ind,
                                        shape=out_per_source.shape[:-1],  # get spatial domain
                                        flatten_to_px=flatten_to_px,
                                        win=out_per_source)

                # cut spectrally
                # cut here so we have full spectrum in _get_meta_info
                spectral_window = self.spectral_windows[source_index]
                out_per_source = out_per_source[..., spectral_window]

                out_per_source = self.resample_(out_per_source)

                if self.exact_wvls is not None:
                    assert torch.is_tensor(out_per_source)

                    # interpolate to correct wvls
                    b_dim, sp_dim0, sp_dim1, wvl_dim = out_per_source.shape
                    wvl_dim = len(self.exact_wvls)

                    win = out_per_source.permute(3, 0, 1, 2).flatten(start_dim=1).permute(1, 0).float()
                    out_per_source = self.interpolator[source_index](win).permute(1, 0)\
                                        .reshape(wvl_dim, b_dim, sp_dim0, sp_dim1).permute(1, 2, 3, 0)

                if flatten_to_px:
                    out_per_source = flatten_window(out_per_source)

            out.append(out_per_source)
        
        if len(out) == 0:
            return None

        out = torch.cat(out, dim=0)
        if patching:
            return out

        for meta_key in meta:
            if len(meta[meta_key]) > 1:
                meta[meta_key] = torch.cat(meta[meta_key], dim=0)
                
            else:
                meta[meta_key] = meta[meta_key][0]

        if len(meta) != 0:
            out = dict(obs=out)
            out.update(meta)

        # reshape to have (samples, wvl) shape
        wvl = self.out_wvls.reshape(1, -1)

        if return_wvl:
            return out, wvl
        
        return out

    def post_process(self, source_ind, out):
        out = self.calibrate(source_ind, out)
        out = self.unit_transform(out)
        #out = self.resample_(out)

        return out

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--spectral_window', type=int, nargs='+', default=None)
        parser.add_argument('--spectral_window_wvl', type=float, nargs='+', default=None)
        parser.add_argument('--out_shape', type=int, default=1)
        parser.add_argument('--out_mode', type=str, default='single_spectra')
        parser.add_argument('--out_stride', type=int, default=1)
        parser.add_argument('--resample', type=int, default=0)
        parser.add_argument('--fwhm', type=str, default=None)
        parser.add_argument('--reader_batch_size', default=1, type=int)
        parser.add_argument('--shuffle_reading', default=1, type=int)
        parser.add_argument('--meta_info', type=str, nargs='*', default=None)
        parser.add_argument('--include_hdr_lines', type=int, default=0)
        parser.add_argument('--constrain_to_xtrack_px', type=int, default=None, nargs=2)
        parser.add_argument('--constrain_to_atrack_px', type=int, default=None, nargs=2)
        parser.add_argument('--edge_mode', type=str, default=None)
        parser.add_argument('--exact_wvls', type=float, nargs='*')
        return parser


class NoneReader(object):
    def __getitem__(self, *args, **kwargs):
        return None

    def __len__(self):
        return 0

    def read_sources(self):
        pass

    def synchronize(self, reader):
        pass


class JointReader(Dataset):
    def __init__(self, readers,
                 spectral_window_label_wvl=None, spectral_window_obs=None, spectral_window_obs_wvl=None, paths=None,
                 out_mode='single_spectra', out_shape=1, search_first='obs',
                 out_stride=1, label_type=None, obs_type=None, resample_obs=False, resample_label=False,
                 fwhm_obs=None, fwhm_label=None, resample_sensor_type=None, resample_sensor_type_label=None, volume_obs=None,
                 volume_label=None, meta_info=None, constrain_to_xtrack_px=None, edge_mode=None, limit_patches=1.0,
                 patch_on_init=False, shard_along=None, shard_along_nr_files_per_proc=1, path_ids=None, *args, **kwargs):

        super(JointReader, self).__init__()

        self.out_mode = out_mode
        self.out_shape = out_shape

        self.search_first = search_first

        self.volume_obs = volume_obs
        self.volume_label = volume_label

        self.paths = paths
        if path_ids is not None and len(path_ids) == 1 and path_ids[0] == 'date':
            self.path_ids = [int(self.get_date(os.path.basename(p[0]))) for p in self.paths]
        
        elif path_ids is not None and len(path_ids) == 1 and path_ids[0] == 'datetime':
            self.path_ids = [int(self.get_datetime(os.path.basename(p[0]))) for p in self.paths]
        
        elif path_ids is None:
            self.path_ids = np.arange(len(paths))

        else:
            self.path_ids = path_ids
            assert len(path_ids) == len(paths)

        self.spectral_window_obs = spectral_window_obs
        self.spectral_window_label = spectral_window_label_wvl

        self.obs_type = obs_type
        self.label_type = label_type
        self.fwhm_obs = fwhm_obs
        self.fwhm_label = fwhm_label

        self.keep_frac = None
        self.max_samples = None

        self.shard_along = shard_along
        if self.shard_along == 'none' or shard_along == 'None':
            self.shard_along = None
        
        if self.shard_along is not None:
            self.global_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            
            if self.shard_along == 'files':
                nr_files = shard_along_nr_files_per_proc
                ind = max(min(self.global_rank // nr_files, len(self.paths) // nr_files - 1), 0)
                self.paths = self.paths[ind*nr_files: (ind+1)*nr_files]
                self.path_ids = self.path_ids[ind*nr_files: (ind+1)*nr_files]

                print(f'LOADING FILE: rank {self.global_rank} loads {self.paths} \n')

            elif self.shard_along == 'indices':
                self.keep_frac = [self.global_rank / self.world_size, (self.global_rank + 1) / self.world_size]

            elif self.shard_along is not None:
                raise Exception(f'shard_along {shard_along} strategy is not implemented.')

            else:
                pass

        self.reader_specs = dict(out_mode=out_mode, out_shape=out_shape, out_stride=out_stride,
                                 read_sources=False, meta_info=meta_info, constrain_to_xtrack_px=constrain_to_xtrack_px,
                                 edge_mode=edge_mode, patch_on_init=patch_on_init, limit_patches=limit_patches, 
                                 keep_frac=self.keep_frac, max_samples=self.max_samples, path_ids=self.path_ids)
        self.reader_specs.update(kwargs)
        
        # OBSERVATION
        obs_dict = dict(paths=[s[0] for s in self.paths],
                        variables=obs_type,
                        resample=resample_obs and (fwhm_obs is None or fwhm_obs > 0),
                        fwhm=fwhm_obs,
                        resample_sensor_type=resample_sensor_type,
                        spectral_window=self.spectral_window_obs,
                        spectral_window_wvl=spectral_window_obs_wvl,
                        )
        obs_dict.update(self.reader_specs)

        self.readers = [readers[0](**obs_dict)]
        self.obs_dict = obs_dict

        # LABEL
        if self.paths[0][1] is not None:
            label_dict = dict(paths=[s[1] for s in self.paths],
                              variables=label_type,
                              resample=resample_label and (fwhm_label is None or fwhm_label > 0),
                              fwhm=fwhm_label,
                              resample_sensor_type=resample_sensor_type_label,
                              spectral_window=self.spectral_window_label,
                              spectral_window_wvl=spectral_window_label_wvl,
                              )
            label_dict.update(self.reader_specs)

            # INSTANTIATE readers
            self.readers.append(readers[1](**label_dict))
            self.label_dict = label_dict

        else:
            self.readers.append(NoneReader())

    def _ensure_sources(self):
        for reader in self.readers:
            if hasattr(reader, "_ensure_sources"):
                reader._ensure_sources()

    def get_date(self, p):
        #return re.match('(.*?)-.*', os.path.basename(p))[0]
        return os.path.basename(p)[:8]
    
    def get_datetime(self, p):
        match = re.match('([0-9]*?)-[A-Z0-9]*?-([0-9]*?)-.*', os.path.basename(p))
        return match[1] + match[2]

    def _make_list(self, item):
        if type(item) is list:
            return item

        else:
            return [item]

    def __getitem__(self, index, **kwargs):
        self._ensure_sources()
        return tuple([r.__getitem__(index=index, **kwargs) for r in self.readers])


    def __len__(self):
        return len(self.readers[0])
        
    def read_sources(self):
        for reader in self.readers:
            reader.read_sources()
                
        # synchronize readers
        for reader in self.readers[1:]:
            reader.synchronize(self.readers[0])

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--spectral_window_obs', type=int, nargs='+', default=None)
        parser.add_argument('--spectral_window_obs_wvl', type=float, nargs='+', default=None)
        parser.add_argument('--spectral_window_label', type=int, nargs='+', default=None)
        parser.add_argument('--spectral_window_label_wvl', type=float, nargs='+', default=None)
        parser.add_argument('--resample_sensor_type', type=str, default=None)
        parser.add_argument('--resample_sensor_type_label', type=str, default=None)
        parser.add_argument('--out_mode', type=str, default='single_spectra')
        parser.add_argument('--out_shape', type=int, default=1)
        parser.add_argument('--out_stride', type=int, default=1)
        parser.add_argument('--resample_obs', type=int, default=False)
        parser.add_argument('--resample_label', type=int, default=False)
        parser.add_argument('--fwhm_obs', type=float, default=None)
        parser.add_argument('--fwhm_label', type=float, default=None)
        parser.add_argument('--reader_batch_size', type=int, default=1)
        parser.add_argument('--shuffle_reading', type=int, default=1)
        parser.add_argument('--meta_info', type=str, nargs='*', default=None)
        parser.add_argument('--include_hdr_lines', type=int, default=0)
        parser.add_argument('--constrain_to_xtrack_px', type=int, default=None, nargs=2)
        parser.add_argument('--constrain_to_atrack_px', type=int, default=None, nargs=2)
        parser.add_argument('--edge_mode', type=str, default=None)
        parser.add_argument('--shard_along', type=str, default=None)
        parser.add_argument('--shard_along_nr_files_per_proc', type=int, default=1),
        parser.add_argument('--path_ids', type=str, default=None, nargs='*')
        return parser


class _BaseDataModule(pl.LightningDataModule):
    DSET_TYPE = None

    def __init__(self, batch_size=1, num_workers=None, pin_memory=False, num_workers_eval=None, persistent_workers=False,
                 paths_file=None, exclude_paths=None, exclude_file=None, trn_patch_on_init=False,
                 val_patch_on_init=False, del_dset_on_dl_creation=False, load_val=False, load_trn=False,
                 load=False, batch_sampler=None, min_samples_in_files=None, *args, **kwargs):
        """

        :param out_shape:
        :param trn_frac:
        :param val_frac:
        :param tst_frac:
        :param spectral_window:
        :param batch_size:
        :param num_workers:
        :param target:
        :param kwargs:
        """
        super(_BaseDataModule, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_workers_eval = num_workers_eval if num_workers_eval is not None else self.num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.paths_file = paths_file
        self.exclude_file = exclude_file
        self.exclude_paths = exclude_paths

        self.volume_obs = None
        self.volume_label = None

        if self.num_workers is None:
            try:
                self.num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])

            except KeyError:
                self.num_workers = 1

        self.trn_patch_on_init = trn_patch_on_init
        self.val_patch_on_init = val_patch_on_init
        self.del_dset_on_dl_creation = del_dset_on_dl_creation

        self.load_val = load or load_val
        self.load_trn = load or load_trn

        self.dset_kwargs = kwargs

        self.batch_sampler = batch_sampler
        self.num_samples_dloader_per_source = min_samples_in_files

        torch.seed()

    def prepare_data(self, *args, **kwargs):
        raise NotImplementedError()

    def setup(self, stage='fit', mode=None, load=True):
        if stage in ("val", "fit") or stage is None:
            limit_patches = None
            if self.val_patch_on_init:
                limit_patches = self.dset_kwargs['limit_val_batches_perc']
                self.dset_kwargs['limit_val_batches_perc'] = 1.0

            self.val_dset = self.get_dset(patch_on_init=self.val_patch_on_init,
                                          limit_patches=limit_patches,
                                          load=self.load_val and load,
                                          **self.dset_kwargs, **self.val_data)

        if stage == "fit" or stage is None:
            limit_patches = None
            if self.trn_patch_on_init:
                limit_patches = self.dset_kwargs['limit_train_batches_perc']
                self.dset_kwargs['limit_train_batches_perc'] = 1.0
            
            self.trn_dset = self.get_dset(patch_on_init=self.trn_patch_on_init,
                                          limit_patches=limit_patches,
                                          load=self.load_trn and load,
                                          **self.dset_kwargs, **self.trn_data)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            raise NotImplementedError()

    def load_paths(self, file):

        if not type(file) is list:
            if file.endswith('pkl'):
                with open(file, 'rb') as f:
                    paths = pkl.load(f)

            elif file.endswith(".txt"):
                with open(file, "r") as f:
                    paths = [line.strip() for line in f if line.strip()]
                    paths = [(p, None) for p in paths]

        else:
            paths = file

        new_paths = []
        for path_obs, path_label in paths:

            if self.volume_obs is not None:
                obs = pjoin(self.volume_obs, path_obs)

            else:
                obs = path_obs

            if path_label is not None:

                if self.volume_label is not None:
                    label = pjoin(self.volume_label, path_label)

                else:
                    label = path_label

            else:
                label = None

            new_paths.append((obs, label))

        return new_paths

    def _exclude_paths(self, paths):
        if self.exclude_file is not None and len(self.exclude_file) > 0:
            with open(self.exclude_file, 'rb') as fil:
                self.exclude_paths = pkl.load(fil)

        if self.exclude_paths is not None and len(self.exclude_paths) > 0:
            in_ = [i for i, path in enumerate(paths) if (os.path.basename(path[0]) not in self.exclude_paths
                                                         and os.path.basename(path[1]) not in self.exclude_paths)]

        else:
            in_ = list(range(len(paths)))

        paths = [path for i, path in enumerate(paths) if i in in_]

        return paths

    def get_dset(self, data, **kwargs):
        """
        Returns a constructed dset.

        :param data:
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def assemble_data(self, *args, **kwargs):
        """
        Returns  trn_data, val_data, tst_data each of which is an object that allows the construction of the dset
        together with dset_kwargs.

        :param args:
        :param kwargs:qqq
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        """
        Expects list [(samples, var, ..., wvl), ..., (samples, var, ..., wvl)]
        :param batch:
        :return:
        """

        def _flatten_samples(out):
            """
            Expects (batches, samples ..., wvl)
            Returns for (batches, samples, var, wvl): (samples * batches = px, var, wvl), if dim(var) == 1, squeeze it
            Returns for (batches, samples, var, ..., wvl): (samples * batches, wvl, var, ...), if dim(var) == 1, squeeze it

            :param out:
            :param start_dim:
            :param end_dim:
            :return:
            """
            try:
                if type(out) in (list, tuple):
                    return [_flatten_samples(o) for o in out]
                
                elif type(out) is dict:
                    return dict([(key, _flatten_samples(val)) 
                                 for key, val in out.items()])
                else:
                    # if there is only a batch dimension, return
                    if out is None:
                        return None

                    if len(out.shape) == 1:
                        return out 
                    
                    out = out.flatten(start_dim=0, end_dim=1)

                    # if there is only a batch dimension, return
                    if len(out.shape) == 1:
                        return out

                    # try to squeeze var dim
                    # out = out.squeeze(dim=1)

                    # if it's images move wvls to channel position
                    if len(out.shape) > 2:
                        out = da.permute_channels(out)

                    return out
            except AttributeError as e:
                print(f'Invalid type of object {out}')
                raise e

        if batch is None:
            return None 

        # there is no label
        if type(batch[0]) is dict:
            batch = default_collate(batch)
        
        # batch has None labels
        elif type(batch[0]) is tuple and batch[0][1] is None:
            batch = (default_collate([b[0] for b in batch]), None)
        
        # there is a label tensor
        else:
            batch = default_collate(batch)
            
        # concatenate samples (so reader can return multiple samples)
        batch = _flatten_samples(batch)
        return batch

    def get_dloader(self, shuffle=False, mode='train'):
        if mode == 'train':
            dset = self.trn_dset

        elif mode == 'val':
            dset = self.val_dset

        else:
            raise NotImplementedError()

        sampler_kwargs = dict(shuffle=shuffle,
                              batch_size=self.batch_size)
       
        if self.batch_sampler is not None and get_avail_batch_samplers()[self.batch_sampler] is not None:
            batch_sampler = get_avail_batch_samplers()[self.batch_sampler](dataset=dset,
                                                                           num_samples=self.num_samples_dloader_per_source * len(dset.paths),
                                                                           same_sequence=True, #not mode == 'train',
                                                                           **sampler_kwargs)
            sampler_kwargs = dict(sampler=batch_sampler)

        other_kwargs = dict(persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn)

        dl = DataLoader(dset, **sampler_kwargs, **other_kwargs)

        if self.del_dset_on_dl_creation:
            del dset

        return dl

    def train_dataloader(self, shuffle=True, **kwargs):
        return self.get_dloader(shuffle=shuffle, mode='train')

    def val_dataloader(self, **kwargs):
        return self.get_dloader(shuffle=False, mode='val')

    def test_dataloader(self, **kwargs):
        return self.get_dloader(shuffle=False, mode='test')

    @classmethod
    def add_argparse_args(cls, parser, *args, **kwargs):
        parser = cls.DSET_TYPE.add_argparse_args(parser, *args, **kwargs)
        parser.add_argument('--num_workers', default=None, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--pin_memory', default=0, type=int)
        parser.add_argument('--persistent_workers', default=0, type=int)
        parser.add_argument('--exclude_file', type=str, default=None, action=ExpandPathAction)
        parser.add_argument('--exclude_paths', type=str, default=None, action=ExpandPathAction, nargs="*")
        parser.add_argument('--paths_file', type=str, default=None, action=ExpandPathAction, nargs="+")
        parser.add_argument('--trn_patch_on_init', type=int, default=0)
        parser.add_argument('--val_patch_on_init', type=int, default=0)
        parser.add_argument('--load_val', type=int, default=0)
        parser.add_argument('--load_trn', type=int, default=0)
        parser.add_argument('--load', default=1, type=int)
        parser.add_argument('--del_dset_on_dl_creation', type=int, default=0)
        parser.add_argument('--batch_sampler', type=str, default=None)
        return parser


if __name__ == '__main__':
    from data.hyplant.hyplant import _HyPlantReader
    
    base_path = '/Volumes/processed/21- FlexSense- June 2018/HyPlant/20180802/FLUO/'
    pattern = '20180802-SEL-1227-600-L1-S-FLUO-rect.dat'
    base_path_label = '/Volumes/products/21- FlexSense- June 2018/Fs maps/SFM/Single Lines'
    
    a = JointReader(readers=[_HyPlantReader, _HyPlantReader], base_path=base_path,
                    base_path_label=base_path_label, pattern=pattern)

