from fluomapper.data._base.base import _Reader, _BaseDataModule, JointReader
from fluomapper.path_config import HYPLANT_DATA_PATH, HYPLANT_METAVAR_PATH
from fluomapper.utils.data import search_spectral_window, select

import os, glob
from os.path import join as pjoin
import numpy as np
import datetime, pytz
import pysolar as sol
import re
from functools import partial
import torch
import xarray as xr
import itertools as it

import pandas as pd
import math


def NDVI(spectrum, wvls, axis=-1):
    MIR_wvls = search_spectral_window(770, 810, where=wvls)
    NIR_wvls = search_spectral_window(650, 680, where=wvls)

    MIR = torch.mean(select(spectrum, MIR_wvls, axis=axis), axis=axis)
    NIR = torch.mean(select(spectrum, NIR_wvls, axis=axis), axis=axis)

    return (MIR - NIR + 1e-5) / (MIR + NIR + 1e-5)


class _HyPlantReader(_Reader):
    def __init__(self, off_nadir_mode=None, norm=1e-2, return_source_ids=True,
                 compatibility_h1alt_to_h2alt=False, use_yaw_as_scene_azimuth=False,
                 *args, **kwargs):
        
        self.use_yaw_as_scene_azimuth = use_yaw_as_scene_azimuth
        self.off_nadir_mode = off_nadir_mode
        
        if return_source_ids and not 'source_id' in getattr(kwargs, 'meta_info', []):
            kwargs['meta_info'].append('source_id')
        
        super(_HyPlantReader, self).__init__(sensor_type='hyplant', norm=norm,
                                             return_source_ids=return_source_ids,
                                             *args, **kwargs)

        if compatibility_h1alt_to_h2alt:
            self.META_INFO_KEYS['h2alt'] = self.get_h1alt


        self.META_INFO_KEYS.update(sza=self.get_sza,
                                   off_nadir=self.get_off_nadir,
                                   ndvi=self.get_ndvi,
                                   source_id=self.get_source_id,
                                   h1alt=self.get_h1alt,
                                   rel_azimuth=self.get_rel_azimuth,
                                   tilt=self.get_tilt,
                                   alt=self.get_altitude,
                                   dist=self.get_dist)


        self.source_h1alts = [float(re.match('(.*?)-(.*?)-(.*?)-(.*?)-.*',
                                             os.path.basename(p))[4]) for p in self.paths] 

    def _complete_sources(self, sources, sources_meta):
        sources, sources_meta = super(_HyPlantReader, self)._complete_sources(sources, sources_meta)
        
        for source_ind in range(len(sources)):
            source_meta = sources_meta[source_ind]
            sources_meta[source_ind].attrs['scene_azimuth'] = self.get_scene_azimuth(source_ind)
            sources_meta[source_ind].attrs['solar_azimuth'] = self.get_solar_azimuth(source_ind, source_meta)
            sources_meta[source_ind].attrs['file_name'] = os.path.basename(self.paths[source_ind]) 

        return sources, sources_meta

    def read_sources(self):
        super(_HyPlantReader, self).read_sources()
        meta = self.sources_meta[0]
        self.wavelengths = torch.from_numpy(np.array(meta.coords['wavelength'])\
                                            .astype(np.float32)).requires_grad_(False)

    def get_ndvi(self, source_index, indices, shape, win, **kwargs):
        ndvi = NDVI(win, self.wavelengths).unsqueeze(-1)
        return ndvi

    def get_h1alt(self, source_index, indices, shape=None,  **kwargs):
        h2alt = self.source_h1alts[source_index] * torch.ones(shape)
        return h2alt.unsqueeze(-1)

    def get_altitude(self, source_index, indices, shape=None, **kwargs):
        lon, lat, alt, yaw, roll, pitch = self.get_nav_file(source_index)
        alt = [alt[min(ind[1], alt.shape[0] - 1):min(ind[1], 
                   alt.shape[0] - 1) + shape[-1]] 
               for ind in indices]
        
        for i, a in enumerate(alt):
            if a.shape[0] != shape[-1]:
                a = np.concatenate([a, a[-1]\
                            .repeat(shape[-1] - a.shape[0], axis=0)], 
                                    axis=-1)
                alt[i] = a

        alt = np.stack(alt, axis=0)[:, None] # add off_nadir dimension
         
        alt = alt * np.ones(shape) 
        alt = alt[..., None] # add var dim

        return alt

    def get_date_lat_lon(self, source_index, source_meta):
        #string = (source_meta.attrs['acquisition_date'].split(':')[1] +
        #          source_meta.attrs['gps_start_time'].split('TIME:')[1].split('.')[0]).strip()

        #filter_ = '%d-%m-%Y' if source_meta.attrs['acquisition_date'].split(':')[0]\
        #                                .startswith('DATE(dd-mm-yyyy)') else '%Y-%m-%d'
        #filter_ += '  %H:%M:%S'
        
        string = os.path.basename(self.paths[source_index]).split('-')
        string = string[0] + '-' + string[2]

        filter_ = '%Y%m%d-%H%M'

        date = datetime.datetime.strptime(string, filter_) - datetime.timedelta(hours=2)

        tz = pytz.timezone('UTC')  # gps = utc up to some seconds
        date = tz.localize(date)

        try:
            lat, lon = list(map(float, source_meta.attrs['gps_starting_point'].strip().split(',')))
        except KeyError as e:
            print(f'hdr of {self.paths[source_index]} is erroneous')
            raise(e)

        return date, lat, lon

    def get_nav_file(self, source_index):
        path = self.paths[source_index]
        try:
            nav_path = path[:-len('_radiance.dat')] + '_syncedNAV.txt'
            if not os.path.exists(nav_path):
                nav_path = path[:-len('_radiance.dat')] + '_syncedNav.txt'
            nav_file = pd.read_csv(nav_path, skiprows=[0], header=None, sep='\s+') 

        except:
            raise Exception(f'Could not find a nav file under {nav_path}')

        if self.constrain_to_atrack_px is not None:
            nav_file = nav_file[slice(*self.constrain_to_atrack_px)]

        lon = nav_file.values[:, 3]
        lat = nav_file.values[:, 2]
        alt = nav_file.values[:, 4]

        yaw = nav_file.values[:, 5]
        roll = nav_file.values[:, 6]
        pitch = nav_file.values[:, 7]

        return lon, lat, alt, yaw, roll, pitch

    def get_sza(self, source_index, indices, shape=None, **kwargs):
        source = self.sources[source_index]
        source_meta = self.sources_meta[source_index]

        date, lat, lon = self.get_date_lat_lon(source_index, source_meta)
        
        sza = 90 - sol.solar.get_altitude(lat, lon, date)
        if shape is not None:
            sza = np.ones(shape) * sza
            sza = sza[..., None]  # unsqueeze a var dimension
        
        return sza

    def get_solar_azimuth(self, source_index, source_meta):
        date, lat, lon = self.get_date_lat_lon(source_index, source_meta)
        solar_azimuth = sol.solar.get_azimuth(lat, lon, date)

        return solar_azimuth

    def calculate_azimuth(self, lat1, lon1, lat2, lon2):
        delta_lon = lon2 - lon1

        azimuth = math.atan2(np.sin(delta_lon) * np.cos(lat2),
                             np.cos(lat1) * np.sin(lat2) -
                             np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))

        # Convert azimuth from radians to degrees
        azimuth = np.degrees(azimuth)

        # Adjust azimuth to be between 0 and 360 degrees
        azimuth = (azimuth + 360) % 360

        return azimuth

    def get_scene_azimuth(self, source_index):
        lon, lat, alt, yaw, roll, pitch = self.get_nav_file(source_index)

        if not self.use_yaw_as_scene_azimuth:
            azs = []
            for i in range(len(lon) - 1):
                azimuth = self.calculate_azimuth(lat[i], lon[i], lat[i+1], lon[i+1])
                azs.append(azimuth)
            azs.append(azimuth)
            azs = np.array(azs)

        else:
            azs = yaw
        
        #azs = self.calculate_azimuth(lat[0], lon[0], lat[-1], lon[-1]) * np.ones(lon.shape[0])
        return azs

    def get_rel_azimuth(self, source_index, indices, shape=None, **kwargs):
        solar_azimuth = self.sources_meta[source_index].attrs['solar_azimuth']
        scene_azimuth = self.sources_meta[source_index].attrs['scene_azimuth']

        rel_azimuth = np.abs(solar_azimuth - scene_azimuth)
        min_ = lambda x: min(x, 360 - x)
        rel_azimuth = np.array(list(map(min_, rel_azimuth)))

        rel_azimuth = np.stack([rel_azimuth[ind[1]:ind[1] + shape[-1]] for ind in indices])

        if shape is not None:
            if shape[-1] > rel_azimuth.shape[-1]:
                rel_azimuth = np.concatenate([rel_azimuth, rel_azimuth[..., [-1]]\
                                             .repeat(shape[-1] - rel_azimuth.shape[-1], axis=-1)], 
                                             axis=-1)
            rel_azimuth = np.expand_dims(rel_azimuth, axis=1) * np.ones(shape)  # add xtrack dim
            rel_azimuth = rel_azimuth[..., None]  # add var dim

        return rel_azimuth

    def get_tilt(self, source_index, indices, shape=None, **kwargs):
        # get off_nadir
        off_nadir = self.get_off_nadir(source_index, indices, shape, **kwargs)
        # calc tilt angle with FOV
        tilt = np.abs(off_nadir) * 16.08 / 192
        return tilt

    def get_dist(self, source_index, indices, shape, **kwargs):
        # len_ = self.sources[source_index].shape[1]
        len_ = 1000
        shift_ = self.constrain_to_atrack_px[0] if self.constrain_to_atrack_px is not None else 0
        dist = np.stack([np.repeat(np.atleast_2d(np.arange(i[1] + shift_, i[1] + shape[-2] + shift_) / len_),
                                    shape[-1], axis=0).transpose() for i in indices])[:, None].transpose(0, 3, 2, 1)

        return dist

    def get_off_nadir(self, source_index, indices, shape, **kwargs):
        shift_ = self.constrain_to_xtrack_px[0] if self.constrain_to_xtrack_px is not None else 0

        if self.off_nadir_mode is None or self.off_nadir_mode == 'none':
            off_nadir = np.stack([np.repeat(np.atleast_2d(np.arange(shift_ + i[0] - 384 / 2, shift_ + i[0] - 384 / 2 + shape[-1])),
                                            shape[-2], axis=0) for i in indices])[:, None].transpose(0, 3, 2, 1)
            
        elif type(self.off_nadir_mode) is int or self.off_nadir_mode.isdigit():
            ks = np.array([2 * np.pi / n for n in range(384 - int(self.off_nadir_mode), 384)])
            cos = np.stack([[
                                        np.repeat(np.atleast_2d(np.cos(k * np.arange(i[0], i[0] + shape[-1]))), shape[-2], axis=0)
                                    for k in ks] 
                                  for i in indices]).transpose(0, 3, 2, 1)

            sin = np.stack([[
                                np.repeat(np.atleast_2d(np.sin(k * np.arange(i[0], i[0] + shape[-1]))), shape[-2], axis=0)
                             for k in ks]
                            for i in indices]).transpose(0, 3, 2, 1)
            
            off_nadir = np.concatenate((cos, sin), axis=3)

        else:
            raise NotImplementedError()

        return off_nadir

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--off_nadir_mode', type=str, default=None)
        parser.add_argument('--compatibility_h1alt_to_h2alt', default=False, type=int)
        parser.add_argument('--use_yaw_as_scene_azimuth', default=False, type=int)

        return parser


class _SFMReader(_Reader):
    def __init__(self, *args, **kwargs):
        super(_SFMReader, self).__init__(sensor_type='sfm', *args, **kwargs)

        vars = ['AOT', 'H1', 'SPR', 'H2OSTR', 'G']
        self.META_INFO_KEYS = dict(zip(vars, [partial(self._getter, key=var) for var in vars]))
        self.META_INFO_KEYS.update(dict(t14=self._get_t14))

    def _getter(self, source_index, indices, key, shape=None, *args, **kwargs):
        source = self.sources[source_index]
        val = source.attrs[key]

        if shape is not None:
            val = np.ones(shape) * val

        return val

    def _get_t14(self, source_index, indices, shape=None, *args, **kwargs):
        source = self.sources[source_index]
        t14 = source.attrs['t14fnct']['t14']
        return torch.from_numpy(t14).float()#.unsqueeze(0)

    def unit_transform(self, data):
        # data comes in (mW/m^2*str*nm)*100.0000
        # we want mW/m^2*str*nm
        return np.clip(data, a_min=0, a_max=None) / 100

    def _complete_sources(self, sources, sources_meta):
        vars = ['AOT', 'H1', 'SPR', 'H2OSTR', 'G', 'MODEL', 'ASTMX', 'SZA', 'SAA', 'RAA']
        for source, source_meta in zip(sources, sources_meta):

            # load infos
            # try, ignore if log file is not present
            try:
                info_path = source_meta.attrs['path'].replace('.bil', '.log')
                with open(info_path, 'rt') as f:
                    info_fil = f.read()

            except FileNotFoundError as e:
                continue

            var_dict = {}
            for var in vars:
                match = re.search(rf'\s+{var}:\s+([\d\.]+)', info_fil)

                if match:
                    match = match.groups()[0]
                    var_dict[var] = float(match)

            source_meta.attrs.update(var_dict)

        return sources, sources_meta


class _iFLDReader(_Reader):
    def __init__(self, *args, **kwargs):
        super(_iFLDReader, self).__init__(sensor_type='ifld', spectral_window=(2, 3), *args, **kwargs)

    def unit_transform(self, data):
        # data comes in (mW/m^2*str*nm)*100.0000
        # we want mW/m^2*str*nm
        return np.clip(data / 100, a_min=0, a_max=None)


class JointHyPlantReader(JointReader):
    READER1 = _HyPlantReader 
    READER2 = _SFMReader

    def __init__(self, volume_obs, volume_label, volume_other=None, paths=None, loc=None, label='sfm',
                 *args, **kwargs):
        
        self.loc = loc
        if self.loc and type(self.loc) is str:
            self.loc = [self.loc]

        self.label = label
        self.volume_obs = volume_obs
        self.volume_label = volume_label
        self.volume_other = volume_other

        super(JointHyPlantReader, self).__init__(readers=[self.READER1, self.READER2],
                                                 paths=paths,
                                                 volume_obs=volume_obs,
                                                 volume_label=volume_label,
                                                 *args, **kwargs)

    @classmethod
    def add_argparse_args(cls, parser):
        parser = super(JointHyPlantReader, cls).add_argparse_args(parser)
        parser = _HyPlantReader.add_argparse_args(parser)
        return parser


class HyPlantMetaVarReader(JointHyPlantReader):
    META_VAR_READER = _Reader

    def __init__(self, meta_vars=None, meta_vars_dir=None, meta_vars_volume=None, *args, **kwargs):
        super(HyPlantMetaVarReader, self).__init__(*args, **kwargs)
        
        if meta_vars_volume is None:
            self.meta_vars_volume = HYPLANT_METAVAR_PATH 
        else:
            self.meta_vars_volume = meta_vars_volume

        self.meta_vars_dir = meta_vars_dir
        self.meta_vars = meta_vars if meta_vars is not None else []
        meta_readers = []

        for var in self.meta_vars:
            kwargs.update(dict(paths=self.paths))
            paths, other_vars = self.path_transform(var, **kwargs)
            
            meta_dict = dict(paths=paths, variables=var)
            meta_dict.update(self.reader_specs)
            meta_dict.update(other_vars)
            
            meta_dict["return_wvl"] = False
            meta_readers.append(self.META_VAR_READER(**meta_dict))

        for i, reader in enumerate(meta_readers):
            self.readers.insert(1 + i, reader)

        self.keys = list(self.meta_vars)

    def path_transform(self, var, paths=None, spectral_window_obs=None, spectral_window_obs_wvl=None, **kwargs):

        if var in ['aoi_sensor', 'parm1_sensor',
                   'dem_sensor', 'demadapted_sensor', 
                   'slope_sensor', 'aspect_sensor',
                   'tilt_sensor', 'scene_incidence_sensor', 
                   'scene_incidence_adapted_sensor']:

            paths = [os.path.normpath(
                        pjoin(self.meta_vars_volume,
                              self.meta_vars_dir,
                              os.path.normpath(path[0][len(self.volume_obs) + 1:-4]))
                           + f'_{var}' + os.path.splitext(path[0])[1]
                        )
                     for path in paths]

            other_vars = dict(var=var)

        elif var == 'reflectance':
            paths = [glob.glob(pjoin(self.meta_vars_volume, 'dual_reflectance',
                                     os.path.basename(path[0])[:len('20190623-WST-1114')] + '*sensor_reflectance.tif'))[0]
                     for path in paths]

            sample_reflectance = xr.open_rasterio(pjoin(HYPLANT_DATA_PATH,
                                                        'processed',
                                                        '22- FlexSense- June 2019',
                                                        'HyPlant/Intermediate processing steps/DUAL',
                                                        '20190620-WST-1416-1500-L1-N-DUAL_radiance_img_atm_pol.bsq'))

            dual_reflectance_wvls = torch.from_numpy(sample_reflectance.coords['wavelength'].data)

            other_vars = dict(resample=True, resample_sensor_type='hyplant', wavelengths=dual_reflectance_wvls, 
                              spectral_window=spectral_window_obs, spectral_window_wvl=spectral_window_obs_wvl, var=var)

        else:
            raise NotImplementedError(f'Variable {var} not known.')

        return paths, other_vars

    def __getitem__(self, item, **kwargs):
        ret = super(HyPlantMetaVarReader, self).__getitem__(item, **kwargs)

        return_wvl = type(ret[0]) is tuple
        if not return_wvl:
            if type(ret[0]) is not dict:
                meta_vars = dict(list(zip(self.keys, ret[1:-1])))
                x = dict(obs=ret[0])
                x.update(meta_vars)

            else:
                meta_vars = dict(list(zip(self.keys, ret[1:-1])))
                ret[0].update(meta_vars)
                x = ret[0]

        else:
            if type(ret[0][0]) is not dict:
                meta_vars = dict(list(zip(self.keys, ret[1:-1])))
                x = dict(obs=ret[0][0])
                x.update(meta_vars)

                x = (x, ret[0][1])
            else:
                meta_vars = dict(list(zip(self.keys, ret[1:-1])))
                ret[0][0].update(meta_vars)
                x = ret[0]

        y = ret[-1]

        return x, y

    @classmethod
    def add_argparse_args(cls, parser):
        parser = super(HyPlantMetaVarReader, cls).add_argparse_args(parser)
        parser.add_argument('--meta_vars', type=str, nargs='*', default=None)
        parser.add_argument('--meta_vars_dir', type=str, default=None)
        parser.add_argument('--meta_vars_volume', type=str, default=None)
        return parser


class DataModule(_BaseDataModule):
    DSET_TYPE = JointHyPlantReader

    def __init__(self, val_frac=0.3, volume=None, shuffle_data_module=False, val_files=None, max_nr_paths=None,
                 train_files=None, *args, **kwargs):
        super(DataModule, self).__init__(*args, **kwargs)
        self.shuffle_data_module = shuffle_data_module

        self.volume = volume
        if self.volume is None:
            self.volume_obs = HYPLANT_DATA_PATH
            self.volume_label = HYPLANT_DATA_PATH

        else:
            self.volume_obs = volume
            self.volume_label = volume

        # add volume since it is not in kwargs
        self.dset_kwargs['volume_obs'] = self.volume_obs
        self.dset_kwargs['volume_label'] = self.volume_label

        self.paths = []
        if type(self.paths_file) not in (tuple, list):
            self.paths_file = [self.paths_file]

        for p in self.paths_file:
            paths = self.load_paths(p)
            paths = self._exclude_paths(paths)
            self.paths.append(paths)
        
        self.paths = list(it.chain(*self.paths))

        if self.shuffle_data_module:
            order = np.random.permutation(list(range(len(self.paths))))
            self.paths = [self.paths[i] for i in order]

        if max_nr_paths is not None:
            self.paths = self.paths[:max_nr_paths]

        self.val_frac = val_frac
        self.val_files = val_files
        self.train_files = train_files

        self.trn_data = None
        self.val_data = None
        self.tst_data = None

    def prepare_data(self):
        self.trn_data, self.val_data, self.tst_data = self.assemble_data(self.val_frac, self.val_files, self.train_files)

    def assemble_data(self, val_frac=None, val_files=None, train_files=None, *args, **kwargs):
        if (val_files is not None or train_files is not None) and self.paths_file is not None and len(self.paths_file) > 0:
            
            val_files = [i for i in range(len(self.paths)) if i not in train_files] if val_files is None else val_files
            val_paths = [path for i, path in enumerate(self.paths) 
                         if i in val_files]
            
            train_files = [i for i in range(len(self.paths)) if i not in val_files] if train_files is None else train_files
            trn_paths = [path for i, path in enumerate(self.paths) 
                         if i in train_files]
        else:
            val_frac = int(len(self.paths) * val_frac)
            val_paths = self.paths[:val_frac]
            trn_paths = self.paths[val_frac:]

        return (dict(paths=trn_paths),
                dict(paths=val_paths),
                dict([]))

    def get_dset(self, **kwargs):
        dset = self.DSET_TYPE(**kwargs)
        dset.read_sources()

        return dset

    @classmethod
    def add_argparse_args(self, parser, *args, **kwargs):
        parser = super(DataModule, self).add_argparse_args(parser, *args, **kwargs)
        parser.add_argument('--volume', type=str, default=None)
        parser.add_argument('--val_frac', type=float, default=0.3)
        parser.add_argument('--val_files', type=int, default=None, nargs="*")
        parser.add_argument('--train_files', type=int, default=None, nargs="*")
        parser.add_argument('--shuffle_data_module', type=int, default=0)
        parser.add_argument('--max_nr_paths', type=int, default=None)
        return parser


class MetaVarDataModule(DataModule):
    DSET_TYPE = HyPlantMetaVarReader

    def __init__(self, *args, **kwargs):
        super(MetaVarDataModule, self).__init__(*args, **kwargs)
