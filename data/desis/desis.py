from fluomapper.data._base.base import _Reader, _BaseDataModule
from fluomapper.path_config import DESIS_DATA_PATH
from fluomapper.utils.data import search_spectral_window, select, logical_or
import fluomapper.utils.data as da

import glob, os, random
from os.path import join as pjoin
import xml
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from torch.utils.data import Dataset
from functools import partial
import re

import xarray as xr
import rioxarray as rx

import subprocess as sp
import itertools as it


def NDVI(spectrum, wvls, axis=-1):
    MIR_wvls = search_spectral_window(770, 810, where=wvls)
    NIR_wvls = search_spectral_window(650, 680, where=wvls)

    MIR = torch.mean(select(spectrum, MIR_wvls, axis=axis), axis=axis)
    NIR = torch.mean(select(spectrum, NIR_wvls, axis=axis), axis=axis)

    return (MIR - NIR + 1e-5) / (MIR + NIR + 1e-5)
                    

class DESISReader(_Reader):
    # Define relevant meta info names in .xml file
    __META_FILE_TAGS_PER_BAND__ = dict(wavelengthCenterOfBand='wavelength', 
                                       wavelengthWidthOfBand='fwhm',
                                       gainOfBand='gain',
                                       offsetOfBand='offset')

    __META_FILE_TAGS__ = dict(
                              parm2=("specific/sunZenithAngle", 1),
                              sunAzimuthAngle=("specific/sunAzimuthAngle", 1),
                              sceneAzimuthAngle=("specific/sceneAzimuthAngle", 1),
                              bckzen=("specific/sceneIncidenceAngle", 1),
                              h1alt=("base/altitudeCoverage", 1e-3),
                              h2alt=("specific/terrain/meanGroundElevation", 1e-3),
                              matchingMethod=('specific/matchingMethod', None),
                              orthoRMSE_x=('specific/orthoRMSE_x', 1),
                              orthoRMSE_y=('specific/orthoRMSE_y', 1)

                            )

    __META_INFO_VARS__ = ['neg_AOT', 'h2ostr']
    
    def __init__(self, include_wvl_info=True, norm=10, return_source_ids=True, *args, **kwargs):
        """
        Basic DESIS Reader, works for any L1B, L1C, L2A. Paths need to point to the DESIS .tif file in a directory with
        valid .xml file describing the meta info.

        :param include_wvl_info: whether to read the .xml file
        :param args:
        :param kwargs:
        """
        self.include_wvl_info = include_wvl_info

        if return_source_ids and not 'source_id' in getattr(kwargs, 'meta_info', []):
            kwargs['meta_info'].append('source_id')
        
        vars_ = list(self.__META_FILE_TAGS__.keys()) + self.__META_INFO_VARS__ 
        super().__init__(sensor_type='desis', norm=norm, return_source_ids=return_source_ids, *args, **kwargs)

        self.META_INFO_KEYS.update(dict(zip(vars_, [partial(self._getter, key=var) for var in vars_])))
        self.META_INFO_KEYS.update(parm1=self.get_parm1,
                                   off_nadir=self.get_off_nadir, 
                                   dist=self.get_dist, 
                                   ndvi=self.get_ndvi,
                                   tilt=self.get_tilt)

    def _getter(self, source_index, indices, key, shape=None, *args, **kwargs):
        source_meta = self.sources_meta[source_index]

        if np.isscalar(source_meta.attrs[key]):
            val = source_meta.attrs[key]
            
            if shape is not None:
                val = np.ones(shape) * val
                val = val[..., None]  # add channel dimension

        else:
            source = source_meta.attrs[key]
            
            win = [source[i[0]: i[0] + self.out_shape,
                          i[1]: i[1] + self.out_shape
                         ]
                   for i in indices]

            win = [w for w in win if not 0 in w.shape]
            
            if len(win) == 0:
                return None
            else:
                val = np.stack(win)[..., None]
            
        return val.astype(np.float32)

    def get_tilt(self, source_index, indices, shape=None, *args, **kwargs):
        bckzen = self.META_INFO_KEYS['bckzen'](source_index, indices, shape=shape, *args, **kwargs)
        h1alt = self.META_INFO_KEYS['h1alt'](source_index, indices, shape=shape, *args, **kwargs)
        factor = 6371 / (6371 + h1alt)

        tilt = np.arcsin(factor * np.sin(np.pi - bckzen * np.pi / 180)) * 180 / np.pi
        return tilt.astype(np.float32)

    def get_parm1(self, source_index, indices, shape=None, *args, **kwargs):
        source_meta = self.sources_meta[source_index]
        val = np.abs(source_meta.attrs['sunAzimuthAngle'] - source_meta.attrs['sceneAzimuthAngle'])
        if shape is not None:
            val = np.ones(shape) * val
            val = val[..., None] # add a var dimension 

        return val.astype(np.float32)

    def get_dist(self, source_index, indices, shape=None, *args, **kwargs):
        dist = np.stack([np.repeat(np.atleast_2d(np.arange(i[0], i[0] + shape[-2])[:, None]),
                                        shape[-1], axis=1) for i in indices])[:, None].transpose(0, 3, 2, 1)

        return dist.astype(np.float32)

    def get_off_nadir(self, source_index, indices, shape, **kwargs):
        onad = np.stack([np.repeat(np.atleast_2d(np.arange(i[0] - 1024 / 2, i[0] - 1024 / 2 + shape[-1])),
                                        shape[-2], axis=0) for i in indices])[:, None].transpose(0, 3, 2, 1)

        return onad.astype(np.float32)

    def get_ndvi(self, source_index, indices, shape, win, **kwargs):
        ndvi = NDVI(win, self.wavelengths[source_index]).unsqueeze(-1)
        return ndvi.float()

    def _complete_sources(self, sources, sources_meta):
        
        if not self.include_wvl_info:
            return sources, sources_meta
        
        new_sources = []
        source_gains = []
        source_offsets = []

        for s, s_meta in zip(sources, sources_meta):
            dirname = os.path.dirname(s_meta.attrs['path'])
            meta_fil = glob.glob(pjoin(dirname, '*-METADATA.xml'))[0]
            meta = xml.etree.ElementTree.parse(meta_fil)
            
            is_l1b = False 
            if 'L1B' in dirname:
                is_l1b = True
                try:
                    meta_l1c_fil = glob.glob(pjoin(dirname.replace('L1B', 'L1C'), '*-METADATA.xml'))[0]
                    meta_l1c = xml.etree.ElementTree.parse(meta_l1c_fil)

                except IndexError as e:
                    print('Could not find L1C METADATA.xml')
                    meta_l1c = None

            else:
                meta_l1c = meta
                
            band_keys = list(self.__META_FILE_TAGS_PER_BAND__.keys())
            band_info = dict([(name, []) for name in band_keys])
            
            # needed to make sure we count in the right order
            band_perm = []

            # read meta data per band
            for band_node in meta.findall('specific/bandCharacterisation/band'):
                info = dict([(node.tag, float(node.text)) for node in band_node if node.tag in band_keys])
                band_perm.append(int((band_node.find('bandNumber').text)))
                for key in band_keys:
                    if key in info:
                        band_info[key].append(info[key])
                    else:
                        band_info[key].append(-9999)
            
            # reverse permutation 
            band_perm = np.argsort(band_perm)

            coords = dict([(self.__META_FILE_TAGS_PER_BAND__[k], ('band', np.array(v)[band_perm]))
                           for k, v in band_info.items()])

            s_meta.coords.update(coords)

            # read global meta data
            for var, (name, mult) in self.__META_FILE_TAGS__.items():
                try:
                    node = meta.findall(name)[0]

                except IndexError:
                    if is_l1b and meta_l1c is not None:
                        print(f'This is a L1B file. Trying to find {name}:{var} in L1C.')
                        try:
                            node = meta_l1c.findall(name)[0]
                           
                        except IndexError:
                            print(f'Could not find {var} ({name}) in meta vars file.')

                    elif meta_l1c is None:
                         print(f'Could not find {var} ({name}) in meta vars file.')

                if mult is not None:
                    s_meta.attrs[var] = float(node.text) * mult

                else:
                    s_meta.attrs[var] = node.text 

            # assign to output
            new_sources.append(s)
            
            source_gains.append(band_info['gainOfBand'])
            source_offsets.append(band_info['offsetOfBand'])
            
            if not 'L2A' in dirname:
                if is_l1b:
                    atmo_vars_fil = glob.glob(pjoin(dirname.replace('L1B', 'L2A'), '*QL_QUALITY-2.tif'))

                    if len(atmo_vars_fil) > 0:
                        atmo_vars_fil = atmo_vars_fil[0]
                        atmo_vars = rx.open_rasterio(atmo_vars_fil)
                    
                        AOT = atmo_vars[-2].data / 100
                        h2ostr = atmo_vars[-1].data / 42
                        s_meta.attrs['neg_AOT'] = - np.mean(AOT[np.where(AOT > 0)].astype(float))
                        s_meta.attrs['h2ostr'] = np.mean(h2ostr[np.where(h2ostr > 0)].astype(float))

                else: 
                    atmo_vars = rx.open_rasterio(glob.glob(pjoin(dirname.replace('L1C', 'L2A'), '*QL_QUALITY-2.tif'))[0])

                    AOT = atmo_vars[-2].data / 100
                    h2ostr = atmo_vars[-1].data / 42
                    s_meta.attrs['neg_AOT'] = -AOT
                    s_meta.attrs['h2ostr'] = h2ostr

        source_gains = np.stack(source_gains)
        source_offsets = np.stack(source_offsets)

        self.source_gains = source_gains
        self.source_offsets = source_offsets
            
        return new_sources, sources_meta

    def calibrate(self, source_ind, px):
        gain = self.source_gains[source_ind]
        offset = self.source_offsets[source_ind]

        return px * gain + offset
    
    @classmethod
    def add_argparse_args(cls, parser, *args, **kwargs):
        super(DESISReader, cls).add_argparse_args(parser, *args, **kwargs)
        parser.add_argument('--off_nadir_mode', type=str, default=None)
        return parser



class DESISQualityReader(_Reader):
    BAND_NAMES = dict(shadow=(0, 1), land=(1, 1), snow=(2, 1), haze_land=(3, 1), haze_water=(4, 1), 
                      cloud_land=(5, 1), cloud_water=(6, 1), water=(7, 1), neg_AOT=(8, -100), h2ostr=(9, 42))

    SPECIAL_META_VARS = dict(ql_mask_w_haze_w_shadow=lambda x: logical_or(x['snow'], x['haze_land'], x['haze_water'], 
                                                          x['cloud_water'], x['cloud_land'], x['water'], x['shadow']), 
                             ql_mask_w_haze=lambda x: logical_or(x['snow'], x['haze_land'], x['haze_water'], 
                                                          x['cloud_water'], x['cloud_land'], x['water']), 
                             ql_mask=lambda x: logical_or(x['snow'], x['cloud_water'], x['cloud_land'], x['water']))

    ALL_VARS = list(BAND_NAMES.keys()) + list(SPECIAL_META_VARS.keys())

    def __init__(self, out_vars, *args, **kwargs):
        super(DESISQualityReader, self).__init__(*args, **kwargs)
        
        self.out_vars = out_vars

    def __getitem__(self, *args, **kwargs):
        ret = super(DESISQualityReader, self).__getitem__(*args, **kwargs)

        is_tuple = type(ret) is tuple
        if is_tuple:
            ret, _ = ret

        all_ = dict()
        for band_name, (band, norm) in self.BAND_NAMES.items():
            all_[band_name] = ret[..., [band]] / norm
        
        out = dict()
        for var in self.out_vars:
            if var in self.BAND_NAMES:
                out[var] = all_[var]

            elif var in self.SPECIAL_META_VARS:
                out[var] = self.SPECIAL_META_VARS[var](all_)

            else:
                raise Exception(f'Unknown var {var}. Only {self.BAND_NAMES.keys()}' 
                                 ' and {self.SPECIAL_META_VARS.keys()} are valid.')

        if is_tuple:
            return out, _

        return out


class DESISMetaVarReader(Dataset):
    READER = DESISReader
    META_VAR_READER = _Reader
    QUALITY_READER = DESISQualityReader

    __WVL_SHORT_CUTS__ = {740:741.9, 760:760, 750:752.2, 775:775.2, 780:780.5, 755:755.0, 745:744.4}

    def __init__(self, paths, path_ids=None, meta_vars=None, data_volume=None, shard_along=None, shard_along_nr_files_per_proc=1,
                 *args, **kwargs):
        """
        Defines the most general data set reading set up, i.e. spectral DESIS (L1C, L1B), reflectance (L2A), meta info
        saved in the .xml files as well as any other meta var (.tif).

        :param meta_vars:
        :param data_volume:
        :param args:
        :param kwargs:
        """
        super(DESISMetaVarReader, self).__init__()

        self.meta_vars = meta_vars if meta_vars is not None else []
        meta_readers = []

        if not type(paths) is list:
            raise Exception(f'paths argument must be a list. you provided {type(paths)}')
        
        kwargs['read_sources'] = False

        if data_volume is None:
            self.data_volume = DESIS_DATA_PATH
        else:
            self.data_volume = data_volume

        join = lambda dir_, p : pjoin(dir_, p) if not os.path.exists(p) else p
        if type(paths[0]) in (tuple, list):
            self.paths = [tuple([join(self.data_volume, pp) for pp in p]) for p in paths]

        else:
            self.paths = [join(self.data_volume, p) for p in paths]
       
        if path_ids is not None and len(path_ids) == 1 and path_ids[0] == 'date':
            path_names = [self.get_desis_date(os.path.basename(p[0])) for p in self.paths]
            self.path_ids = [int(p.replace('T', '')) for p in path_names]

        elif path_ids is None:
            self.path_ids = np.arange(len(self.paths))

        else:
            self.path_ids = path_ids

        self.shard_along = shard_along
        if self.shard_along == 'none' or shard_along == 'None':
            self.shard_along = None

        if self.shard_along is not None:
            self.global_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

            if self.shard_along == 'files':
                nr_files = shard_along_nr_files_per_proc
                ind = min(self.global_rank, len(self.paths) // nr_files - 1)
                self.paths = self.paths[ind * nr_files: (ind + 1) * nr_files]
                self.path_ids = self.path_ids[ind * nr_files: (ind + 1) * nr_files]

                print(f'LOADING FILE: rank {self.global_rank} loads {self.paths} \n')

            elif self.shard_along is not None:
                raise Exception(f'shard_along {shard_along} strategy is not implemented.')

            else:
                pass

        # put in standard form when rho path is provided
        if type(paths[0]) in (tuple, list):
            paths_obs = [p[0] for p in self.paths]

        else:
            paths_obs = self.paths
        
        kwargs['path_ids'] = self.path_ids
        self.readers = [DESISReader(paths=paths_obs, *args, **kwargs)]

        # make sure we only load meta_info of first reader
        kwargs['meta_info'] = []

        # if multiple rho wvls are in meta vars, let them be handled by a single reader
        self.rho_wvls = np.array([self.__WVL_SHORT_CUTS__[int(v[len('rho'):])] 
                                  for v in self.meta_vars if v.startswith('rho')])
        self.rho_labels = [v for v in self.meta_vars if v.startswith('rho')]

        self.ql_meta_vars = [v for v in self.meta_vars if v in DESISQualityReader.ALL_VARS]

        self.meta_vars = [v for v in self.meta_vars if not v.startswith('rho') 
                                                    and not v in DESISQualityReader.ALL_VARS]

        if len(self.ql_meta_vars) > 0:
            self.meta_vars += ['ql']

        if len(self.rho_wvls) > 0:
            self.meta_vars += ['rho']

        for var in self.meta_vars:
            kwargs.update(dict(paths=self.paths))
            paths_, other_vars, reader = self.get_var_reader(var, **kwargs)

            meta_dict = copy.deepcopy(kwargs)
            meta_dict.update(dict(paths=paths_, variables=var))
            meta_dict.update(other_vars)

            meta_readers.append(reader(*args, **meta_dict))

        for i, reader in enumerate(meta_readers):
            self.readers.insert(1 + i, reader)

        self.keys = list(self.meta_vars)

    def get_desis_date(self, p):
        return str(re.match('DESIS-HSI-(.*?)-(.*?)_(.*?)-(.*)-V.*', os.path.basename(p))[4])

    def get_var_reader(self, var, paths=None, **kwargs):

        def find_path(base_path, var):
            splits = base_path.split(os.sep)
            idx = [True if f'DESIS.HSI.{var}' in split or f'DESIS-HSI-{var}' in split else False 
                    for split in splits].index(True) + 1
            splits[idx] = '*'

            glob_path = os.sep.join(splits)
            return glob.glob(glob_path)[0]

        if type(paths[0]) is str or len(paths[0]) == 1:
                if 'L1B' in paths[0]:
                    var = 'L1B'

                elif 'L1C' in paths[0]:
                    var = 'L1C'

                else:
                    raise NotImplementedError('You did not provide a L2A path. Searching the correct path failed since '
                                              'the base path must be a L1B or L1C product')

                paths = [(p, find_path(p.replace(var, 'L2A'), 'L2A')) for p in paths]

        if var == 'dem':
            paths = [pjoin(os.path.dirname(path[0]), '_geolayer')
                     for path in paths]

            for i, path in enumerate(paths):
                if not os.path.exists(path):
                    paths[i] = path.replace('L1B', 'L2A')
           
            other_vars = dict(read_specs=dict(cols=1024, rows=None, channels=3, min_=0, get_channel=2, 
                                              driver='bil', move_to_non_neg=False), 
                              norm=1, return_source_ids=False,)
            reader = self.META_VAR_READER
        
        elif var == 'rho':
            paths = [p[1] for p in paths]

            other_vars = dict(exact_wvls=self.rho_wvls, norm=1, return_source_ids=False)
            reader = self.READER

        elif var == 'full_rho':
            paths = [p[1] for p in paths]

            other_vars = dict(exact_wvls=None, norm=1, return_source_ids=False)
            reader = self.READER
            
        elif var == 'cloud':
            paths = [p[1] for p in paths]
            paths = [pjoin(os.path.dirname(p), 'MASK_buffer10.tif') for p in paths]

            other_vars = dict(norm=1, return_source_ids=False)
            reader = self.META_VAR_READER

        elif var == 'ql':
            paths = [p[1] for p in paths]
            paths = [glob.glob(pjoin(os.path.dirname(p), '*QL_QUALITY-2.tif'))[0] for p in paths]

            other_vars = dict(norm=1, return_source_ids=False, out_vars=self.ql_meta_vars)
            reader = self.QUALITY_READER

        elif var == 'smile_corr':
            paths = [p[0].replace('_nosmilecorr', '').replace('-nosmilecorr', '') for p in paths]

            other_vars = dict(return_source_ids=False, out_vars=None)
            reader = self.READER


        else:
            raise NotImplementedError(f'Variable {var} not known.')

        return paths, other_vars, reader

    def __getitem__(self, item, **kwargs):

        ret = [reader.__getitem__(item, **kwargs) for reader in self.readers]
        meta_vars = dict(list(zip(self.keys, ret[1:])))
        
        is_return_wvl = type(ret[0]) is tuple

        if len(self.rho_wvls) > 0: 
            for i, label in enumerate(self.rho_labels):
                if not is_return_wvl:
                    meta_vars[label] = meta_vars['rho'][..., [i]]
                else:
                    meta_vars[label] = (meta_vars['rho'][0][..., [i]], meta_vars['rho'][1][..., [i]])
            del meta_vars['rho']

        if len(self.ql_meta_vars) > 0:
            if is_return_wvl:
                meta_vars.update(dict([(m, (var, meta_vars['ql'][1])) for m, var in meta_vars['ql'][0].items()]))
            else:
                meta_vars.update(meta_vars['ql'])
            del meta_vars['ql']

        if not is_return_wvl:
            # item[0] is a tensor, create a dict with keys corresponding to the meta vars
            if type(ret[0]) is not dict:
                x = dict(obs=ret[0])
                x.update(meta_vars)

            # item is a dict with tensors for each meta info, put meta vars together with meta_info
            else:
                ret[0].update(meta_vars)
                x = ret[0]
 
        else:
            # item[0] is a tensor, create a dict with keys corresponding to the meta vars, preserve wvls
            if type(ret[0][0]) is not dict:
                x = dict(obs=ret[0][0])
                x.update(meta_vars)
                x = (x, ret[0][1])

            # item[0] is a tensor, create a dict with keys corresponding to the meta vars, preserve wvls
            else:
                ret[0][0].update(meta_vars)
                x = ret[0]

        return x, None

    def __len__(self):
        return len(self.readers[0])

    def read_sources(self):
        for reader in self.readers:
            reader.read_sources()

        # synchronize readers
        for reader in self.readers[1:]:
            reader.synchronize(self.readers[0])

    @classmethod
    def add_argparse_args(cls, parser, *args, **kwargs):
        parser = cls.READER.add_argparse_args(parser, *args, **kwargs)
        parser.add_argument('--meta_vars', type=str, default=None, nargs="*")
        parser.add_argument('--data_volume', type=str, default=None)
        parser.add_argument('--spectral_window_label_wvl', type=float, default=None, nargs='*')
        parser.add_argument('--shard_along', type=str, default=None)
        parser.add_argument('--shard_along_nr_files_per_proc', type=int, default=1)
        parser.add_argument('--path_ids', type=str, default=None, nargs='+')
        return parser


class DESISModule(_BaseDataModule):
    DSET_TYPE = DESISReader

    def __init__(self, paths=None, val_frac=0.5, val_files=None, train_files=None, *args, **kwargs):
        super(DESISModule, self).__init__(*args, **kwargs)
        
        self.paths = paths if paths is not None else []
        self.masks = None
        
        if self.paths_file is not None:
            for p in self.paths_file:
                paths = self.load_paths(p)
                paths = self._exclude_paths(paths)
                self.paths.append(paths)

            self.paths = list(it.chain(*self.paths))

        self.val_frac = val_frac
        self.val_files = val_files
        self.train_files = train_files

    def prepare_data(self):
        self.trn_data, self.val_data, self.tst_data = self.assemble_data(self.val_frac, self.val_files, self.train_files)

    def assemble_data(self, val_frac=None, val_files=None, train_files=None, *args, **kwargs):
        if (val_files is not None or train_files is not None) and self.paths is not None and len(self.paths) > 0:

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

        if self.masks is None:
            return (dict(paths=trn_paths),
                    dict(paths=val_paths),
                    dict([]))

        else:
            return (dict(paths=trn_paths, masks=[self.masks[os.path.basename(p)] for p in trn_paths]),
                    dict(paths=val_paths, masks=[self.masks[os.path.basename(p)] for p in val_paths]),
                    dict([]))

    def get_dset(self, **kwargs):
        dset = self.DSET_TYPE(**kwargs)
        dset.read_sources()

        return dset
    
    @classmethod
    def add_argparse_args(cls, parser, *args, **kwargs):
        parser = super(DESISModule, cls).add_argparse_args(parser, *args, **kwargs)
        parser.add_argument('--val_frac', type=float, default=0.3)
        parser.add_argument('--val_files', type=int, default=None, nargs="*")
        parser.add_argument('--train_files', type=int, default=None, nargs="*")
        parser.add_argument('--shuffle_data_module', type=int, default=0)
        return parser


class DESISMetaVarDataModule(DESISModule):
    DSET_TYPE = DESISMetaVarReader

    def __init__(self, *args, **kwargs):
        super(DESISMetaVarDataModule, self).__init__(*args, **kwargs)
