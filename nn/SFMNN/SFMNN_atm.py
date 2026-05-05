import copy
from collections import OrderedDict

import os
from os.path import join as pjoin

import torch
import xarray as xr
import numpy as np
from torch import nn

import fluomapper
from fluomapper.utils.gaussian import Resampler
from fluomapper.nn.SFMNN.SFMNN_helpers import WeightMask, Clamp, AttentionReducer, Unsqueeze, SepConv,\
        ImageMean, Param, ParametrizedMatrixMult
from fluomapper.utils.nn import ApplyOnImage
from fluomapper.nn._base.heads import PCAHead, ConstantHead
from fluomapper.nn.simulation_mlp.mlp import _MLP
from fluomapper.utils import data as da
from fluomapper.utils.data import search_spectral_window, select
from fluomapper.utils.func.interp1d import Interpolate
from fluomapper.utils.run import add_model_specific_args, split_kwargs

import fluomapper.path_config
from fluomapper.path_config import HYPLANT_METAVAR_PATH

from pyspectral.solar import SolarIrradianceSpectrum, TOTAL_IRRADIANCE_SPECTRUM_2000ASTM



class ScalableLin(nn.Module):
    def __init__(self, out_wvls=None, window=None):
        super(ScalableLin, self).__init__()

        if out_wvls is not None:
            self.window = out_wvls[0], out_wvls[-1]

        else:
            self.window = window

        # LOAD spectra
        # "resample" -> fwhm = 0, i.e. just read out at closest wvls
        # dm = ToysetDataModule(resample_obs=True, resample_label=False,)  # fwhm=0)
        # resampler = dm.train_dataloader().dataset.resampler
        # dat_spectra = dm.spectra

        # spectra = torch.from_numpy(resampler.resample(dat_spectra))
        # wvls = spectra[0, 0]

        ssi_data = xr.load_dataset(pjoin(SSI_DATA_PATH, 'NOAA_yearly.nc'))
        ssi = ssi_data.SSI[-1, :].data  # get 2020
        wvls = torch.from_numpy(ssi_data.coords['wavelength'].data)
        # sensor_wvls, sensor_fwhm = da.load_sensor_types('hyplant')

        resampled_ssi = torch.from_numpy(Resampler(wvls, new_wvls=out_wvls, fwhm=0.33).resample(ssi) * 1000).float()

        if self.window is not None:
            spectral_window = da.search_spectral_window(*self.window, where=out_wvls)
            resampled_ssi = da.select(windows=spectral_window, signals=resampled_ssi, axis=-1)

        self.lin = nn.Parameter(resampled_ssi).requires_grad_(False)

    def forward(self, weight, *args, **kwargs):
        # if weight.device != self.lin.device:
        #     self.lin = self.lin.to(weight.device)
        if len(weight.squeeze().shape) == 1:
            return torch.einsum('b, n -> bn', weight.squeeze(), self.lin)

        else:
            return torch.einsum('bn, n -> bn', weight.squeeze(), self.lin)


class ACVModel(nn.Module):
    def __init__(self, out_wvls, dim_in, unenc_dim_in, dims_spatial, on_inp=True, on_enc=False, on_inp_enc=False,
                 on_off_nadir=False, on_meta_vars=None, acv_mode='mult', pca_reduce=True, pca_n_components=None,
                 **kwargs):
        super(ACVModel, self).__init__()

        self.on_inp = on_inp
        self.on_enc = on_enc
        self.on_inp_enc = on_inp_enc
        self.on_off_nadir = on_off_nadir
        self.on_meta_vars = on_meta_vars
        if on_inp:
            self.dim_in = unenc_dim_in

        elif on_enc:
            self.dim_in = dim_in

        elif on_inp_enc:
            self.dim_in = dim_in + unenc_dim_in

        elif on_off_nadir:
            self.dim_in = 2

        elif on_meta_vars:
            self.dim_in = len(on_meta_vars)

            if 'off_nadir' in on_meta_vars:
                self.dim_in += 1  # off_nadir is included as [x, x**2]

        else:
            raise NotImplementedError('Either on_inp, on_enc, on_inp_enc or on_off_nadir must be True')

        self.dims_spatial = dims_spatial
        self.acv_model = nn.Sequential(*nn.ModuleList(
                                                      [nn.Sequential(SepConv(dim_in=self.dim_in,
                                                                             dim_out=self.dims_spatial[0],
                                                                             kernel_size=3, padding=1),
                                                                     nn.BatchNorm2d(self.dims_spatial[0]),
                                                                     nn.ReLU())]

                                                      +
                                                      [nn.Sequential(SepConv(dim_in=dim_in, dim_out=dim_out,
                                                                             kernel_size=3, padding=1),
                                                                     nn.BatchNorm2d(dim_out),
                                                                     nn.ReLU())
                                                                for dim_in, dim_out in zip(self.dims_spatial[1:-1],
                                                                                       self.dims_spatial[2:])]

                                                     ))
        self.acv_mode = acv_mode
        if acv_mode == 'mult':
            out_dim = 1

        elif acv_mode == 'linear':
            out_dim = 2

        elif 'pca' in acv_mode:

            if kwargs['meta_vars_volume'] is None:
                kwargs['meta_vars_volume'] = HYPLANT_METAVAR_PATH

            if acv_mode in ('pca_add', 'pca'):
                components = np.load(pjoin(kwargs['meta_vars_volume'], 'acv/acv_pca_components.npy'))
                wvls = np.load(pjoin(kwargs['meta_vars_volume'], 'acv/acv_pca_components_wvls.npy'))

            elif acv_mode == 'pca_mult':
                components = np.load(pjoin(kwargs['meta_vars_volume'], 'acv/acv_pca_components_mult.npy'))
                wvls = np.load(pjoin(kwargs['meta_vars_volume'], 'acv/acv_pca_components_wvls.npy'))

            else:
                raise NotImplementedError(f'{acv_mode} is not a valid mode')

            if pca_n_components is not None:
                components = components[:pca_n_components]
                
            # cut components to right wvls
            spectral_window = search_spectral_window(out_wvls[0], out_wvls[-1], where=wvls)
            components = select(components, spectral_window, axis=1)
            wvls_components = select(wvls, spectral_window)

            components = torch.from_numpy(components).float().requires_grad_(False)
            if len(wvls_components) != len(out_wvls):
                components = Interpolate(torch.from_numpy(wvls_components), out_wvls).forward(components).float()

            self.components = nn.Parameter(components, requires_grad=False)
            out_dim = self.components.shape[0]

        else:
            raise NotImplementedError('acv_mode must be in ("mult", "linear", "pca")')

        self.pca_reduce = pca_reduce
        if 'pca' in acv_mode and pca_reduce:
            self.out = nn.Sequential(SepConv(self.dims_spatial[-1], out_dim, kernel_size=3, padding=1),
                                     ImageMean(keep_channel=True))
        else:
            self.out = SepConv(self.dims_spatial[-1], out_dim, kernel_size=3, padding=1)

        self.xs = None

    def apply_acv(self, L, acv, recalc_xs=False):
        if self.xs is None or recalc_xs:
            # if window is odd shfit x range
            is_even = L.shape[-2] % 2 == 0
            self.xs = torch.arange(-L.shape[-2] // 2, L.shape[-2] // 2 + is_even, 
                                   device=acv.device).requires_grad_(False)

        if self.acv_mode in ('mult', 'linear'):
            acv_m = torch.einsum('i, b -> bi', self.xs, acv[:, 0])

            if acv.shape[1] == 2:
                acv_m = acv_m + nn.functional.relu(acv[:, [1]])
            acv_m += 1

            ret = torch.einsum('bcij, bi -> bcij', L, acv_m)

        elif self.acv_mode in ('pca', 'pca_add'):
            modeled_comp = torch.einsum('kc, bk... -> bc...', self.components, acv)

            if self.pca_reduce:
                modeled_comp = modeled_comp[..., None, None]

            ret = L + modeled_comp

        elif self.acv_mode == 'pca_mult':
            modeled_comp = torch.einsum('kc, bk... -> bc...', self.components, acv)

            if self.pca_reduce:
                modeled_add = torch.einsum('bcij, bc -> bcij', L, modeled_comp)

            else:
                modeled_add = torch.einsum('bcij, bcij -> bcij', L, modeled_comp)
            
            ret = L + modeled_add

        return ret

    def forward(self, x, **kwargs):
        if self.on_inp:
            x = x[1]

        elif self.on_enc:
            x = x[0]

        elif self.on_inp_enc:
            x = torch.cat(x, dim=1)

        elif self.on_off_nadir:
            x = kwargs['off_nadir']
            x = torch.cat([x, x**2], dim=1)

        elif self.on_meta_vars:
            x = torch.cat([kwargs[var] for var in self.on_meta_vars if var not in ('off_nadir', )], dim=1)

            if 'off_nadir' in self.on_meta_vars:
                x = torch.cat([x, kwargs['off_nadir'], kwargs['off_nadir']**2], dim=1)

        x = self.out(self.acv_model(x))

        return x

    @classmethod
    def add_model_specific_args(cls, parser, prepend="", **kwargs):
        parser_spec = dict(
            [('dims_spatial', dict(type=int, nargs='+', default=None)),
             ('on_inp', dict(type=int, default=0)),
             ('on_enc', dict(type=int, default=0)),
             ('on_inp_enc', dict(type=int, default=0)),
             ('on_off_nadir', dict(type=int, default=0)),
             ('on_meta_vars', dict(type=str, default=None, nargs="+")),
             ('acv_mode', dict(type=str, default='mult')),
             ('pca_reduce', dict(type=int, default=1)),
             ('pca_n_components', dict(type=int, default=None))
             ])
        parser = add_model_specific_args(parser, parser_spec=parser_spec, prepend=prepend)
        return parser


class T14Forward(nn.Module):
    T14_names = ('toa_sun', 'rso', 'rdd', 'tss', 'tsd', 'too', 'tdo', 'tsstoo',
                 'tsdtoo', 'tsstdo', 'tsdtdo', 'tssrdd', 'toordd', 'tssrddtoo')

    IDS_MODE_APPEND = ('enc_append', )
    IDS_MODE_MATRIX_ID = ('matrix_id', 'matrix_id_all', 'enc_all_w_skip_w_matrix_id')

    downwelling_constant_vars = ('rso', 'tss', 'tsd')

    def __init__(self, out_wvls, dims=None, unenc_dim_in=None, dim_in=None, dim_out=None,
                 weight_model='mlp', var_min_atm=0.9, var_max_atm=1.1, var_min_toa_sun=0.6,
                 var_max_toa_sun=1.05, to_hyplant_ssi=True, toa_fixed=False, resample_fwhm=None,
                 with_acv=False, toa_sun_on_inp=False, spectral_inp_len=None, toa_sun_no_inp=False,
                 with_ids=False, ids_mode=None, data_source_id_dict=None, id_size=None, enc_dim=None,
                 downwelling_constant=False, *args, **kwargs):
        super(T14Forward, self).__init__()

        self.out_wvls = out_wvls
        self.unenc_dim_in = unenc_dim_in
        self.dim_in = dim_in
        self.enc_dim = enc_dim
        self.dim_out = dim_out
        self.dims = dims

        self.toa_fixed = toa_fixed

        self.var_min_atm = var_min_atm
        self.var_max_atm = var_max_atm
        self.var_min_toa_sun = var_min_toa_sun
        self.var_max_toa_sun = var_max_toa_sun

        self.weight_model = weight_model
        self.to_hyplant_ssi = to_hyplant_ssi

        self.resample_fwhm = resample_fwhm

        self.predict_t14_name_set = self.T14_names[:7]
        self.complete_t14_name_set = self.T14_names
        
        self.with_acv = with_acv
        self.acv_kwargs = split_kwargs(kwargs, 'acv')

        self.toa_sun_on_inp = toa_sun_on_inp
        self.toa_sun_no_inp = toa_sun_no_inp
        self.spectral_inp_len = spectral_inp_len

        self.with_ids = with_ids
        self.ids_mode = ids_mode
        self.id_size = id_size

        if self.with_ids:
            self.data_source_id_dict = data_source_id_dict

        self.downwelling_constant = downwelling_constant
        # if self.downwelling_constant:
        #     assert self.with_acv and ('w_skip' in self.ids_mode or self.ids_mode in self.IDS_MODE_APPEND)

    def create_acv_model(self, out_wvls):
        self.acv_model = ACVModel(out_wvls=out_wvls, dim_in=self.dim_in, unenc_dim_in=self.unenc_dim_in,
                                  **self.acv_kwargs)

    def create_atmo_net(self, pred_vars, var_ranges, var_lims, weight_model, head_mode, dim_in, unenc_dim_in, dim_out,
                        dims, out_wvls, *args, **kwargs):

        # DEFINE model parts
        # # define weight model
        self.weight_model = weight_model if weight_model is None or weight_model == 'mean' \
                                         else WeightMask(weight_model=weight_model, dims=dims, dim_in=unenc_dim_in,
                                                         *args, **kwargs)

        # # define t14 predictors
        t14_modules = []
        clamp_modules = []

        if self.with_ids and self.ids_mode in self.IDS_MODE_APPEND:
            dim_in += self.id_size

        if self.toa_fixed:
            head_mode['toa_sun'] = 'fixed'

        for var in self.complete_t14_name_set:
            clamp = Clamp(*var_lims[var])
            clamp_modules.append((var, clamp))

        for var in pred_vars:
            var_range = var_ranges[var]
            dims_ = copy.deepcopy(dims)
            dim_in_ = dim_in

            if head_mode[var] == 'pca':
                pca_head_input_dim = kwargs['n_components'] + kwargs['with_mult'][var]
                n_components = kwargs['n_components']

                if var == 'toa_sun':
                    pca_head_input_dim = 1
                    n_components = 0
                    dim_in_ = dim_in if not self.toa_sun_on_inp else self.spectral_inp_len
                    dim_in_ = dim_in_ if not self.toa_sun_no_inp else 0

                elif var in self.downwelling_constant_vars and self.downwelling_constant:
                    dim_in_ = dim_in - self.enc_dim

                if n_components > 0:
                    components = kwargs['components'][var][:n_components]

                else:
                    components = None

                # create PCA head
                head = PCAHead(out_wvls=out_wvls, components=components,
                               mean=kwargs['means'][var], std=kwargs['stds'][var], var_range=var_range,
                               with_mult=kwargs['with_mult'][var], mult_bounds=kwargs['mult_bounds'][var])

                # create atmo encoder
                if dim_in_ != 0:
                    forward_model = ApplyOnImage(nn.Sequential(_MLP(dim_in=dim_in_, dim_out=dim_out,
                                                                    dims=dims_, *args, **kwargs),))

                else:
                    forward_model = ApplyOnImage(nn.Sequential(Param(dim_out=dim_out, *args, **kwargs),))

                atmo_encoder = AttentionReducer(forward_model=forward_model,
                                                weight_model=self.weight_model,
                                                encoding_to_module=(True if not (var == 'toa_sun' and self.toa_sun_on_inp)
                                                                    else False),
                                                restrain_inp_to_module=self.spectral_inp_len,
                                             *args, **kwargs)

                # create transformation from encoding to PCA head input
                if self.with_ids and self.ids_mode in self.IDS_MODE_MATRIX_ID:
                    transf = nn.Sequential(ParametrizedMatrixMult(param_dim=self.id_size,
                                                                       matrix_dim=(pca_head_input_dim, dim_out)),
                                                nn.BatchNorm1d(pca_head_input_dim),)

                else:
                    transf = nn.Sequential(nn.Linear(dim_out, pca_head_input_dim),
                                           nn.BatchNorm1d(pca_head_input_dim),)

                alpha = nn.Sequential(atmo_encoder,
                                      transf,
                                      head,
                                      Unsqueeze())

                t14_modules.append((var, alpha))

            elif head_mode[var] == 'fixed':
                alpha = ConstantHead(return_=kwargs['components'][var])
                t14_modules.append((var, alpha))

            else:
                raise NotImplementedError

        self.t14_modules = nn.ModuleDict(t14_modules)
        self.clamp_modules = nn.ModuleDict(clamp_modules)

    def cut_and_resample(self, arr, wvls=None, new_wvls=None, out_windows=None, resampler=None, do_resample=True,
                         axis=-1):

        if out_windows is None and wvls is not None:
            out_windows = search_spectral_window(new_wvls[0], new_wvls[-1], where=wvls)

        arr = select(arr, out_windows, axis=axis)

        if wvls is not None:
            wvls = select(wvls, out_windows, axis=axis)

        if resampler is None:
            resampler = Resampler(wvls, new_wvls=new_wvls, fwhm=self.resample_fwhm)

        if do_resample:
            resampled = torch.from_numpy(resampler.resample(arr, axis=-1)).float()

            wvls = new_wvls.numpy()

        elif type(arr) is not torch.Tensor:
            resampled = torch.from_numpy(arr).float()

        else:
            resampled = arr

        if wvls is not None:
            return resampled, resampler, wvls

        return resampled, resampler

    def forward(self, x, ids=None, **kwargs):

        if self.with_ids:
            x_encoded, x_inp = x

            if self.ids_mode in self.IDS_MODE_APPEND:
                x_encoded = torch.cat([x_encoded, ids], dim=1)
                x = x_encoded, x_inp

            elif self.ids_mode in self.IDS_MODE_MATRIX_ID:
                x_encoded, x_inp = x

                # ids has shape (b, id_size, s, s), since these are all the same ids we just take 
                # a single one 
                x = x_encoded, x_inp, ids[..., 0, 0]
        
        if not self.downwelling_constant:
            T14 = OrderedDict([(key, module(x)) for key, module in self.t14_modules.items()])

        else:
            T14 = OrderedDict([(key, module(x)) if key not in self.downwelling_constant_vars else
                               (key, module((x[0][:, self.enc_dim:], x[1])))
                               for key, module in self.t14_modules.items() ])

        expand_shape = (T14['too'].shape[0], T14['too'].shape[1], T14['too'].shape[2], T14['too'].shape[3])
        T14 = OrderedDict([(var, val) if val.shape == expand_shape else (var, val.expand(expand_shape))
                            for var, val in T14.items()])

        T14.update(OrderedDict(tsstoo=T14['tss']*T14['too'],
                               tsdtoo=T14['tsd']*T14['too'],
                               tsstdo=T14['tss']*T14['tdo'],
                               tsdtdo=T14['tsd']*T14['tdo'],
                               tssrdd=T14['tss']*T14['rdd'],
                               toordd=T14['too']*T14['rdd'],
                               tssrddtoo=T14['tss']*T14['rdd']*T14['too']))

        mat = torch.stack([T14[key] for key in self.complete_t14_name_set], dim=1).permute(1, 0, 2, 3, 4)

        T14 = dict(zip(list(T14.keys()), mat))

        # clamp
        T14 = dict([(var, self.clamp_modules[var](val)) for var, val in T14.items()])

        # include across-track variation
        if self.with_acv: 
            T14.update(acv=self.acv_model(x, **kwargs))

        return T14

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        parser = ACVModel.add_model_specific_args(parser, prepend='acv')

        parser.add_argument('--mult', type=int, default=1)
        parser.add_argument('--reduced_set', type=int, default=0)
        parser.add_argument('--mask', type=str, default='none')

        parser.add_argument('--to_hyplant_ssi', type=int, default=1)
        parser.add_argument('--toa_fixed', type=int, default=0)

        parser.add_argument('--var_max_atm', type=float, default=1.1)
        parser.add_argument('--var_min_atm', type=float, default=0.9)
        parser.add_argument('--var_max_toa_sun', type=float, default=1.05)
        parser.add_argument('--var_min_toa_sun', type=float, default=0.6)

        parser.add_argument('--resample_fwhm', type=float, default=None)

        parser.add_argument('--weight_model', type=str, default='mlp')

        parser.add_argument('--with_acv', type=int, default=0)
        parser.add_argument('--toa_sun_on_inp', default=0, type=int)
        parser.add_argument('--toa_sun_no_inp', default=0, type=int)

        parser.add_argument('--downwelling_constant', default=0, type=int)
        
        return parser


class T14PCA(T14Forward):
    def __init__(self, n_components=5, with_mult=False, with_mult_atm=False, mult_bounds=None, mult_bounds_atm=None,
                 fixed_toa_sun=False, meta_vars_volume=None, *args, **kwargs):
        super(T14PCA, self).__init__(*args, meta_vars_volume=meta_vars_volume, **kwargs)

        self.with_mult = with_mult
        self.with_mult_atm = with_mult_atm
        self.mult_bounds = mult_bounds
        self.mult_bounds_atm = mult_bounds_atm
        self.fixed_toa_sun = fixed_toa_sun

        if meta_vars_volume is None:
            meta_vars_volume = HYPLANT_METAVAR_PATH
        
        T14_DIR = pjoin(os.path.dirname(fluomapper.__file__),
                        'parameterization', 'T14fncts', 'hires_pca')
 
        components = np.load(pjoin(T14_DIR, 'components.npy'))
        means = np.load(pjoin(T14_DIR, 'means.npy'))
        stds = np.load(pjoin(T14_DIR, 'stds.npy'))

        wvls = np.load(pjoin(T14_DIR, 'wvls.npy'))
        wvls = torch.from_numpy(wvls)

        resampled, self.resampler, self.t14_wvls = self.cut_and_resample(arr=np.concatenate([components, stds[:, None],
                                                                                             means[:, None]], axis=1),
                                                                         wvls=wvls, new_wvls=self.out_wvls,
                                                                         do_resample=self.to_hyplant_ssi, axis=-1)

        components = dict(zip(self.predict_t14_name_set, resampled[:, :-2]))
        stds = dict(zip(self.predict_t14_name_set, resampled[:, -2]))
        means = dict(zip(self.predict_t14_name_set, resampled[:, -1]))

        solar_irr = SolarIrradianceSpectrum(TOTAL_IRRADIANCE_SPECTRUM_2000ASTM,
                                            dlambda=np.diff(self.t14_wvls).mean() / 1e3)
        solar_wvls, irr = solar_irr.wavelength, solar_irr.irradiance * 1e3 / 1e3 / 2 / np.pi
        solar_wvls = torch.from_numpy(solar_wvls * 1e3)
        irr = torch.from_numpy(irr)
        irr, _, _wvls = self.cut_and_resample(arr=torch.atleast_2d(irr),
                                              wvls=solar_wvls, new_wvls=self.t14_wvls,
                                              do_resample=True, axis=-1)

        components['toa_sun'] = irr

        self.t14_wvls = self.t14_wvls.squeeze().float()

        _kwargs = kwargs.copy()
        if not self.to_hyplant_ssi:
            _kwargs['out_wvls'] = self.t14_wvls

        else:
            _kwargs['out_wvls'] = self.out_wvls

        _kwargs['dims'] = self.dims
        _kwargs['dim_out'] = self.dim_out
        _kwargs['dim_in'] = self.dim_in
        _kwargs['dims'] = self.dims
        _kwargs['unenc_dim_in'] = self.unenc_dim_in
        _kwargs['weight_model']= self.weight_model

        _kwargs['head_mode'] = dict(zip(self.predict_t14_name_set, ['pca'] * len(self.predict_t14_name_set)))
        if self.fixed_toa_sun:
            _kwargs['head_mode']['toa_sun'] = 'fixed'

        var_lims = dict(zip(self.complete_t14_name_set, [(0, 1)] * len(self.complete_t14_name_set)))
        var_lims['toa_sun'] = (None, None)
        _kwargs['var_lims'] = var_lims

        var_ranges = dict(zip(self.predict_t14_name_set, [(self.var_min_atm, self.var_max_atm)] * len(self.predict_t14_name_set)))
        var_ranges['toa_sun'] = (self.var_min_toa_sun, self.var_max_toa_sun)
        _kwargs['var_ranges'] = var_ranges

        mult_bounds = dict(zip(self.predict_t14_name_set, [self.mult_bounds_atm] * len(self.predict_t14_name_set)))
        mult_bounds['toa_sun'] = self.mult_bounds
        _kwargs['mult_bounds'] = mult_bounds

        with_mult = dict(zip(self.predict_t14_name_set, [self.with_mult_atm] * len(self.predict_t14_name_set)))
        with_mult['toa_sun'] = self.with_mult
        _kwargs['with_mult'] = with_mult

        self.create_atmo_net(self.predict_t14_name_set,
                             n_components=n_components,
                             components=components,
                             means=means, stds=stds,
                             **_kwargs)
        
        if self.with_acv:
            self.create_acv_model(self.t14_wvls)

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        parser.add_argument('--n_components', type=int, default=3)
        parser.add_argument('--variable_components', type=int, default=0)
        parser.add_argument('--with_mult', default=0, type=int)
        parser.add_argument('--with_mult_atm', default=0, type=int)
        parser.add_argument('--mult_bounds', default=None, type=float, nargs=2)
        parser.add_argument('--mult_bounds_atm', default=None, type=float, nargs=2)
        parser.add_argument('--fixed_toa_sun', default=0, type=int)
        return parser
