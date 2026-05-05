import torch
import torch.nn as nn

import torch.nn.functional as F

import fluomapper
from fluomapper.utils.lightning_base import _Module
from fluomapper.nn.SFMNN.SFMNN_helpers import AttentionReducer, SharedInLayer, WeightedSum, Param, ParamPredictor

from fluomapper.nn.SFMNN.SFMNN_atm import T14Forward, T14PCA
from fluomapper.utils.nn import SoftHistogram, InputNorm, ApplyOnImage, MultiSequential, differ
from fluomapper.nn._base.heads import SplineHead, LinearHead, TwoGaussians, RSquareHead
from fluomapper.utils.gaussian import GaussianSmoothing
from fluomapper.utils.func.interp1d import Interpolate
from fluomapper.utils.run import add_model_specific_args, make_iterable

from fluomapper.utils.data import search_spectral_window, select, zero_nonfinite
from fluomapper.nn.simulation_mlp.mlp import _MLP
from fluomapper.losses.losses import mutual_information, conditional_entropy, jensen_shannon_diff

import copy
import os
from os.path import join as pjoin

import numpy as np


class _SFMNN(nn.Module):
    IDS_MODE_INPUT = ('enc_all', 'enc_all_w_skip', 'enc_all_w_skip_w_matrix_id', 'matrix_id_all')
    IDS_MODE_SKIP =  ('enc_all_w_skip', 'enc_all_w_skip_w_matrix_id')

    def __init__(self, dim_in, out_wvls, at_sensor_wvls=None, dims=None, f_mode="weighted_sum", R_mode='weighted_sum',
                 max_f=20, max_R=1.0, min_R=0, min_f=0, atm_enc_dim=16, stack_enc_inp_for_f_R=False, shared_in_layer=False,
                 sim_with_aoi=False,slope_min=0, slope_max=1, offset_max=0, offset_min=1, f_slope_min=0, f_slope_max=1,
                 f_offset_max=0, f_offset_min=1, use_rdd=1, atmosphere_mode='default', mean_over_angle=False,
                 with_ids=False, ids_mode='enc_append', id_size=32,  data_sources=None, data_source_ids=None,
                 load_ids_from_ckpt=False, load_ids_mode=None, init_ids_from=None, reload_args=None, 
                 min_sigma=None, max_sigma=None, min_off=None, max_off=None, max_wvl_shift=0, overall_mult=False,
                 param_ranges_overall_mult=(0.5, 3), overall_mult_on_toa_sun=False, overall_mult_on_atmo=False,  
                 off_single_param=False, with_diffuse=True, with_diffuse_f=True, with_inp_norm=True, 
                 with_inp_norm_ids=False, max_fwhm_shift=0, global_fwhm_shift=True, with_fwhm_shift_ids=False,
                 emin=1, emax=1, wvl_shift_per_wvl=False, patchwise_wvl_shift=False, patchwise_fwhm_shift=False, 
                 with_spectral_deriv=False, spectral_inp_len=None, spectral_inp_feat_len=None, *args, **kwargs):

        super(_SFMNN, self).__init__()

        self._loaded_ckpt = None

        self.out_wvls = nn.Parameter(out_wvls, requires_grad=False)
        self.at_sensor_wvls = at_sensor_wvls

        self.spectral_inp_len = spectral_inp_len
        self.spectral_inp_feat_len = spectral_inp_feat_len

        self.with_spectral_deriv = with_spectral_deriv
        self.dim_in = dim_in

        unenc_dim_in = self.dim_in

        self.sim_with_aoi = sim_with_aoi

        self.use_rdd = use_rdd
        self.with_diffuse = with_diffuse
        self.with_diffuse_f = with_diffuse_f

        self.stack_enc_inp_for_f_R = stack_enc_inp_for_f_R 

        self.mean_over_angle = mean_over_angle

        self.with_ids = with_ids
        self.ids_mode = ids_mode
        self.id_size = id_size
        self.ids = None

        self.overall_mult = overall_mult
        self.overall_mult_on_toa_sun = overall_mult_on_toa_sun
        self.overall_mult_on_atmo = overall_mult_on_atmo

        n_ids = len(data_source_ids)
        self.data_sources = [os.path.basename(p[0]) for p in data_sources]
        self.data_source_ids = data_source_ids
        self.data_source_id_dict = dict(list(zip(self.data_source_ids, np.arange(len(self.data_source_ids)))))
        
        self.with_inp_norm_ids = with_inp_norm_ids
        self.with_inp_norm = with_inp_norm

        if self.with_inp_norm:
            self.inp_norm = InputNorm(unenc_dim_in, n_ids=None if not self.with_inp_norm_ids else n_ids)

        if self.with_ids:
            self.ids = nn.Parameter(torch.randn((n_ids, id_size)), requires_grad=True)

            if self.ids_mode in self.IDS_MODE_INPUT:
                unenc_dim_in += id_size

        else:
            self.ids = None
            self.data_source_id_dict = None

        self.init_ids_from = init_ids_from
        self.load_ids_mode = load_ids_mode
        
        self.ids_nn = None
        if load_ids_from_ckpt:
            self.load_ids_from_checkpoint(checkpoint_file=reload_args.reload_ckpt,
                                          init_ids_from=self.init_ids_from,
                                          load_ids_mode=self.load_ids_mode,
                                          dim_in=dim_in, dims=dims, **kwargs)

        # # SHARED LAYER ######################################
        self.shared_in_layer = shared_in_layer
        if shared_in_layer:
            if self.with_ids and self.ids_mode in self.IDS_MODE_INPUT:
                dim_in += self.id_size

            self.in_layer = SharedInLayer(dim_in=dim_in, *args, **kwargs)
            bands_in_layer = self.in_layer.out_wvls
            self.dim_in = len(bands_in_layer)

            if self.with_ids and self.ids_mode in self.IDS_MODE_SKIP:
                self.dim_in += self.id_size
        
        # # T14 MODEL
        self.atmosphere_mode = atmosphere_mode
        T14 = T14PCA

        self.t14_model = T14(out_wvls=out_wvls, dims=dims, unenc_dim_in=unenc_dim_in, 
                             dim_in=self.dim_in + self.stack_enc_inp_for_f_R * (unenc_dim_in - self.spectral_inp_len),
                             dim_out=atm_enc_dim, spectral_inp_len=spectral_inp_len, with_ids=self.with_ids,
                             ids_mode=self.ids_mode, data_source_id_dict=self.data_source_id_dict, id_size=self.id_size,
                             enc_dim=len(self.in_layer.out_wvls), **kwargs)

        if not self.t14_model.to_hyplant_ssi:
            self.out_wvls = nn.Parameter(self.t14_model.t14_wvls, requires_grad=False)

        # # F MODEL
        dims_ = copy.deepcopy(dims)
        if f_mode == 'linear':
            self.Fhead = LinearHead(out_wvls=self.out_wvls, slope_min=f_slope_min, slope_max=f_slope_max,
                                    offset_max=f_offset_max, offset_min=f_offset_min)
            kwargs['head'] = self.Fhead
            dims_.append(2)

        elif f_mode == 'gaussian':
            self.Fhead = TwoGaussians(out_wvls=self.out_wvls, only=1, max_amp=max_f, max_sigma=max_sigma, min_sigma=min_sigma,
                                      max_off=max_off, min_off=min_off, fixed_means=max_off is None,
                                      off_single_param=off_single_param)
            kwargs['head'] = self.Fhead
            dims_.append(3)
        else:
            raise NotImplementedError

        self.f = ApplyOnImage(_MLP(dim_in=self.dim_in + self.stack_enc_inp_for_f_R * (unenc_dim_in - self.spectral_inp_len),
                                   dims=dims_, out_wvls=self.out_wvls, *args, **kwargs))

        # # R MODEL
        dims_ = copy.deepcopy(dims)
        if R_mode == 'spline':
            nr_fixed = int((out_wvls[-1] - out_wvls[0]) / 0.9)
            self.Rhead = SplineHead(nr_fixed=nr_fixed, out_wvls=out_wvls)
            dims_.append(nr_fixed)

        elif R_mode == 'linear':
            self.Rhead = LinearHead(out_wvls=self.out_wvls, slope_min=slope_min, slope_max=slope_max,
                                    offset_max=offset_max, offset_min=offset_min)
            dims_.append(2)

        elif R_mode == 'square':
            self.Rhead = RSquareHead(out_wvls=self.out_wvls, slope_min=slope_min, slope_max=slope_max,
                                    offset_max=offset_max, offset_min=offset_min, emin=emin, emax=emax)
            dims_.append(3)

        else:
            raise NotImplementedError

        kwargs['head'] = self.Rhead

        self.R = ApplyOnImage(_MLP(dim_in=self.dim_in + self.stack_enc_inp_for_f_R * (unenc_dim_in - self.spectral_inp_len),
                                   dims=dims_, out_wvls=self.out_wvls, *args, **kwargs))

        self.max_wvl_shift = max_wvl_shift
        self.patchwise_wvl_shift = patchwise_wvl_shift
        if self.max_wvl_shift > 0:
            dims_ = copy.deepcopy(dims)
            kwargs['head'] = None
            kwargs['dim_out'] = 1 if not wvl_shift_per_wvl else len(self.at_sensor_wvls)
            kwargs['out_nonlin'] = 'none'
            kwargs['out_bn'] = True

            shift_estimation = _MLP(dim_in=self.dim_in + self.stack_enc_inp_for_f_R *\
                                                (unenc_dim_in - self.spectral_inp_len),
                                    dims=dims_, out_wvls=self.at_sensor_wvls,
                                    *args, **kwargs)
            self.wvl_shift_estimation = ApplyOnImage(ParamPredictor(shift_estimation,
                                                                    dim_param=[kwargs['dim_out']],
                                                                    param_ranges=dict(
                                                                        shift=(-max_wvl_shift, max_wvl_shift))))

            if self.patchwise_wvl_shift:
                self.wvl_shift_estimation = AttentionReducer(self.wvl_shift_estimation, weight_model='mean')

            #elif self.across_track_wise_wvl_shift:
            #    self.wvl_shift_estimation = AttentionReducer(self.wvl_shift_estimation, weight_model='mean', axis=1)

        self.max_fwhm_shift = max_fwhm_shift
        self.patchwise_fwhm_shift = patchwise_fwhm_shift
        self.global_fwhm_shift = global_fwhm_shift
        self.with_fwhm_shift_ids = with_fwhm_shift_ids

        if self.max_fwhm_shift > 0:
            dl = torch.diff(self.out_wvls)[0]
            
            ms = []
            n = 1 if not self.with_fwhm_shift_ids else n_ids
            for _ in range(n):
                if not self.global_fwhm_shift:
                    dims_ = copy.deepcopy(dims)
                    kwargs['head'] = None
                    kwargs['dim_out'] = 1
                    kwargs['out_nonlin'] = 'none'
                    kwargs['out_bn'] = True

                    shift_estimation = _MLP(dim_in=self.dim_in + self.stack_enc_inp_for_f_R * \
                                                  (unenc_dim_in - self.spectral_inp_len),
                                           dims=dims_, out_wvls=self.out_wvls,
                                           *args, **kwargs)

                else:
                    shift_estimation = Param(1, init=0)

                m = ParamPredictor(shift_estimation, 
                                   param_ranges=dict(shift=(-max_fwhm_shift / dl, max_fwhm_shift / dl)),
                                   with_bn=False)

                if not self.global_fwhm_shift:
                    m = ApplyOnImage(m)

                    if self.patchwise_fwhm_shift:
                        m = AttentionReducer(m, weight_model='mean')
                
                ms.append(m)

            if self.with_fwhm_shift_ids:
                self.fwhm_shift_estimation = nn.ModuleList(ms)

            else:
                self.fwhm_shift_estimation = m

        if self.overall_mult:
            if not self.overall_mult_on_atmo:
                self.overall_mult_model = ParamPredictor(Param(len(self.out_wvls)), param_ranges=param_ranges_overall_mult, 
                                                         dim_param=len(self.out_wvls))
            else:
                self.overall_mult_model = nn.ModuleDict([(name, ParamPredictor(Param(len(self.out_wvls)),
                                                                               param_ranges=param_ranges_overall_mult, 
                                                                               dim_param=len(self.out_wvls)))
                                                         for name in self.t14_model.T14_names])

    def on_train_epoch_start(self):
        super(_SFMNN, self).on_train_epoch_start()
        if self.current_epoch >= self.input_norm_nr_train_epochs:
            m = self.get_module_by_name('model.t14_model.t14_modules.toa_sun')
            params = list(m.named_parameters())

            for _, p in params:
                p.requires_grad_(False)

    def load_ids_from_checkpoint(self, checkpoint_file=None, init_ids_from=None, load_ids_mode=None, **kwargs):
        ckpt = torch.load(checkpoint_file)
        ckpt_ids = ckpt['state_dict']['model.ids']
        
        if load_ids_mode is not None:

            if load_ids_mode == 'weighted_sum':
                self.ids_nn = ApplyOnImage(WeightedSum(vects=ckpt_ids, **kwargs))
                self.ids = None

            else:
                raise NotImplementedError

        else:
            if self.ids is not None and checkpoint_file is not None and init_ids_from is not None:
                assert len(init_ids_from) == self.ids.shape[0] or len(init_ids_from) == 1

            if len(init_ids_from) == 1:
                init_ids_from = torch.ones(self.ids.shape[0]).int() * init_ids_from[0]

            self.ids = nn.Parameter(torch.stack([ckpt_ids[j].clone() 
                                    for i, j in enumerate(init_ids_from)], dim=0).requires_grad_(True))

    def simulate_atmo(self, ypred, aoi=None, sza=None, return_all=False, w_Rmean=True, **kwargs):

        T14, R, sif = self.get_params(ypred, R_detach=True, f_detach=True, T14_detach=False)
        T14 = self.get_T14(T14, ypred)

        toa_sun = self.get_toa_sun(T14, aoi=aoi, sza=sza)

        if w_Rmean:
            R = R.flatten(start_dim=-2).mean(-1)[..., None, None].detach()
        
        _, _, ats = self.simulate_ats(ypred, R_mean=w_Rmean, R_detach=True, f_detach=True, 
                                      T14_detach=False, apply_acv_on_bg=True, aoi=aoi, sza=sza,
                                      **kwargs)

        _, _, toc = self.simulate_toc(ypred, R_mean=w_Rmean, R_detach=True, f_detach=True, 
                                      T14_detach=False, aoi=aoi, sza=sza, 
                                      **kwargs)

        atmo_down = (toc / (toa_sun  * R + 1e-4)) 
        atmo_up = (ats / (toc + 1e-4))

        atmo_all = ats / (toa_sun * R + 1e-4)
        
        if return_all:
            return atmo_all, atmo_up, atmo_down
        
        return atmo_all

    def get_T14(self, T14, ypred):
        T14 = dict([(key, val.clone()) for key, val in T14.items()])

        if self.overall_mult and self.overall_mult_on_toa_sun:
            T14['toa_sun'] = torch.einsum('bkij, bk -> bkij', T14['toa_sun'], ypred['mult'])

        if self.overall_mult and self.overall_mult_on_atmo:
            for key in T14.keys():
                T14[key] = torch.einsum('bkij, bk -> bkij', T14[key], ypred['mult'][key])

        return T14

    def get_toa_sun(self, T14, aoi=None, sza=None, **kwargs):
        if self.sim_with_aoi:
            angle = self.get_angle(aoi=aoi, sza=sza)
            toa_sun = T14['toa_sun'].clone() * torch.cos(angle)

        else:
            toa_sun = T14['toa_sun'].clone()  # * torch.cos(angle)

        return toa_sun

    def simulate_toc(self, ypred, aoi=None, sza=None, R_detach=False, f_detach=False, T14_detach=False,
                     R_mean=False, **kwargs):

        T14, R, sif = self.get_params(ypred, R_detach=R_detach, f_detach=f_detach, T14_detach=T14_detach)
        T14 = self.get_T14(T14, ypred) 

        rso = rdo = R
        rsd = rdd = R.flatten(start_dim=-2).mean(-1)[..., None, None]

        if R_mean:
            rso = rdo = R.flatten(start_dim=-2).mean(-1)[..., None, None]

        toa_sun = self.get_toa_sun(T14, aoi=aoi, sza=sza)

        fs = sif
        fd = fs.flatten(start_dim=-2).mean(-1)[..., None, None]

        bg = toa_sun * (T14['tss'] * rso
                               + (T14['tsd'] + T14['tssrdd'] * rsd) / (1 - rdd * T14['rdd']) * rdo * self.with_diffuse
                               + (T14['tsd'] * rdd) / (1 - rdd * T14['rdd']) * self.with_diffuse) 

        f = fs + fd * T14['rdd'] / (1 - rdd * T14['rdd']) * self.with_diffuse_f

        L = f + bg

        return L, f, bg

    def get_angle(self, aoi=None, sza=None):
        angle = aoi if aoi is not None else sza
        angle = angle / 180 * np.pi

        if self.mean_over_angle:
            angle = angle.mean(dim=tuple(np.arange(2, len(angle.shape))), keepdim=True)

        return angle

    def get_params(self, pred, R_detach=False, f_detach=False, T14_detach=False):
        T14 = dict([(key, val.clone()) for key, val in pred['T14'].items()])
        R = pred['R'].clone()
        sif = pred['f'].clone()

        if R_detach:
            R = R.detach()

        if f_detach:
            sif = sif.detach()

        if T14_detach:
            T14 = dict([(key, val.detach()) for key, val in T14.items()])

        return T14, R, sif

    def simulate_ats(self, ypred, aoi=None, sza=None, R_detach=False, f_detach=False, T14_detach=False,
                     R_mean=False, apply_acv_on_bg=False, **kwargs):

        T14, R, sif = self.get_params(ypred, R_detach=R_detach, f_detach=f_detach, T14_detach=T14_detach)
        T14 = self.get_T14(T14, ypred)
        
        toa_sun = self.get_toa_sun(T14, aoi=aoi, sza=sza)

        rso = rdo = R
        rsd = rdd = R.flatten(start_dim=-2).mean(-1)[..., None, None]

        if R_mean:
            rso = rdo = R.flatten(start_dim=-2).mean(-1)[..., None, None]

        fs = sif
        fd = fs.flatten(start_dim=-2).mean(-1)[..., None, None]

        bg = toa_sun * (T14['rso'] + T14['tsstoo'] * rso
                               + (T14['tsdtoo'] + T14['tssrddtoo'] * rsd) / (1 - rdd * T14['rdd']) * rdo * self.with_diffuse
                               + (T14['tsstdo'] * rsd + T14['tsdtoo'] * rdd) / (1 - rdd * T14['rdd']) * self.with_diffuse)

        f = fs * T14['too'] + fd * (T14['tdo'] + T14['toordd'] * rdo) / (1 - rdd * T14['rdd']) * self.with_diffuse_f

        L = f + bg

        # if we also model across track variation
        if 'acv' in T14.keys():
            L = self.t14_model.acv_model.apply_acv(L, T14['acv'])

            if apply_acv_on_bg:
                bg = self.t14_model.acv_model.apply_acv(bg, T14['acv'])

        if self.overall_mult and not self.overall_mult_on_toa_sun and not self.overall_mult_on_atmo:
            mult = ypred['mult']
            L = torch.einsum('bkij, bk -> bkij', L, mult)

        if self.with_spectral_deriv:
            L = differ(L, axis=1)

        return L, f, bg

    def get_ids(self, x=None, **kwargs):

        window_shape = (x.shape[0], self.id_size, x.shape[2], x.shape[3])

        if self.ids_nn is None:
            ids = torch.stack([self.ids[self.data_source_id_dict[int(p)]]
                               for p in kwargs['source_id']], dim=0)
            ids = ids[..., None, None].expand(window_shape)

        else:
            ids = self.ids_nn(x)

        return ids

    def mask_input(self, inp, **kwargs):
        return zero_nonfinite(inp, **kwargs)

    def forward(self, inp, **kwargs):

        inp, mask, kwargs = self.mask_input(inp, **kwargs)
        
        if self.with_inp_norm:
            inp_normed = self.inp_norm(inp, **kwargs)

        else:
            inp_normed = inp
        
        ids = None
        if self.with_ids:
            ids = self.get_ids(inp_normed, **kwargs)

        if self.with_ids and self.ids_mode in self.IDS_MODE_INPUT:
            inp_normed = torch.cat([inp_normed, ids], dim=1)

        enc = self.in_layer(inp_normed)
        if self.with_ids and self.ids_mode in self.IDS_MODE_SKIP:
            enc = torch.cat([enc, ids], dim=1)

        T14 = self.t14_model((enc, inp_normed), ids=ids, **kwargs)

        if self.stack_enc_inp_for_f_R:
            cat_inp = torch.cat((enc, inp_normed[self.spectral_inp_feat_len:]), dim=1)
            f = self.f(cat_inp)
            R = self.R(cat_inp)

        else:
            f = self.f(enc)
            R = self.R(enc)

        if self.max_wvl_shift > 0:
            if not self.patchwise_wvl_shift:
                wvl_shift = self.wvl_shift_estimation(enc)['shift']
            #shift = torch.atleast_2d(torch.tensor(shift))[..., None, None]\
            #            .expand((shift.shape[0], shift.shape[1], f.shape[2], f.shape[3]))

            else:
                wvl_shift = self.wvl_shift_estimation((enc, None))['shift']
                wvl_shift = wvl_shift[..., None, None].repeat(1, 1, f.shape[2], f.shape[3])


        else:
            wvl_shift = 0

        if self.max_fwhm_shift > 0:
            if not self.global_fwhm_shift:
                if not self.patchwise_fwhm_shift:
                    fwhm_shift = self.fwhm_shift_estimation(enc)['shift']
                
                else:
                    fwhm_shift = self.fwhm_shift_estimation((enc, None))['shift']
                    fwhm_shift = fwhm_shift[..., None, None].repeat(1, 1, f.shape[2], f.shape[3])

            
            else:
                if not self.with_fwhm_shift_ids:
                    fwhm_shift = self.fwhm_shift_estimation()['shift'].squeeze()
                    
                else:
                    source_ids = [int(p) for p in kwargs['source_id']] 
                    fwhm_shift = torch.cat([self.fwhm_shift_estimation[p]()['shift'] for p in source_ids])

                    fwhm_shift = torch.atleast_2d(torch.tensor(fwhm_shift))[..., None, None]\
                                    .expand((fwhm_shift.shape[0], fwhm_shift.shape[1], f.shape[2], f.shape[3]))
        else:
            fwhm_shift = 0 

        if self.overall_mult:
            if self.overall_mult_on_atmo:
                mult = dict([(name, self.overall_mult_model[name](inp_normed)) for name in self.overall_mult_model.keys()])
            else:
                mult = self.overall_mult_model(inp_normed)

        else:
            mult = None

        return dict(pred=dict(T14=T14, f=f, R=R, wvl_shift=wvl_shift, fwhm_shift=fwhm_shift, mult=mult), enc=enc,
                    mask=mask)

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        parser = _MLP.add_model_specific_args(parser, prepend=prepend, **kwargs)
        parser = AttentionReducer.add_model_specific_args(parser, prepend=prepend, **kwargs)
        parser = SharedInLayer.add_model_specific_args(parser, prepend=prepend, **kwargs)
        parser = T14Forward.add_model_specific_args(parser, prepend, **kwargs)
        parser = T14PCA.add_model_specific_args(parser, prepend, **kwargs)

        parser_spec = dict([
            ('shared_in_layer', dict(type=int, default=1)),
            ('stack_enc_inp_for_f_R', dict(type=int, default=0)),
            ('atm_enc_dim', dict(type=int, default=16)),
            ('sim_with_aoi', dict(type=int, default=False)),

            ('f_mode', dict(default='weighted_sum', type=str)),
            ('R_mode', dict(default='weighted_sum', type=str)),
            ('atmosphere_mode', dict(default='default', type=str)),

            ('overall_mult', dict(default=0, type=int)),
            ('param_ranges_overall_mult', dict(default=(0.5, 3), type=float, nargs=2)),
            ('overall_mult_on_toa_sun', dict(default=0, type=int)),
            ('overall_mult_on_atmo', dict(default=0, type=int)),

            ('max_f', dict(default=6, type=float)),
            ('max_sigma', dict(default=None, type=float)),
            ('min_sigma', dict(default=0, type=float)),
            ('max_off', dict(default=None, type=float)),
            ('min_off', dict(default=0, type=float)),
            ('off_single_param', dict(default=0, type=int)),

            ('max_R', dict(default=0.7, type=float)),

            ('min_f', dict(default=0, type=float)),
            ('min_R', dict(default=0, type=float)),

            ('slope_min', dict(default=0, type=float)),
            ('offset_min', dict(default=0, type=float)),
            ('f_slope_min', dict(default=0, type=float)),
            ('f_offset_min', dict(default=0, type=float)),

            ('slope_max', dict(default=1, type=float)),
            ('offset_max', dict(default=1, type=float)),
            ('f_slope_max', dict(default=1, type=float)),
            ('f_offset_max', dict(default=1, type=float)),
            ('emin', dict(default=1, type=float)),
            ('emax', dict(default=1, type=float)),

            ('use_rdd', dict(default=1, type=int)),
            ('with_diffuse', dict(default=1, type=int)),
            ('with_diffuse_f', dict(default=1, type=int)),
            ('fourier_atm', dict(default=0, type=int)),

            ('mean_over_angle', dict(type=int, default=0)),

            ('with_ids', dict(type=int, default=0)),
            ('ids_mode', dict(type=str, default='enc_append')),
            ('id_size', dict(default=32, type=int)),
            ('init_ids_from', dict(default=None, nargs='*', type=int)),
            ('load_ids_from_ckpt', dict(default=False, type=int)),
            ('load_ids_mode', dict(default=None, type=str)),

            ('with_inp_norm_ids', dict(type=int, default=0)),
            ('with_inp_norm', dict(type=int, default=1)),
        ])

        parser = add_model_specific_args(parser, parser_spec=parser_spec, prepend=prepend, **kwargs)
        return parser


class SFMNN(_Module):
    _FITTING_MODEL = _SFMNN

    def __init__(self, in_wvls=None, out_wvls=None, prediction_bands=None, prediction_window=None,
                 with_sif_focus_reg=False,  sif_focus_window=None, sif_focus_mult=10, with_distribution_reg=False,
                 distr_reg_weight=1, with_ndvi_reg=False, ndvi_reg_weight=3, non_sif_focus_window=None, sif_reg_weight=1,
                 with_sif_reg=False, ndvi_threshold=0.15, refl_reg_weight=1, with_refl_reg=False,
                 resample_smoothing_fwhm=0.33, with_snr_weighting=False, with_R_snr_weighting=False, R_focus_mult=1,
                 with_R_consistency_reg=False, with_consistency_reg=False, consistency_reg_weight=1,
                 consistency_reg_with_bg=False, with_f_consistency_reg=False, with_trivial_distr_reg=False, trivial_distr_weight=1,
                 square_weights=False, unphysical_atmo_reg_weight=1, with_unphysical_atmo_reg=False, sif_focus_detach_R=True,
                 with_shrinkage_weights=False, shrinkage_param_a=30, shrinkage_param_c=0.2, meta_vars_exclude=None,
                 with_mi_reduction_reg=False, mi_reduction_reg_weight=1, with_feature_dist_reg=False, feature_dist_weight=1,
                 min_f_consistency=1e-3, consistency_on_toc=False, consistency_under_eval=False, 
                 consistency_randomized=False, consistency_mult_randomized=1.5, refl_reg_clip=0,
                 sif_consistency=True, atmo_consistency=False, with_no_sif_reg=False, no_sif_reg_weight=1,
                 no_sif_reg_threshold=0.1, unphysical_atmo_resample=True, max_wvl_shift=0, max_fwhm_shift=0, normalize_loss=False,
                 with_sif_distr_reg=False, with_unphysical_atmo_reg_2=False, with_ndvi_R_masking=False,
                 ndvi_R_masking_threshold=None, consistency_f_weighted=False,  R_consistency=False, wvl_shift_per_wvl=False,
                 global_fwhm_shift=True, patchwise_wvl_shift=False, patchwise_fwhm_shift=False, 
                 sif_focus_detach_wvl_shift=False, sif_focus_detach_fwhm_shift=False, sif_focus_detach_atmo=False, 
                 geo_consistency=False, sensor_consistency=False, with_loss_threshold=False, loss_threshold=6,
                 with_spectral_deriv=False, with_spectral_deriv_reg=False, spectral_deriv_reg_weight=1,
                 include_spectr=True, with_loss_threshold_relative=False, loss_threshold_relative=1,
                 loss_threshold_epoch=0, **kwargs):

        # pass everything to super class for logging
        all_kwargs = dict([(key, val) for key, val in locals().items() if key != 'self'])
        all_kwargs.update(kwargs)
        super(SFMNN, self).__init__(**all_kwargs)

        self.out_wvls = out_wvls[0]
        self.in_wvls = in_wvls[0]

        self.pred_wvls = in_wvls[0]
        self.pred_window = None

        if prediction_window is not None or prediction_bands is not None:

            if prediction_bands is None:
                self.pred_window = search_spectral_window(*prediction_window, where=in_wvls[0])

            else:
                self.pred_window = [(prediction_bands[i], prediction_bands[i+1]) for i in range(0, len(prediction_bands), 2)]
            self.pred_wvls = select(signals=self.pred_wvls, windows=self.pred_window, axis=-1)

            out_wvls = self.pred_wvls

        self.pred_wvls = nn.Parameter(self.pred_wvls, requires_grad=False).float()

        self.with_distribution_reg = with_distribution_reg
        self.distr_reg_weight = distr_reg_weight
        if self.with_distribution_reg or with_trivial_distr_reg:
            self.soft_histogram = SoftHistogram(bins=100, min=0, max=6, sigma=0.4, eps=1e-4)

        self.with_ndvi_reg = with_ndvi_reg
        self.ndvi_reg_weight = ndvi_reg_weight
        self.ndvi_threshold = ndvi_threshold

        self.with_sif_reg = with_sif_reg
        self.sif_reg_weight = sif_reg_weight
        self.with_sif_distr_reg = with_sif_distr_reg 

        self.with_refl_reg = with_refl_reg
        self.refl_reg_weight = refl_reg_weight
        self.refl_reg_clip = refl_reg_clip

        self.normalize_loss = normalize_loss
        self.with_spectral_deriv = with_spectral_deriv

        self.with_consistency_reg = with_consistency_reg
        self.sif_consistency = sif_consistency
        self.atmo_consistency = atmo_consistency
        self.sensor_consistency = sensor_consistency

        self.R_consistency = R_consistency
        self.geo_consistency = geo_consistency

        self.consistency_reg_weight = consistency_reg_weight
        self.min_f_consistency = min_f_consistency
        self.consistency_on_toc = consistency_on_toc
        self.consistency_under_eval = consistency_under_eval
        self.consistency_randomized = consistency_randomized
        self.consistency_mult_randomized = consistency_mult_randomized
        self.consistency_f_weighted = consistency_f_weighted

        self.with_feature_dist_reg = with_feature_dist_reg
        self.feature_dist_weight = feature_dist_weight

        self.with_trivial_distr_reg= with_trivial_distr_reg
        self.trivial_distr_weight = trivial_distr_weight

        self.with_unphysical_atmo_reg = with_unphysical_atmo_reg
        self.with_unphysical_atmo_reg_2 = with_unphysical_atmo_reg_2
        self.unphysical_atmo_reg_weight = unphysical_atmo_reg_weight
        self.unphysical_atmo_resample = unphysical_atmo_resample

        self.with_no_sif_reg = with_no_sif_reg
        self.no_sif_reg_weight = no_sif_reg_weight
        self.no_sif_reg_threshold = no_sif_reg_threshold

        self.with_shrinkage_weights = with_shrinkage_weights
        self.shrinkage_param_a = shrinkage_param_a
        self.shrinkage_param_c = shrinkage_param_c

        self.with_loss_threshold = with_loss_threshold
        self.loss_threshold = loss_threshold

        self.with_loss_threshold_relative = with_loss_threshold_relative
        self.loss_threshold_relative = loss_threshold_relative

        self.loss_threshold_epoch = loss_threshold_epoch

        self.with_mi_reduction_reg = with_mi_reduction_reg
        self.mi_reduction_reg_weight = mi_reduction_reg_weight

        self.with_spectral_deriv_reg = with_spectral_deriv_reg
        self.spectral_deriv_reg_weight = spectral_deriv_reg_weight 

        self.with_ndvi_R_masking = with_ndvi_R_masking
        self.ndvi_R_masking_threshold = ndvi_R_masking_threshold 

        self.resample_smoothing_fwhm = resample_smoothing_fwhm
        
        self.patchwise_fwhm_shift = patchwise_fwhm_shift
        self.patchwise_wvl_shift = patchwise_wvl_shift

        self.include_spectr = include_spectr
 
        if 'meta_info' not in kwargs or kwargs['meta_info'] is None:
            kwargs['meta_info'] = []
        self.meta_info = kwargs['meta_info']

        if 'manual_features' not in kwargs or kwargs['manual_features'] is None:
            kwargs['manual_features'] = []
        self.manual_features = kwargs['manual_features'] 

        if 'meta_vars' not in kwargs or kwargs['meta_vars'] is None:
            kwargs['meta_vars'] = []

        self.meta_vars = kwargs['meta_vars'] + self.meta_info
        self.input_stack_order = self.manual_features + self.meta_vars

        self.meta_vars_exclude = [] if meta_vars_exclude is None else meta_vars_exclude
        self.meta_vars_exclude += ['reflectance', 'source_id']  # always exclude reflectance from input
        
        self.online_meta_vars = [m for m in self.meta_vars if '_online' in m and not m in self.meta_vars_exclude]
        self.input_stack_order = [m for m in self.input_stack_order if m not in self.online_meta_vars
                                  and m not in self.meta_vars_exclude]
        
        n_manual_feature_bands = self._manual_input_features_dim(len(self.in_wvls))
        n_meta_info_bands = self._meta_info_dim(kwargs)

        self._loaded_ckpt = None

        dim_in = len(self.in_wvls) * self.include_spectr + n_meta_info_bands + n_manual_feature_bands

        kwargs['meta_vars'] = self.meta_vars
        kwargs['meta_vars_exclude'] = self.meta_vars_exclude
              
        self.spectral_inp_len = len(self.in_wvls) * self.include_spectr
        self.spectral_inp_feat_len = len(self.in_wvls) * self.include_spectr + n_manual_feature_bands

        self.model = self._FITTING_MODEL(dim_in=dim_in, in_wvls=in_wvls, out_wvls=self.pred_wvls,
                                         at_sensor_wvls=self.pred_wvls, global_fwhm_shift=global_fwhm_shift,  
                                         spectral_inp_len=self.spectral_inp_feat_len, max_wvl_shift=max_wvl_shift,
                                         wvl_shift_per_wvl=wvl_shift_per_wvl, max_fwhm_shift=max_fwhm_shift,
                                         patchwise_wvl_shift=patchwise_wvl_shift,
                                         patchwise_fwhm_shift=patchwise_fwhm_shift,
                                         with_spectral_deriv=self.with_spectral_deriv,
                                         spectral_inp_feat_len=self.spectral_inp_feat_len, 
                                         device=self.device, **kwargs)

        # if T14 has other sampling, instantiate a resampler for the loss here
        self.do_resample = False
        if len(self.model.out_wvls) != len(self.pred_wvls) and self._FITTING_MODEL == _SFMNN:
            self.do_resample = True

            sigma = self.resample_smoothing_fwhm / 2.35 / torch.diff(self.model.out_wvls)[0]

            if max_fwhm_shift > 0:
                gaussian_smoothing = GaussianSmoothing(channels=1, kernel_size_factor=10,
                                                       sigma=sigma, variable_kernel=True,
                                                       dim=1, keep_dim=False)
            else:
                gaussian_smoothing = GaussianSmoothing(channels=1, kernel_size_factor=10,
                                                       sigma=sigma, dim=1, keep_dim=False)

            kernel_size = gaussian_smoothing.kernel_size[0]

            self.restrained_wvls_in = self.model.out_wvls[kernel_size//2:-kernel_size//2 + 1]
            self.out_pred_wvls_restraining_windows = search_spectral_window(*self.restrained_wvls_in[[0, -1]],
                                                                            where=self.pred_wvls)
            self.restrained_wvls_out = select(self.pred_wvls,
                                              windows=self.out_pred_wvls_restraining_windows)
            
            if type(self.restrained_wvls_out) is np.ndarray:
                self.restrained_wvls_out = torch.from_numpy(self.restrained_wvls_out)

            interpolation = Interpolate(x=self.restrained_wvls_in, xnew=self.restrained_wvls_out)
            self.resampler = ApplyOnImage(MultiSequential(gaussian_smoothing, interpolation))

            if self.with_unphysical_atmo_reg or self.with_unphysical_atmo_reg_2:
                interpolation = Interpolate(x=self.restrained_wvls_in, xnew=self.restrained_wvls_out) ## TODO: fix error due to different func.ind
                self.atmo_resampler = ApplyOnImage(nn.Sequential(gaussian_smoothing, interpolation))

            self.interpolator = ApplyOnImage(Interpolate(x=self.model.out_wvls, xnew=self.restrained_wvls_out))

            self.at_sensor_wvls = nn.Parameter(self.restrained_wvls_out, requires_grad=False)

        else:
            self.at_sensor_wvls = self.pred_wvls

        self.f_window = search_spectral_window(self.out_wvls, where=self.model.out_wvls)

        self.with_sif_focus_reg = with_sif_focus_reg
        self.sif_focus_mult = sif_focus_mult
        self.R_focus_mult = R_focus_mult
        
        self.sif_focus_detach_R = sif_focus_detach_R
        self.sif_focus_detach_wvl_shift = sif_focus_detach_wvl_shift
        self.sif_focus_detach_fwhm_shift = sif_focus_detach_fwhm_shift 
        self.sif_focus_detach_atmo = sif_focus_detach_atmo

        self.sif_focus_window = None
        if sif_focus_window is not None:
            self.sif_focus_window = search_spectral_window(*sif_focus_window,
                                                           where=self.at_sensor_wvls)

            if non_sif_focus_window is None:
                self.non_sif_focus_window = search_spectral_window(*sif_focus_window,
                                                                   where=self.at_sensor_wvls, invert=True)

            else:
                self.non_sif_focus_window = search_spectral_window(*non_sif_focus_window,
                                                                   where=self.at_sensor_wvls)

        # LOAD weights
        self.square_weights = square_weights
        
        SNR_WEIGHTS_DIR = pjoin(os.path.dirname(fluomapper.__file__), 'parameterization', 'snr_weights', 'hyplant')

        self.with_snr_weighting = with_snr_weighting
        if self.with_snr_weighting:
            
            # TODO: do wvl alignment properly, this is aligned to fluomap - 0.22 wvls
            snr_weights_wvls = np.load(pjoin(SNR_WEIGHTS_DIR, 'snr_weights_wvls.npy')) - 0.33
            snr_weights = np.load(pjoin(SNR_WEIGHTS_DIR, 'snr_weights.npy'))
            win = search_spectral_window(*self.at_sensor_wvls[[0, -1]], where=snr_weights_wvls)
            
            snr_weights = np.interp(self.at_sensor_wvls, snr_weights_wvls, snr_weights)
            snr_weights = torch.from_numpy(snr_weights).float()

            if self.square_weights:
                snr_weights = snr_weights ** 2

            self.snr_weights = nn.Parameter(snr_weights, requires_grad=False)
            self.snr_weights /= self.snr_weights.sum()

            self.snr_weights_wvls = self.in_wvls

        self.with_R_snr_weighting = with_R_snr_weighting
        if self.with_R_snr_weighting:
            snr_weights_wvls = np.load(pjoin(SNR_WEIGHTS_DIR, 'R_snr_weights_wvls.npy'))

            self.R_snr_weights = np.load(pjoin(SNR_WEIGHTS_DIR, 'R_snr_weights.npy'))
            self.R_snr_weights = select(signals=self.R_snr_weights, windows=search_spectral_window(*self.at_sensor_wvls[[0, -1]],
                                                                                                   where=snr_weights_wvls),
                                        inclusive=False)

            if self.square_weights:
                self.R_snr_weights = self.R_snr_weights **2

            self.R_snr_weights = nn.Parameter(torch.from_numpy(self.R_snr_weights), requires_grad=False)
            self.R_snr_weights /= self.R_snr_weights.sum()

    @property
    def loaded_checkpoint(self):
        return self._loaded_ckpt

    @loaded_checkpoint.setter
    def loaded_checkpoint(self, value):
        self._loaded_ckpt = value
        if hasattr(self, 'model'):
            self.model._loaded_ckpt = value

    def resample(self, arr, do_smooth=True, wvl_shift=0, fwhm_shift=0, wvl_shift_detach=False, 
                 fwhm_shift_detach=False, **kwargs):

        if wvl_shift is not None and torch.is_tensor(wvl_shift) and wvl_shift.shape[1] > 1:
            wvl_shift = self._cut_y(wvl_shift)

        if wvl_shift is not None and torch.is_tensor(wvl_shift) and wvl_shift_detach:
            wvl_shift = wvl_shift.detach()

        if fwhm_shift is not None and torch.is_tensor(fwhm_shift) and fwhm_shift_detach:
            fwhm_shift = fwhm_shift.detach()

        if self.do_resample:
            if do_smooth:
                arr = self.resampler(arr, wvl_shift=wvl_shift, fwhm_shift=fwhm_shift)
            else:
                arr = self.interpolator(arr, wvl_shift=wvl_shift)

        return arr

    @property
    def dim_out(self):
        return self.model.dim_out

    def _cut_y(self, y):
        if self.do_resample:
            return select(y, windows=self.out_pred_wvls_restraining_windows, axis=1)

        else:
            return y

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

            
    def _rename(self, data):
        return data


    def prepare_batch(self, batch):
        """
        batch: (x, sif) or (x, y, sif)

        """
        if len(batch) == 2:
            x, sif = batch
            y = copy.deepcopy(x)

        elif len(batch) == 1:
            x = batch
            sif = None
            y = copy.deepcopy(x)

        else:
            x, y, sif = batch
        
        #sif = None

        if type(y) is dict:
            y = y['obs'].float()

        if type(x) is dict:
            meta_info = dict([(k, v) for k, v in x.items()
                              if k != 'obs'])
            x = x['obs'].float()

        else:
            meta_info = {}

        if self.with_spectral_deriv:
            x = differ(x, axis=1)
            y = differ(y, axis=1)
       
        # sif = None
        if type(sif) is dict:
            sif_meta_info = dict([(k, v) for k, v in sif.items()
                                  if k != 'obs'])

            sif = sif['obs'].float()

        else:
            sif_meta_info = {}
                
        # add meta vars that are loaded with labels
        # the labels are excluded
        meta_info.update(sif_meta_info)

        # adopt ffs name convention
        meta_info = self._rename(meta_info)

        # add features manually
        meta_info.update(self._add_manual_input_features(x, meta_info))

        # online computation of variables 
        # if 'h1alt_online' in self.online_meta_vars or ('h1alt' in self.meta_vars_order and not 'h1alt' in meta_info):
        #     meta_info['h1alt'] = meta_info['alt'] #- meta_info['h2alt']
        # if 'h2alt_adapted_by_opt_path_online' in self.online_meta_vars:
        #     meta_info['h2alt'] = self.adapt_h2alt_by_opt_path(**meta_info)

        # add meta vars to input
        xs = [x] if self.include_spectr else []
        if meta_info is not None:
            xs += list([meta_info[key] for key in self.input_stack_order])

        xs = torch.cat(xs, 1).float()
        
        if self.pred_window is not None:
            y = select(signals=y, windows=self.pred_window, axis=1)

            if 'reflectance' in meta_info:
                meta_info['reflectance'] = select(signals=meta_info['reflectance'], windows=self.pred_window, axis=1)

            if 'smile_corr' in meta_info:
                meta_info['smile_corr'] = select(signals=meta_info['smile_corr'], windows=self.pred_window, axis=1)

        for key in [key for key in meta_info.keys() if key.endswith('_sensor')]:
            meta_info[key[:-len('_sensor')]] = meta_info[key]
            del meta_info[key]

        # add label to meta vars for access in validation and for metrics
        meta_info.update(sif=sif)
        meta_info.update(spectrum=x)

        return xs, y, meta_info

    def _add_manual_input_features(self, x, meta_info):
        add = dict()

        if 'fft' in self.manual_features:
            add['fft'] = ApplyOnImage.unflatten(torch.log10(torch.fft.rfft(ApplyOnImage.flatten(x))),
                                                batch_dim=x.shape[0], spatial_dim=x.shape[-1])
            
        return add

    def _manual_input_features_dim(self, wvl_dim_len):
        out = 0

        if 'fft' in self.manual_features:
            out += wvl_dim_len // 2 + 1

        return out

    def _meta_info_dim(self, kwargs):
        # add meta_info bands
        added_bands = 0 

        if kwargs['meta_info'] is not None:
            meta_info = kwargs['meta_info']
            accounted_for = 0
            if 't14' in meta_info:
                accounted_for += 1

            if 'off_nadir' in meta_info:
                if kwargs['off_nadir_mode'] is not None and kwargs['off_nadir_mode'] != 'none':
                    added_bands += 2 * int(kwargs['off_nadir_mode'])
                else:
                    added_bands += 1
                
                accounted_for += 1
            
            added_bands += len(meta_info) - accounted_for
        
        # add meta_vars_bands
        added_bands += 0 if kwargs['meta_vars'] is None else len(kwargs['meta_vars'])
        added_bands -= len([var for var in kwargs['meta_vars']
                            if var in self.meta_vars_exclude
                            or var in self.online_meta_vars])

        added_bands -= len([var for var in kwargs['meta_info']
                            if var in self.meta_vars_exclude 
                            or var in self.online_meta_vars])

        return added_bands

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y, other = self.prepare_batch(batch)
        
        for optimizer_idx, opt in enumerate(make_iterable(self.optimizers())):
            # prevent name clashes
            out = self.forward(x, **other)
            loss, _, _, yp, y, mask = self.loss(y=y, inp=x, mode='train', optimizer_idx=optimizer_idx, **out, **other)

            sifpred = self.get_sif(out['pred'])

            # LOGGING ######################################################
            if optimizer_idx == 0:
                self.log('loss', loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

                if 'sif' in other and other['sif'] is not None:
                    mse_sif = F.mse_loss(other['sif'], sifpred)
                    self.log('mse_sif', mse_sif.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

            else:
                self.log('loss_' + self._optimizer_prefixes[optimizer_idx], loss.cpu().detach(),
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # BACKPROP IN MANUAL MODE ######################################################
            if not self.automatic_optimization:
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()

        return loss

    def _mean_reduce(self, arr, mask=None):
        if mask is None:
            return arr.mean()

        else:
            return arr[~mask].mean()

    def validation_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs):
        x, y, other = self.prepare_batch(batch)

        # prevent name clashes 
        out = self.forward(x, **other)
        val_step_loss, _, _, yp, y, mask = self.loss(y=y, inp=x, mode='val', optimizer_idx=optimizer_idx, **out, **other)
        
        if optimizer_idx == 0:
            logs = dict([])
            if not torch.isnan(val_step_loss):
                logs.update({'val_step_loss': val_step_loss})

            val_step_overall_mse = self._mean_reduce((y - yp).mean(1) ** 2, mask=mask)
            logs.update({'val_step_overall_mse': val_step_overall_mse})
            
            if self.with_consistency_reg:
                val_step_consist = self.consistency_loss(x, consistency_on_toc=self.consistency_on_toc,
                                                     consistency_under_eval=self.consistency_under_eval,
                                                     mode='val', pred=out['pred'], **other)
            else:
                val_step_consist = None

            if val_step_consist is not None:
                logs.update({'val_step_consist': val_step_consist})

            if self.sif_focus_window is not None:
                val_step_sif_focus = self._mean_reduce(((select(signals=yp, windows=self.sif_focus_window, axis=1)  - 
                                                       select(signals=y, windows=self.sif_focus_window, axis=1)) ** 2).mean(1), mask=mask)

                logs.update(dict(val_step_sif_focus=val_step_sif_focus))

            if 'sif' in other and other['sif'] is not None:
                sif = other['sif']
                sifpred = self.get_sif(out['pred'])
                
                val_step_f761_mse = self._mean_reduce(((sifpred - sif) ** 2).mean(1), mask=mask)
                logs.update({'val_step_f761_mse': val_step_f761_mse})
               
                val_step_frmse = torch.sqrt(self._mean_reduce((((sif - sifpred) / (sif + 1e-3)) ** 2).mean(1), mask=mask))
                logs.update(dict(val_step_f761_frmse=val_step_frmse))

                if self.val_logger is not None:
                    logs.update(self.val_logger.log_validation_step(sifpred, sif, inp=input))

        self.VAL_LOGS.append(logs)

    def on_validation_epoch_end(self):
        outputs = self.VAL_LOGS

        if self.val_logger is not None:
            _ = self.val_logger.log_validation_epoch_end(outputs=outputs, model=self)
            #self.val_logger.reset()
        
        list_ = [x["val_step_loss"] for x in outputs if 'val_step_loss' in x]
        list_ = [l for l in list_ if not torch.isnan(l)]
        avg_loss = torch.stack(list_).mean() if len(list_) > 0 else torch.tensor(torch.nan)
        self.log('val_loss', avg_loss.item(), sync_dist=True)
        
        list_ = [x["val_step_overall_mse"] for x in outputs if 'val_step_overall_mse' in x]
        list_ = [l for l in list_ if not torch.isnan(l)]
        avg_loss = torch.stack(list_).mean() if len(list_) > 0 else  torch.tensor(torch.nan)
        self.log('overall_mse', avg_loss.item(), sync_dist=True)

        if 'val_step_sif_focus' in outputs[0]:
            avg_loss = torch.stack([x["val_step_sif_focus"] for x in outputs]).mean()
            self.log('sif_focus_mse', avg_loss.item(), sync_dist=True)

        if 'val_step_f761' in outputs[0]:
            avg_loss = torch.stack([x["val_step_f761_mse"] for x in outputs]).mean()
            self.log('mse_f761', avg_loss.item(), sync_dist=True)
        
        if 'val_step_f761_frmse' in outputs[0]:
            avg_loss = torch.stack([x["val_step_f761_frmse"] for x in outputs]).mean()
            self.log('frmse_f761', avg_loss.item(), sync_dist=True)

        if 'val_step_consist' in outputs[0]:
            avg_loss = torch.stack([x["val_step_consist"] for x in outputs]).mean()
            self.log('val_consist', avg_loss.item(), sync_dist=True)

        self.VAL_LOGS.clear()

    def get_sif(self, pred):
        if pred['f'].shape[1] > 1:
            return select(signals=pred['f'].clone(), windows=self.f_window, axis=1)
        else:
            return pred['f'].clone()

    def loss(self, pred, y, inp, sif, enc, mode='train', reflectance=None, ndvi=None, mask=None,
             *args, **kwargs):
        sifpred = self.get_sif(pred)

        y = self._cut_y(y)

        n = 1
        if self.normalize_loss:
            n = y

        # MAIN loss ######################################
        if not self.with_sif_focus_reg:

            y_ats, f_ats, bg_ats = self.model.simulate_ats(pred, **kwargs)
            y_ats = self.resample(y_ats, **pred)
           
            loss = (((y - y_ats) / n) ** 2).mean(dim=1)

        elif self.with_sif_focus_reg:
            # BG PART
            if self.with_R_snr_weighting:
                y_ats, f_ats, bg_ats = self.model.simulate_ats(pred, f_detach=True, R_detach=False, **kwargs)

                y_ats = self.resample(y_ats, wvl_shift_detach=self.sif_focus_detach_wvl_shift, 
                                      fwhm_shift_detach=self.sif_focus_detach_fwhm_shift, **pred)

                focus_R_reg = (torch.einsum('ij..., j -> i...', ((y_ats - y) / n) ** 2, self.R_snr_weights))

                focus_R_relative = torch.abs((y_ats - y) / y).mean(dim=1)

            else:
                y_ats, f_ats, bg_ats = self.model.simulate_ats(pred, f_detach=False, R_detach=False, **kwargs)

                y_ats = self.resample(y_ats, wvl_shift_detach=self.sif_focus_detach_wvl_shift, 
                                      fwhm_shift_detach=self.sif_focus_detach_fwhm_shift, **pred)

                focus_R_reg = ((y - y_ats) / n) ** 2
                focus_R_relative = torch.abs((y_ats - y) / y).mean(dim=1)
                focus_R_reg = focus_R_reg.mean(dim=1)

            # F PART
            y_ats, f_ats, bg_ats = self.model.simulate_ats(pred, f_detach=False, 
                                                           R_detach=self.sif_focus_detach_R, 
                                                           T14_detach=self.sif_focus_detach_atmo,
                                                           **kwargs)
            y_ats = self.resample(y_ats, wvl_shift_detach=self.sif_focus_detach_wvl_shift, 
                                  fwhm_shift_detach=self.sif_focus_detach_fwhm_shift, **pred)
            
            if not self.with_snr_weighting:
                esti = select(signals=y_ats, windows=self.sif_focus_window, axis=1)
                true = select(signals=y, windows=self.sif_focus_window, axis=1)
                
                n_ = 1
                if self.normalize_loss:
                    n_ = true

                focus_reg = (((esti - true) / n_) ** 2).mean(dim=1)

            else:
                focus_reg = (torch.einsum('ij..., j -> i...', ((y_ats - y) / n) ** 2, self.snr_weights))

            if mode == 'train':
                self.log('focus_reg', focus_reg.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            loss = self.sif_focus_mult * focus_reg + self.R_focus_mult * focus_R_reg

        if self.with_consistency_reg:
            const_reg = self.consistency_loss(inp, f_ats=f_ats, pred=pred, consistency_on_toc=self.consistency_on_toc,
                                              consistency_under_eval=self.consistency_under_eval, 
                                              **kwargs)

            if mode == 'train':
                self.log('consistency_reg', const_reg.mean().item(),
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)

            loss += const_reg * self.consistency_reg_weight

        if self.with_feature_dist_reg:
            feature_dist = self.feature_dist_loss(inp=inp, f_ats=f_ats, enc=enc, **kwargs)

            if mode == 'train':
                self.log('feature_dist', feature_dist.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

            loss += feature_dist * self.feature_dist_weight

        if self.with_ndvi_reg:
            #ndvi = NDVI(inp[:, :len(self.in_wvls)], wvls=self.in_wvls, axis=1).unsqueeze(1).expand(sifpred.shape)
            ndvi_reg = sifpred
            ndvi_reg[torch.where(ndvi > self.ndvi_threshold)] = 0 
            ndvi_reg = ndvi_reg.mean(1)
            
            if mode == 'train':
                self.log('ndvi_reg', ndvi_reg.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            loss += ndvi_reg * self.ndvi_reg_weight

        if self.with_refl_reg:
            reflectance = self._cut_y(reflectance)
            R = self.resample(pred['R'].clone(), do_smooth=False)

            reflectance = reflectance.float() / 1000

            invalid_inds = torch.where(torch.all(reflectance <= 0, axis=1))[0]
            refl_reg = ((R - reflectance) ** 2).mean(dim=1)
            refl_reg[invalid_inds] = 0

            if self.refl_reg_clip:
                refl_reg[torch.where(refl_reg < 0.05 ** 2)] = 0

            if mode == 'train':
                self.log('refl_reg', refl_reg.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            loss += refl_reg * self.refl_reg_weight

        if self.with_unphysical_atmo_reg:
            atmo_ = self.model.simulate_atmo(pred, **kwargs)
            if self.do_resample and self.unphysical_atmo_resample:
                atmo_ = self.atmo_resampler(atmo_)

            reg = (torch.nn.functional.relu(atmo_ - 1) ** 2).mean(dim=1)

            if mode == 'train':
                self.log('unphysical_atmo_reg', reg.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

            loss += reg * self.unphysical_atmo_reg_weight

        if self.with_unphysical_atmo_reg_2:
            atmo_, atmo_up, atmo_down = self.model.simulate_atmo(pred, return_all=True, **kwargs)
            if self.do_resample and self.unphysical_atmo_resample:
                atmo_ = self.atmo_resampler(atmo_)
                atmo_up = self.atmo_resampler(atmo_up)
                atmo_down = self.atmo_resampler(atmo_down)
            
            atmo_up = torch.clamp(atmo_up, min=0, max=1.2)
            
            reg = (torch.nn.functional.relu(atmo_ - 1) ** 2).max(dim=1)[0] \
                    + (torch.nn.functional.relu(-atmo_.max(dim=1)[0] + 1) ** 2)
            reg += (torch.nn.functional.relu(atmo_up - 1) ** 2).max(dim=1)[0] \
                        + (torch.nn.functional.relu(-atmo_up.max(dim=1)[0]  + 1) ** 2)
            reg += (torch.nn.functional.relu(atmo_down - 1) ** 2).max(dim=1)[0] \
                    + (torch.nn.functional.relu(-atmo_down.max(dim=1)[0]  + 1) ** 2)

            if mode == 'train':
                self.log('unphysical_atmo_reg', reg.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

            loss += reg * self.unphysical_atmo_reg_weight

        if self.with_shrinkage_weights:
            with torch.no_grad():
                min_, max_ = torch.quantile(loss.flatten(), torch.tensor([0.0, 1.0], device=loss.device))
                delta_l = max_ - min_
                loss_level = min_ + delta_l * self.shrinkage_param_c
                # loss_level = 12
                # delta_l = 1
                
                shrinkage_weights = 1 / (1 + torch.exp(self.shrinkage_param_a / delta_l * (loss_level - loss)))
                shrinkage_weights /= shrinkage_weights.sum()

            weights = shrinkage_weights

        else:
            weights = None

        if self.with_mi_reduction_reg:
            atmo = self.model.simulate_atmo(pred, **kwargs) / torch.cos(self.model.get_angle(aoi=kwargs['aoi']))
            atmo = self.resample(atmo)

            out_shape = (atmo.shape[0], sifpred.shape[-2], sifpred.shape[-1])
            atmo_min = atmo.min(dim=1)[0].expand(out_shape).flatten()
            aoi = kwargs['aoi'].flatten().float()
            
            xy = torch.stack([atmo_min, aoi], dim=0)
            xy = (xy - xy.min(dim=-1)[0].unsqueeze(-1)) / (xy.max(dim=-1)[0].unsqueeze(-1) - xy.min(dim=-1)[0].unsqueeze(-1))
            pxy = histogram2d(xy[0].unsqueeze(0), xy[1].unsqueeze(0), 
                    torch.linspace(0, 1, 100, device=loss.device), bandwidth=torch.tensor(0.1))
            
            #pxy = torch.einsum('i..., i -> i...', pxy, 1 / pxy.sum(dim=(1, 2)))
            #mi_reg = - conditional_entropy(pxy.squeeze(), given=1)

            #mi_reg = mutual_information(pxy)
            mi_reg = - conditional_entropy(pxy.squeeze(), given=1)

            if mode == 'train':
                self.log('mi_reg', mi_reg.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            loss += mi_reg * self.mi_reduction_reg_weight 

        if self.with_sif_reg:
            loss += ((sifpred - sif) ** 2 * self.sif_reg_weight).mean(dim=1)

        if self.with_sif_distr_reg:
            #predp = torch.histc(sifpred.flatten() / (torch.max(sifpred) + 1e-8)) + 1e-8
            #p = torch.histc(sif.flatten() / (torch.max(sif) + 1e-8)) + 1e-8
            #predp = torch.histc(sifpred.flatten(), min=0.1, max=4) + 1e-8
            #p = torch.histc(sif.flatten(), min=0.1, max=4) + 1e-8

            #p /= p.sum()
            #predp /= predp.sum()

            #loss +=  jensen_shannon_diff(predp, p) * self.sif_reg_weight
            where = sif > 0.1
            loss += (sifpred[where].mean() - sif[where].mean())**2 * self.sif_reg_weight

        if self.with_no_sif_reg:
            mask_ = sif > self.no_sif_reg_threshold
            no_sif_reg = ((sifpred - sif) ** 2)

            # only allow loss in no_sif pixels
            no_sif_reg[mask_] = 0
            no_sif_reg = no_sif_reg.mean(dim=1)
            
            if mode == 'train':
                self.log('no_sif_reg', no_sif_reg.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            loss += no_sif_reg * self.no_sif_reg_weight

        if self.with_spectral_deriv_reg:
            reg = ((differ(y) - differ(y_ats)) ** 2).mean(dim=1)
            loss += reg * self.spectral_deriv_reg_weight

        if self.with_loss_threshold and self.loss_threshold_epoch <= self.current_epoch:
            nan_mask = torch.where(torch.logical_and(focus_R_reg > self.loss_threshold,
                                                    ndvi.squeeze() < self.ndvi_R_masking_threshold[0]))
            loss[nan_mask] = torch.nan

        if self.with_loss_threshold_relative and self.loss_threshold_epoch <= self.current_epoch:
            nan_mask = torch.where(torch.logical_and(focus_R_relative > self.loss_threshold_relative,
                                                     ndvi.squeeze() < self.ndvi_R_masking_threshold[0]))
            loss[nan_mask] = torch.nan

        mask = self.mask_loss(pred, loss=loss, premask=mask, **kwargs)

        if self.with_ndvi_R_masking:
            R = pred['R'][:, 0]
            mask = torch.logical_and(ndvi.squeeze() < self.ndvi_R_masking_threshold[0],
                                     R > self.ndvi_R_masking_threshold[1])

        if weights is not None:
            loss = loss * weights

        if mask is not None:
            loss = loss[~mask]

        loss = loss.mean()

        return loss, f_ats, bg_ats, y_ats, y, mask

    def mask_loss(self, *args, **kwargs):
        return None
   
    def feature_dist_loss(self, inp, enc, f_ats, **kwargs):
        f_ats = self.resample(f_ats.clone())

        lo, hi = search_spectral_window(*self.at_sensor_wvls[[0, -1]], where=self.in_wvls)[0]
        hi += 1
        window = slice(lo, hi)

        inp_copy = inp.clone()
        inp_copy[:, window] = inp[:, window] - f_ats.clone().detach()

        out2 = self.forward(inp_copy, **kwargs)

        dot_prod = torch.einsum('ij..., ij... -> i...', enc.clone(), out2['enc'].clone())
        norm_a = torch.einsum('ij..., ij... -> i...', enc.clone(), enc.clone()) + 1e-8
        norm_b = torch.einsum('ij..., ij... -> i...', out2['enc'].clone(), out2['enc'].clone()) + 1e-8
        feature_dist = dot_prod / norm_a / norm_b

        return feature_dist

    def consistency_loss(self, inp, f_ats=None, pred=None, consistency_under_eval=True,
                         consistency_on_toc=False,  mode='train', **kwargs):

        if consistency_under_eval or pred is None or f_ats is None:
            self.model.eval()

            pred = self.forward(inp, **kwargs)['pred']
            f_toc = pred['f'].clone()
            _, f_ats, _ = self.model.simulate_ats(pred, f_detach=False, R_detach=False, **kwargs)

            self.model.train()
            self.set_inp_norm_eval()

        else:
            f_toc = pred['f'].clone()  # the f_ats from above can be used

        if f_ats is None:
            return None

        f_ats = self.resample(f_ats.clone())
        lo, hi = search_spectral_window(*self.at_sensor_wvls[[0, -1]], where=self.in_wvls)[0]
        hi += 1
        window = slice(lo, hi)

        f_ats_ = f_ats.detach()
        f_toc_ = f_toc.detach()

        if self.consistency_randomized:
            shape_ = tuple(list(f_ats.shape[:1]) + list(f_ats.shape[2:]))
            frac_ = torch.rand(np.prod(shape_), device=f_ats.device) * self.consistency_mult_randomized
            frac_ = frac_.reshape(shape_)

            f_ats_new = torch.einsum('ik..., i... -> ik...', f_ats_, frac_)
            f_toc_new = torch.einsum('ik..., i... -> ik...', f_toc_, frac_)

        else:
            f_ats_new = 0

        inp_copy = inp.clone()
        inp_copy[:, window] = inp[:, window] - f_ats_ + f_ats_new
      
        if consistency_under_eval:
            self.model.eval()
            pred2 = self.forward(inp_copy, **kwargs)['pred']

            self.model.train()
            self.set_inp_norm_eval()

        else:
            pred2 = self.forward(inp_copy, **kwargs)['pred']

        const_reg = 0

        if self.sif_consistency:
            if consistency_on_toc:
                f1_toc = self.resample(f_toc_)
                f2_toc = self.resample(pred2['f'])  # clone
                f_toc_new = self.resample(f_toc_new)

                const_reg += torch.abs(f2_toc - f_toc_new).mean(dim=1)

            else:
                f1_ats = f_ats
                f_ats_new = f_ats_new
                f2_ats = self.resample(
                    self.model.simulate_ats(pred2, f_detach=False, R_detach=self.sif_focus_detach_R, **kwargs)[1])

                const_reg += torch.abs(f2_ats - f_ats_new).mean(dim=1)

        if self.atmo_consistency:
            atmo1 = pred['T14']
            atmo2 = pred2['T14']

            for key in atmo1.keys():
                atmo1_key = atmo1[key].clone().detach() 
                const_reg += torch.abs(atmo1_key - atmo2[key]).mean(dim=1)

        if self.R_consistency:
            R1 = pred['R']
            R2 = pred2['R']

            const_reg += torch.abs(R2 - R1).mean(dim=1)

        if self.consistency_f_weighted:
            const_reg *= f2_toc_.mean(1) / 3

        if mode == 'val':
            self.model.eval()

        return const_reg

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        parser = super(SFMNN, cls).add_model_specific_args(parser, prepend=prepend, **kwargs)
        parser = cls._FITTING_MODEL.add_model_specific_args(parser, prepend=prepend, **kwargs)
        
        parser_spec = dict([
            ('prediction_window', dict(default=None, type=float, nargs='+')),
            ('prediction_bands', dict(default=None, type=int, nargs='+')),
            ('max_wvl_shift', dict(default=0, type=float)),
            ('wvl_shift_per_wvl', dict(default=0, type=float)),
            ('max_fwhm_shift', dict(default=0, type=float)),
            ('global_fwhm_shift', dict(default=1, type=float)),
            ('with_fwhm_shift_ids', dict(default=1, type=float)),
    
            ('patchwise_wvl_shift', dict(default=0, type=int)),
            ('patchwise_fwhm_shift', dict(default=0, type=int)),
    
            ('with_sif_focus_reg', dict(default=0, type=int)),
            ('sif_focus_window', dict(type=float, nargs='*', default=None)),
            ('non_sif_focus_window', dict(type=float, nargs='*', default=None)),
            ('sif_focus_mult', dict(type=float, default=1)),
            
            ('sif_focus_detach_R', dict(type=int, default=1)),
            ('sif_focus_detach_wvl_shift', dict(type=int, default=0)),
            ('sif_focus_detach_fwhm_shift', dict(type=int, default=0)),
            ('sif_focus_detach_atmo', dict(type=int, default=0)),
    
            ('R_focus_mult', dict(type=float, default=1)),
    
            ('normalize_loss', dict(default=0, type=int)),
    
            ('with_distribution_reg', dict(default=0, type=int)),
            ('distr_reg_weight', dict(type=float, default=1)),
    
            ('with_sif_reg', dict(default=0, type=int)),
            ('sif_reg_weight', dict(default=1, type=float)),
            ('with_sif_distr_reg', dict(default=0, type=int)),
    
            ('with_no_sif_reg', dict(default=0, type=int)),
            ('no_sif_reg_weight', dict(default=1, type=float)),
            ('no_sif_reg_threshold', dict(default=0.1, type=float)),
    
            ('with_ndvi_reg', dict(default=0, type=int)),
            ('ndvi_reg_weight', dict(type=float, default=1)),
            ('ndvi_threshold', dict(type=float, default=0.15)),
    
            ('with_refl_reg', dict(default=0, type=int)),
            ('refl_reg_weight', dict(type=float, default=1)),
            ('refl_reg_clip', dict(type=int, default=0)),
    
            ('with_consistency_reg', dict(default=0, type=int)),
            ('sif_consistency', dict(default=1, type=int)),
            ('atmo_consistency', dict(default=0, type=int)),
            ('geo_consistency', dict(default=0, type=int)),
            ('sensor_consistency', dict(default=0, type=int)),

            ('R_consistency', dict(default=0, type=int)),
            ('consistency_reg_weight', dict(type=float, default=1)),
            ('min_f_consistency', dict(type=float, default=1e-3)),
            ('consistency_on_toc', dict(type=int, default=0)),
            ('consistency_randomized', dict(type=int, default=0)),
            ('consistency_mult_randomized', dict(type=float, default=1.5)),
            ('consistency_under_eval', dict(type=int, default=0)),
            ('consistency_f_weighted', dict(type=int, default=0)),
    
            ('with_feature_dist_reg', dict(default=0, type=int)),
            ('feature_dist_weight', dict(default=0, type=float)),
    
            ('with_trivial_distr_reg', dict(default=0, type=int)),
            ('trivial_distr_weight', dict(default=1.0, type=float)),
    
            ('with_unphysical_atmo_reg', dict(default=0, type=int)),
            ('with_unphysical_atmo_reg_2', dict(default=0, type=int)),
            ('unphysical_atmo_reg_weight', dict(default=1.0, type=float)),
            ('unphysical_atmo_resample', dict(default=1, type=int)),
    
            ('resample_smoothing_fwhm', dict(default=0.33, type=float)),
    
            ('with_snr_weighting', dict(default=0, type=int)),
            ('with_R_snr_weighting', dict(default=0, type=int)),
            ('square_weights', dict(default=0, type=int)),
    
            ('with_shrinkage_weights', dict(default=0, type=int)),
            ('shrinkage_param_a', dict(default=30, type=float)),
            ('shrinkage_param_c', dict(default=0.2, type=float)),
    
            ('with_mi_reduction_reg', dict(default=0, type=int)),
            ('mi_reduction_reg_weight', dict(type=float, default=1)),
    
            ('meta_vars_exclude', dict(nargs='+', type=str, default=None)),
    
            ('with_ndvi_R_masking', dict(default=0, type=int)),
            ('ndvi_R_masking_threshold', dict(default=(0.1, 0.5), type=float, nargs=2)),

            ('with_loss_threshold', dict(default=0, type=int)),
            ('loss_threshold', dict(default=6, type=float)),

            ('with_loss_threshold_relative', dict(default=0, type=int)),
            ('loss_threshold_relative', dict(default=1, type=float)),

            ('loss_threshold_epoch',  dict(default=0, type=int)),

            ('with_spectral_deriv', dict(default=False, type=int)),
            ('with_spectral_deriv_reg', dict(default=False, type=int)), 
            ('spectral_deriv_reg_weight', dict(default=1, type=float)),

            ('manual_features', dict(default=None, type=str, nargs='*')),
            ('include_spectr', dict(default=True, type=int))

        ])
        
        parser = add_model_specific_args(parser, parser_spec=parser_spec, prepend=prepend, **kwargs)
        return parser
