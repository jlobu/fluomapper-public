import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F

import numpy as np
import copy, os

from os.path import join as pjoin

import h5py
import pickle as pkl
import itertools as it

import fluomapper
from fluomapper.utils.func.PolyFeatures import PolynomialFeaturesTorch
from fluomapper.utils.nn import InputNorm, ApplyOnImage, ScaledSigmoid, differ
from fluomapper.utils.data import select, search_spectral_window, to_ffs_names, logical_or, zero_nonfinite
from fluomapper.utils.run import _remove, _add, add_model_specific_args, add_prepend, split_kwargs

from fluomapper.nn.SFMNN.SFMNN import SFMNN
from fluomapper.nn.SFMNN.SFMNN_helpers import SharedInLayer

from fluomapper.nn._base import simulation_interface
from fluomapper.nn.simulation_mlp.mlp import _MLP
from fluomapper.nn.SFMNN.SFMNN_helpers import ApplyOnImage, Param, ParamPredictor





class FFS(nn.Module):
    def __init__(self, ffs_model_file, warn_extrapolated=False, normalize=True, unit_scale=10, out_wvls=None, *args, **kwargs):
        """
        :param ffsir: fast forward simulator input ranges
        :param ffsaux: fast forward simulator auxiliary data
        """
        super(FFS, self).__init__()

        self.name = os.path.basename(ffs_model_file)
        if 'DESIS' in self.name or 'HyPlant' in self.name:
            if not os.path.isabs(ffs_model_file):
                ffs_model_file = pjoin(os.path.dirname(fluomapper.__file__),
                                       'parameterization/poly_fluomap', ffs_model_file)
            ffs = simulation_interface.load(ffs_model_file)

        elif 'FLEX' in self.name:
            with open(ffs_model_file) as f:
                ffs = pkl.load(f)

        sklearn_model, self.ffsir, self.ffsaux = ffs
        self.input_dim = len(self.ffsir)
        self.ffsir_dict = dict(self.ffsir)

        self.fwhm_key = 'FWHM shift [nm]'
        self.cw_key = 'CW shift [nm]'

        self.fwhm_index = list(self.ffsir_dict.keys()).index(self.fwhm_key)
        self.cw_index = list(self.ffsir_dict.keys()).index(self.cw_key)

        # self.out_wvls = nn.Parameter(torch.tensor(out_wvls).squeeze(), requires_grad=False)

        # PolynomialFeatures
        self.poly_degree = sklearn_model['poly'].degree
        self.poly_features = PolynomialFeaturesTorch(self.poly_degree)

        self.coefs = nn.Parameter(torch.from_numpy(sklearn_model['ridge'].coef_), requires_grad=False)
        self.intercept = nn.Parameter(torch.from_numpy(sklearn_model['ridge'].intercept_), requires_grad=False)
        
        self.dim = self.intercept.shape[0]

        self.warn_extrapolated = warn_extrapolated
        self.normalize = normalize
        self.unit_scale = unit_scale

        wvls_file = pjoin(os.path.dirname(fluomapper.__file__), 'parameterization/poly_fluomap',
                          ffs_model_file[:-4] + '_wvls.txt')
        with open(wvls_file, "r") as f:
            wvls = [line.strip() for line in f if line.strip()]

        self.out_wvls = torch.from_numpy(np.asarray(wvls).astype(float)).requires_grad_(False).float()
        self.out_wvls = nn.Parameter(self.out_wvls, requires_grad=False)

    def forward(self, x, wvl_ind=None, return_feats=False, **kwargs):
        do_flatten = False
        if len(x.shape) == 4:
            do_flatten = True

            keep_dims = x.shape[0], x.shape[-1]
            x = ApplyOnImage.flatten(x)

        feats = self.get_features(x, **kwargs)

        if wvl_ind is None:
            x = torch.einsum('jk, ik -> ij', self.coefs, feats) + self.intercept

        else:
            x = (torch.einsum('k, ik -> i', self.coefs[wvl_ind], feats) + self.intercept[wvl_ind])[:, None]

        x = x * self.unit_scale

        if do_flatten:
            x = ApplyOnImage.unflatten(x, *keep_dims)

        if return_feats:
            return x, feats

        return x

    def get_features(self, x, **kwargs):
        if self.normalize:
            x = self.normalize_inputs(x, **kwargs)

        x = self.get_poly_features(x)

        return x

    def normalize_inputs(self, x, extrapolate=False, **kwargs):
        """Normalise input points to database ranges.
        :param x: input points [#samples]x[#input_dim] (numpy array)

        :returns: normalised input points [#samples]x[#input_dim] (numpy array)
        """

        # Check dimensions of input points
        if len(self.ffsir) != x.shape[1]:
            raise InputException('Wrong input dimensions: ', x.shape)

        # Normalise input points to [0,1]
        hh = dict()
        ipn = torch.zeros_like(x)
        for ii, ir in enumerate(self.ffsir):
            var = ir[0]
            m, M = ir[1]
            ipn[:, ii] = (x[:, ii] - m) / (M - m)

            # Check outliers
            if self.warn_extrapolated and \
                    torch.sum(ipn[:, ii] < 0) or torch.sum(ipn[:, ii] > 1) and not extrapolate:
                raise InputException('Warning: There are values outside the range of ' + \
                                     var + ': ', m, ' - ', M, '. The output of the fast ' + \
                                     'forward simulator for those values is extrapolated and ' + \
                                     'should not be trusted.')

            # Save h_gnd, h_sen
            if var in ['h_gnd [km]', 'h_sen [km]']:
                hh[var] = x[:, ii]

        # Note that for HyPlant DB3-4, the valid range of sensor altitudes is
        # defined wrt the ground.
        if self.ffsaux['sensor'] == 'HyPlant' and self.ffsaux['db'] in ['DB3', 'DB4']:
            hsengnd = hh['h_sen [km]'] - hh['h_gnd [km]']

            if self.ffsaux['db_extra'] == '675':
                m, M = 0.659, 0.691
            elif self.ffsaux['db_extra'] == '1559':
                m, M = 1.543, 1.598

            if self.warn_extrapolated and \
                    (hsengnd < m or hsengnd > M) and not extrapolate:
                raise InputException('Warning: There are values outside the range of sensor ' + \
                                     'altitude wrt ground:', m, ' - ', M, '. The output of the ' + \
                                     'fast forward simulator for those values is extrapolated ' + \
                                     'and should not be trusted.')

        return ipn

    def to_physical_scale(self, param, x, extrapolate=False):
        if param in self.ffsir_dict:
            m, M = self.ffsir_dict[param]
            out = x * (M - m) + m

            if not extrapolate:
                return torch.clamp(out, min=m, max=M)

            return out

        else:
            return None

    def to_fwd_scale(self, param, x, extrapolate=False):
        if param in self.ffsir_dict:
            m, M = self.ffsir_dict[param]
            out = (x - m) / (M - m)

            if not extrapolate:
                return torch.clamp(out, min=0, max=1)

            return out

        else:
            return None

    def get_poly_features(self, x):
        if not self.poly_features.is_fitted():
            self.poly_features.fit(x)
            if not self.poly_features.n_output_features_ == self.coefs.shape[1]:
                raise Exception(f'Loaded coefs shape {self.coefs.shape} does not match'
                                f'the Linear Model implemented in FFS. You probably have '
                                f'chosen the wrong number of input parameters ({x.shape[1]} -> {self.poly_features.n_output_features_} features).')

        return self.poly_features.transform(x)


class FFS_corrector(nn.Module):
    def __init__(self, filepath, exclude_incoming):
        super(FFS_corrector, self).__init__()
        
        with open(filepath, 'rb') as f:
            model = pkl.load(f)

        if not 'type' in os.path.basename(filepath):
            self.type = 'poly'

        else:
            splits = os.path.basename(filepath).split('__')
            idx = [True if 'type' in split else False for split in splits].index(True)
            self.type = splits[idx].split('_')[1]

        if self.type =='poly':
            self.ccoefs, self.cintercept, self.cnorms = [nn.Parameter(torch.from_numpy(p).float(), 
                                                                      requires_grad=False)
                                                         for p in model[:3]]

            self.poly_deg, self.deg, self.use_external_poly, self.with_large_feat_vect, self.with_interaction = model[3:8]
            self.poly = None
            if not self.use_external_poly:
                self.poly = PolynomialFeaturesTorch(degree=self.poly_deg)

            self.n_wvls = self.ccoefs.shape[0]
            self.n_features = self.feats_dim = self.ccoefs.shape[1]

            self.exclude_incoming = exclude_incoming

            self.is_fitted = False

        elif self.type == 'MLP':
            model, poly_deg, deg, use_external_poly, with_large_feat_vect, with_interaction = model

            self.ccoefs, self.mean, self.std = [], [], []
            for m in model:
                    coefs, mean, std = m
                    
                    if len(self.ccoefs) == 0:
                            for _ in coefs:
                                    self.ccoefs.append([])
                                    
                    for i, c in enumerate(coefs):
                            if len(c.shape) == 1:
                                    c = c.unsqueeze(-1)
                                    
                            self.ccoefs[i].append(c)
                    
                    self.mean.append(mean)
                    self.std.append(std)
                    
            self.ccoefs = torch.nn.ParameterList([torch.Parameter(
                                        torch.from_numpy(np.stack(self.ccoefs[i])), 
                                    requires_grad=False) 
                                        for i in len(self.ccoefs)])
                                                                               
            self.mean = torch.Parameter(torch.from_numpy(np.asarray(self.mean)), 
                                                                    requires_grad=False)
                                                                    
            self.std = torch.Parameter(torch.from_numpy(np.asarray(self.std)), 
                                                               requires_grad=False)


    def forward(self, *args, **kwargs):
        if self.type == 'poly':
            return self.forward_poly(*args, **kwargs)

        elif self.type == 'MLP':
            #return self.MLP
            raise NotImplementedError
        

    def forward_poly(self, x, fwhms, cws, poly_feats=None, inds=None):
        feats = self.transform(x, fwhms, cws, poly_feats=poly_feats, inds=inds)

        if inds is None:
            return (torch.einsum('jk, ijk -> ij', self.ccoefs, feats) + self.cintercept) \
                    * self.cnorms[:, 1][None] + self.cnorms[:, 0][None]

        else:
            return (torch.einsum('jk, ijk -> ij', self.ccoefs[inds], feats) + self.cintercept[inds]) \
                   * self.cnorms[inds, 1][None] + self.cnorms[inds, 0][None]
    
    def update(self, feats, vec, c=0):
        c_ = c + vec.shape[-1]
        feats[..., c:c_] = vec
        return c, c_

    def get(self, feats, from_, to_=None):
        return feats[..., from_:to_].clone()

    def fit(self, x):
        if self.poly is not None and not self.poly.is_fitted():
            self.poly.fit(x)

    def transform(self, x, fwhms, cws, poly_feats=None, inds=None):
        n_wvls = self.n_wvls 

        if inds is not None:
            fwhms = fwhms[:, inds]
            cws = cws[:, inds]
            n_wvls = fwhms.shape[-1]

        x = self.remove_excluded(x)

        if not self.is_fitted:
           self.fit(x)

        if poly_feats is None or not self.use_external_poly:
            x = self.poly.transform(x)

        else:
            x = poly_feats

        feats, c = torch.empty(x.shape[0], n_wvls, self.n_features, device=x.device), 0

        c, c_ = self.update(feats, x[:, None].expand(-1, n_wvls, -1), c)
        
        c0, c0_ = self.update(feats, torch.einsum('ik, ij ->ijk', x, cws), c_)
        c1, c1_ = self.update(feats, torch.einsum('ik, ij ->ijk', x, fwhms), c0_)

        if self.deg == 1 and self.with_interaction:
            c2, c2_ = self.update(feats, torch.einsum('ik, ij ->ijk', x, fwhms * cws), c1_)

        else:
            c2, c2_ = c1, c1_
        
        i, i_ = c1, c2_
        for r in range(1, self.deg):
            # p, x, 
            # x c , x f, (1)
            # x c2, x f c, x f2 (2)
            # x c3, x f c2, x f2 c, x f3 (3)
            
            #only_cw = torch.einsum('ij, ij ->ij', update[..., 0], cws).unsqueeze(-1)
            c0, c0_ = self.update(feats, 
                                  torch.einsum('ijk, ij ->ijk', 
                                                 self.get(feats, c0, c0_), cws), 
                                  c2_)
            
            #only_fwhm = torch.einsum('ij, ij ->ij', update[..., -1], fwhms).unsqueeze(-1)
            c1, c1_ = self.update(feats, 
                                  torch.einsum('ijk, ij ->ijk', 
                                                 self.get(feats, c1, c1_), cws), 
                                  c0_)

            
            if self.with_interaction:
                #interaction = torch.einsum('ijk, ij ->ijk', update[..., 1:], cws)
                #update = torch.cat([only_cw, interaction, only_fwhm], axis=-1)
                c2, c2_ = self.update(feats, 
                                      torch.einsum('ijk, ij ->ijk', 
                                                   self.get(feats, i, i_), cws), 
                                      c1_)
            i, i_ = c1, c2_
            
        return feats

    def remove_excluded(self, x):
        if self.exclude_incoming is not None:
            incoming_inds = sorted(list(set(np.arange(x.shape[-1])) - set(self.exclude_incoming)))
            return x[:, incoming_inds]

        return x


class FFS_sensitivity_corrector(nn.Module):
    def __init__(self, filepath):
        super(FFS_sensitivity_corrector, self).__init__()

        if not os.path.isabs(filepath):
            filepath = pjoin(os.path.dirname(fluomapper.__file__),
                                   'parameterization/poly_fluomap', filepath)

        with open(filepath, 'rb') as f:
            model = pkl.load(f)

        self.A, self.b = [nn.Parameter(torch.from_numpy(p).float(),
                                       requires_grad=False)
                          for p in model[2:4]]

        self.pbands = model[1]

        self.poly_deg = model[0]

        self.poly = PolynomialFeaturesTorch(self.poly_deg)
        self.poly.fit(torch.tensor([[0, 0]]))

    def forward(self, x, fwhms, cws, inds=None):
        if inds is None:
            inds = range(fwhms.shape[1])
        
        inp = torch.stack([fwhms, cws], axis=-1)
        inp = inp[:, inds]
        n_samples, n_wvls, _ = inp.shape

        inp = inp.flatten(end_dim=1)
        poly = self.poly.transform(inp)
        poly = poly.reshape(n_samples, n_wvls, -1)

        return torch.einsum('ijk, jk -> ij', poly, self.A[inds]) + self.b[inds]


class FFS_var_lambda(FFS):
    def __init__(self, *args, corrector_file=None, sensitivity_file=None, **kwargs):
        super(FFS_var_lambda, self).__init__(*args, **kwargs)
        
        self.powers = None

        self.with_corrector = False
        self.with_sensitivity = False

        self.ffs_corrector = None
        self.ffs_sensitivity_corrector = None

        if corrector_file is not None:
            self.with_corrector = True
            self.ffs_corrector = FFS_corrector(corrector_file, 
                                               exclude_incoming=[self.fwhm_index, self.cw_index])
            
            if self.ffs_corrector.use_external_poly:
                # assert fwd.poly has the required dimensionality
                assert f'PRR-{self.ffs_corrector.poly_deg}-0' in self.name

        if sensitivity_file is not None:
            self.with_sensitivity = True

            self.ffs_sensitivity_corrector = FFS_sensitivity_corrector(sensitivity_file)
            self.pbands = self.ffs_sensitivity_corrector.pbands

        self.fwd_zeros = torch.tensor([self.to_fwd_scale(self.fwhm_key, torch.tensor(0)),
                                       self.to_fwd_scale(self.cw_key, torch.tensor(0))])
        self.fwd_zeros = nn.Parameter(self.fwd_zeros.float(), requires_grad=False)


    def set_up_powers(self):
        if self.powers is None:
            self.powers = nn.Parameter(torch.from_numpy(self.poly_features.get_powers()).float(),
                                       requires_grad=False)
            self.cw_feature_update_inds = list(np.where(self.powers[:, self.cw_index])[0])
            self.fwhm_feature_update_inds = list(np.where(self.powers[:, self.fwhm_index])[0])
            self.fwhm_cw_feature_update_inds = list(set(self.cw_feature_update_inds
                                                        + self.fwhm_feature_update_inds))

    def forward(self, *args, **kwargs):
        if not self.with_corrector and not self.with_sensitivity:
            return self.forward_naive(*args, **kwargs)

        elif self.with_corrector and not self.with_sensitivity:
            return self.forward_w_corrector(*args, **kwargs)

        elif not self.with_corrector and self.with_sensitivity:
            return self.forward_w_corrector(*args, **kwargs, only_w_sensitivity=True)

        else:
            return self.forward_w_corrector(*args, **kwargs, pbands=self.pbands)

    def forward_w_corrector(self, x, fwhms=None, cws=None, pbands=None, only_w_sensitivity=False,
                            *args, **kwargs):
        if fwhms is None and cws is None:
            return super(FFS_var_lambda, self).forward(x, *args, **kwargs)

        else:
            if fwhms.shape != cws.shape:
                dim = max(fwhms.shape[1], cws.shape[1])
                fwhms = fwhms.expand(-1, dim, -1, -1) if fwhms.shape[1] == 1 else fwhms
                cws = cws.expand(-1, dim, -1, -1) if cws.shape[1] == 1 else cws

            do_flatten, keep_dims, x, fwhms, cws = self.flatten(x, fwhms, cws)
            
            if self.normalize:
                xn = self.normalize_inputs(x, **kwargs)
                fwhmsn = self.to_fwd_scale(self.fwhm_key, fwhms)
                cwsn = self.to_fwd_scale(self.cw_key, cws)
                    
                # change with values s.t. after normalization fwhm and cw are set 0 as was
                # used during training of ffs_corrector
                x[:, [self.fwhm_index, self.cw_index]] = 0


            else:
                xn = x
                fwhmsn = fwhms
                cwsn = cws

                x[:, [self.fwhm_index, self.cw_index]] = self.fwd_zeros

            y, feats = super(FFS_var_lambda, self).forward(x, *args, return_feats=True, **kwargs)
            
            if self.ffs_corrector is not None and self.ffs_corrector.use_external_poly:
                self.set_up_powers()
                feats = feats[:, self.fwhm_cw_feature_update_inds]

            else:
                feats = None

            if pbands is None and not only_w_sensitivity:
                corr = self.ffs_corrector(xn, fwhms=fwhmsn, cws=cwsn, poly_feats=feats)
                x = y + corr

            elif pbands is not None and not only_w_sensitivity:
                res = torch.empty(x.shape[0], fwhms.shape[1], device=x.device)

                sensitivity_inds = list(set(np.arange(self.coefs.shape[0])) - set(pbands))
                corr = self.ffs_corrector(xn, fwhms=fwhmsn, cws=cwsn, poly_feats=feats, inds=pbands)
                res[:, pbands] = y[:, pbands] + corr
                
                s = self.ffs_sensitivity_corrector(xn, fwhms=fwhmsn, cws=cwsn, inds=sensitivity_inds)
                res[:, sensitivity_inds] = y[:, sensitivity_inds] * s

                x = res

            else:
                s = self.ffs_sensitivity_corrector(xn, fwhms=fwhmsn, cws=cwsn, inds=None)
                x = y * s

            if do_flatten:
                x = ApplyOnImage.unflatten(x, *keep_dims)

            return x

    def forward_naive(self, x, fwhms=None, cws=None, *args, **kwargs):

        if fwhms is None and cws is None:
            return super(FFS_var_lambda, self).forward(x, *args, **kwargs)

        else:
            res_per_wvl = []
            for wvl_ind in range(self.dim):
                if fwhms is not None:
                    x[:, self.fwhm_index] = fwhms[:, wvl_ind]

                if cws is not None:
                    x[:, self.cw_index] = cws[:, wvl_ind]

                res_per_wvl.append(super(FFS_var_lambda, self).forward(x, *args, wvl_ind=wvl_ind, **kwargs))

            return torch.cat([res_per_wvl[wvl_ind]
                              for wvl_ind in range(self.dim)], axis=1)

    def prep_band_wise_features(self, x, var, var_columns):
        n_reps = var.shape[-1]
        x = x.repeat(n_reps, 1)

        for i in range(var.shape[1]):
            v = var[:, i].flatten()
            x[..., var_columns[i]] = v

        return x

    def flatten(self, x, fwhms, cws):
        do_flatten = len(x.shape) == 4
        keep_dims = x.shape[0], x.shape[-1]

        if do_flatten:
            x = ApplyOnImage.flatten(x)
            fwhms = ApplyOnImage.flatten(fwhms) if fwhms is not None else None
            cws = ApplyOnImage.flatten(cws) if cws is not None else None

        return do_flatten, keep_dims, x, fwhms, cws

    def forward_w_wvl_to_batch(self, x, fwhms=None, cws=None, *args, **kwargs):

        if fwhms is None and cws is None:
            return super(FFS_var_lambda, self).forward(x, *args, **kwargs)

        do_flatten, keep_dims, x, fwhms, cws = self.flatten(x, fwhms, cws)

        if fwhms is not None and cws is not None:
            var = torch.stack([fwhms, cws], axis=1)
            var_columns = [self.fwhm_index, self.cw_index]

        elif fwhms is not None:
            var = fwhms
            var_columns = [self.fwhm_index]

        elif fwhms is not None and cws is not None:
            var = cws
            var_columns = [self.cw_index]

        n_samples = x.shape[0]
        n_reps = var.shape[-1]
        x = self.prep_band_wise_features(x, var, var_columns)

        feats = self.get_features(x, **kwargs)
        feats = feats.reshape(n_samples, n_reps, feats.shape[-1])
        x = torch.einsum('jk, ijk -> ij', self.coefs, feats) + self.intercept

        x = x * self.unit_scale

        if do_flatten:
            x = ApplyOnImage.unflatten(x, *keep_dims)

        return x

    def forward_w_feature_update(self, x, fwhms=None, cws=None, **kwargs):
        do_flatten, keep_dims, x, fwhms, cws = self.flatten(x, fwhms, cws)

        feats = self.get_features(x, **kwargs)
        self.set_up_powers()

        if not (fwhms is None and cws is None):
            update_inds = self.fwhm_cw_feature_update_inds \
                if fwhms is not None and cws is not None \
                else self.cw_feature_update_inds if cws is not None \
                else self.fwhm_feature_update_inds

            update_inds = np.asarray(update_inds)

            # update features for different wavelengths
            res_per_wvl = []
            for wvl_ind in range(self.dim):
                if fwhms is not None:
                    x[:, self.fwhm_index] = fwhms[:, wvl_ind]

                if cws is not None:
                    x[:, self.cw_index] = cws[:, wvl_ind]

                if self.normalize:
                    xn = self.normalize_inputs(x, **kwargs)

                else:
                    xn = x

                feats[:, update_inds] = self.calc_poly_features(xn, update_inds)
                res_per_wvl.append(torch.einsum('k, ik', self.coefs[wvl_ind], feats))

            x = torch.stack(res_per_wvl, dim=1) + self.intercept[None]

        else:
            x = torch.einsum('jk, ik -> ij', self.coefs, feats) + self.intercept

        x = x * self.unit_scale

        if do_flatten:
            x = ApplyOnImage.unflatten(x, *keep_dims)

        return x

    def calc_poly_features(self, x, feature_inds):
        powers = self.powers[feature_inds].transpose(1, 0)
        #return torch.prod(x[..., None] ** powers[None], dim=1)
        return torch.prod(torch.pow(x[..., None], powers[None]), dim=1)


class _FWDNN(nn.Module):
    FWD_INP_PARAM_NAME_DICT = dict([('TA [deg]', 'tilt'),
                                    ('SZA [deg]', 'parm2'),
                                    ('RAA [deg]', 'parm1'),
                                    ('h_gnd [km]', 'h2alt'),
                                    ('h_sen [km]', 'h1alt'),
                                    ('g []', 'g'),
                                    ('-AOT 550 nm []', 'neg_AOT'),
                                    ('H2O [cm]', 'h2ostr'),
                                    ('CW shift [nm]', 'CW'),
                                    ('FWHM shift [nm]', 'fwhm'),
                                    ('rho(740 nm) []', 'rho',),
                                    ('drho/dl [1/nm]', 'rho_slope'),
                                    ('e', 'e'),
                                    ('LF(737 nm) [mW/m2/sr/nm]', 'f'),
                                    ])

    FWD_INP_PARAM_NAME_DICT_INV = {v: k for k, v in FWD_INP_PARAM_NAME_DICT.items()}

    PX_VARS = ['rho', 'rho_slope', 'f', 'a', 'b', 'e', 'tilt', 'bckzen',  # FLUOMAP

               'APAR_Chl', 'APAR_Car', 'LCC', 'LAI', 'FVC', 'FAPAR', 'photons_dissipated_as_NPQ',  # FLEX
               'PRI', 'Electron_transport_rate', 'F_quantum_yield', 'sif_corrected_abs'  # FLEX
               ]

    ACROSS_TRACK_VARS = ['CW', 'fwhm']

    ALONG_TRACK_VARS = ['h1alt']
    
    SOURCE_VARS = []

    IDS_MODE_TO_DEC = ('to_dec', 'to_enc_dec')
    IDS_MODE_TO_ENC = ('to_enc', 'to_enc_dec')

    def __init__(self, dim_in, in_wvls, out_wvls, fwd_model_file, fwd_corrector_model_file=None, 
                 fwd_sensitivity_corrector_model_file=None, model_type='mlp', meta_vars=None, meta_vars_exclude=False,
                 pass_meta_vars_to_fwd=True, pass_meta_vars_to_dec=False, spectral_fwdnn_bands=None,
                 with_ids=False, ids_mode=None, id_size=None, data_sources=None, data_source_ids=None,
                 pass_meta_vars_to_fwd_select=None, physical_emulator_transform=False,
                 allowed_fwd_range=None, fixed_rho_in_ef=False, only_slope=False, patchwise_wvl_shift=False,
                 patchwise_fwhm_shift=False, fix_fwhm=False, fixed_fwhm=0, global_fwhm_shift=False,
                 with_xtrack_vars=False, allowed_xtrack_fwd_range=None, xtrack_id_reduce_to_n=None, 
                 xtrack_vars_on_off_nadir=False, with_atrack_vars=False, atrack_vars_on_dist=False,
                 xtrack_vars_on_source_id=True, geo_sourcewise=False, fwhm_sourcewise=False,
                 atrack_vars_on_source_id=True, pxwise_fwhm_shift=False, pxwise_wvl_shift=False, extrapolate_fwd=False, constant_fwhm_shift=True,
                 constant_cw_shift=True, non_constant_sensor_mode='free', non_constant_cw_mode=None, non_constant_fwhm_mode=None, 
                 xtrack_vars_source_id_trainable=False, atrack_vars_source_id_trainable=False, non_constant_sensor_mode_range=1,
                 pass_idparam_to_enc=False, load_ids_from_ckpt=True, pass_meta_vars_to_xtrack_on_off_nadir=False, 
                 pass_dist_to_xtrack_vars_on_off_nadir=False, mask_dark_px=False, with_spectral_deriv=False,
                 spectral_inp_len=None, spectral_inp_feat_len=None, xtrack_off_nadir_additive_embedding=False,
                 xtrack_off_nadir_concat_embedding=False, xtrack_off_nadir_embedding_size=8, device=None, *args, **kwargs):

        super(_FWDNN, self).__init__()

        self._loaded_ckpt = None
        self.device = device

        if constant_cw_shift and constant_fwhm_shift:
            fwd_cls = FFS
        else:
            fwd_cls = FFS_var_lambda

        if fwd_corrector_model_file == 'none':
            fwd_corrector_model_file = None

        if fwd_sensitivity_corrector_model_file == 'none':
            fwd_sensitivity_corrector_model_file = None

        self.fwd = fwd_cls(fwd_model_file, corrector_file=fwd_corrector_model_file, 
                           sensitivity_file=fwd_sensitivity_corrector_model_file, 
                           normalize=False, out_wvls=in_wvls)

        if allowed_fwd_range is None:
            allowed_fwd_range = (0, 1)

        self.allowed_fwd_range = allowed_fwd_range

        self.constant_cw_shift = constant_cw_shift
        self.constant_fwhm_shift = constant_fwhm_shift

        self.with_spectral_deriv = with_spectral_deriv

        self.sensor_modes = dict(CW=non_constant_cw_mode \
                        if non_constant_cw_mode is not None else non_constant_sensor_mode,
                                 fwhm=non_constant_fwhm_mode \
                        if non_constant_fwhm_mode is not None else non_constant_sensor_mode) 
    
        self.sensor_dims  = dict()
        for key, mode in self.sensor_modes.items():
            if self.constant_cw_shift:
                dim = 1

            elif mode == 'free':
                dim = self.fwd.dim

            elif mode == 'linear':
                dim = 2

            elif mode == 'square':
                dim = 3

            elif mode.isdigit():
                dim = int(mode) + 1

            else:
                raise NotImplementedError(f'Mode {mode} is not known.')

            self.sensor_dims[key] = dim 

        self.non_constant_sensor_mode_range = non_constant_sensor_mode_range

        self.out_wvls = self.fwd.out_wvls

        self.with_ids = with_ids
        self.ids_mode = ids_mode
        self.id_size = id_size
        self.ids_nn = None

        need_ids = self.with_ids or (xtrack_vars_on_source_id and xtrack_vars_source_id_trainable)
        if need_ids:
            data_source_ids = list(set(data_source_ids))
            n_ids = len(data_source_ids)

            self.data_sources = [os.path.basename(p[0]) for p in data_sources]
            self.data_source_id_dict = dict(list(zip(data_source_ids, 
                                                     np.arange(len(data_source_ids)))))
            self.ids = nn.Parameter(torch.randn((n_ids, id_size)), requires_grad=True)
            
        else:
            self.ids = None
            self.data_source_id_dict = None
        
        # for loading purposes save source_ids
        data_source_ids = torch.tensor(list(set(data_source_ids))).long()
        self.data_source_ids = nn.Parameter(torch.tensor(data_source_ids, dtype=torch.long),
                                            requires_grad=False)

        self.in_wvls = in_wvls
        self.band680nm, self.band780nm = search_spectral_window(680, 780, where=self.in_wvls)[0]
        self.mask_dark_px = mask_dark_px

        self.spectral_fwdnn_bands = spectral_fwdnn_bands
        if spectral_fwdnn_bands is not None:
            self.spectral_fwdnn_bands = [(spectral_fwdnn_bands[i], spectral_fwdnn_bands[i + 1]) for i in
                                         range(0, len(spectral_fwdnn_bands), 2)]

            self.out_wvls = nn.Parameter(torch.from_numpy(select(self.out_wvls, windows=self.spectral_fwdnn_bands)), requires_grad=False)

        self.meta_vars = to_ffs_names(meta_vars)
        self.meta_vars_exclude = to_ffs_names(meta_vars_exclude)

        self.meta_vars = [param for param in self.meta_vars if param not in self.meta_vars_exclude]

        self.fwd_inputs = [self.FWD_INP_PARAM_NAME_DICT.get(param, param) for param, _ in self.fwd.ffsir]
        self.dim_out = len(self.fwd_inputs)

        self.global_fwhm_shift = global_fwhm_shift
        self.fix_fwhm = fix_fwhm if not global_fwhm_shift else global_fwhm_shift
        self.fixed_fwhm = fixed_fwhm

        self.pass_meta_vars_to_fwd = pass_meta_vars_to_fwd if pass_meta_vars_to_fwd_select is None else True
        self.pass_idparam_to_enc = pass_idparam_to_enc

        if pass_meta_vars_to_fwd_select is not None:
            self.meta_vars_to_fwd = [param for param in self.meta_vars if param in pass_meta_vars_to_fwd_select]

        elif self.pass_meta_vars_to_fwd:
            self.meta_vars_to_fwd = self.meta_vars

        else:
            self.meta_vars_to_fwd = None

        if self.pass_meta_vars_to_fwd:
            self.predicted_fwd_inputs = [param for param in self.fwd_inputs if param not in self.meta_vars_to_fwd]

        else:
            self.predicted_fwd_inputs = copy.copy(self.fwd_inputs)

        self.pass_meta_vars_to_dec = pass_meta_vars_to_dec
        self.spectral_inp_len = spectral_inp_len
        self.spectral_inp_feat_len = spectral_inp_feat_len

        fwd_dim_in = len(self.predicted_fwd_inputs)

        self.with_xtrack_vars = with_xtrack_vars
        self.xtrack_vars_on_off_nadir = xtrack_vars_on_off_nadir
        self.xtrack_vars_on_source_id = xtrack_vars_on_source_id
        self.xtrack_vars_source_id_trainable = xtrack_vars_source_id_trainable
        self.xtrack_id_reduce_to_n = xtrack_id_reduce_to_n

        self.xtrack_off_nadir_additive_embedding = xtrack_off_nadir_additive_embedding
        self.xtrack_off_nadir_concat_embedding = xtrack_off_nadir_concat_embedding
        self.xtrack_off_nadir_embedding_size = xtrack_off_nadir_embedding_size
        assert not (self.xtrack_off_nadir_concat_embedding and self.xtrack_off_nadir_additive_embedding)

        self.pass_meta_vars_to_xtrack_on_off_nadir = pass_meta_vars_to_xtrack_on_off_nadir  
        self.pass_dist_to_xtrack_vars_on_off_nadir = pass_dist_to_xtrack_vars_on_off_nadir

        self.with_atrack_vars = with_atrack_vars
        self.atrack_vars_on_dist = atrack_vars_on_dist
        self.atrack_vars_on_source_id = atrack_vars_on_source_id
        self.atrack_vars_source_id_trainable = atrack_vars_source_id_trainable

        self.with_source_vars = geo_sourcewise

        is_px_var = lambda var: var in self.PX_VARS or np.any([p.startswith(var) for p in self.PX_VARS])
        is_xtrack_var = lambda var: var in self.ACROSS_TRACK_VARS if self.with_xtrack_vars else False
        is_atrack_var = lambda var: var in self.ALONG_TRACK_VARS if self.with_atrack_vars else False
        is_source_var = lambda var: var in self.SOURCE_VARS if self.with_source_vars else False

        self.patch_fwd_inputs = [var for var in self.predicted_fwd_inputs if
                                 not is_px_var(var) and not is_xtrack_var(var) and not is_atrack_var(var) 
                                 and not is_source_var(var)]
        self.px_fwd_inputs = [var for var in self.predicted_fwd_inputs if is_px_var(var)]
        self.xtrack_fwd_inputs = [var for var in self.predicted_fwd_inputs if is_xtrack_var(var)]
        self.atrack_fwd_inputs = [var for var in self.predicted_fwd_inputs if is_atrack_var(var)]
        self.source_fwd_inputs = [var for var in self.predicted_fwd_inputs if is_source_var(var)]
        
        # define dimensionalities of variables, in particular CW and fwhm are subject
        # to change depending on the non_constant_mode
        self.param_dim_out = dict(CW=self.sensor_dims['CW'] if not self.constant_cw_shift else 1,
                                  fwhm=self.sensor_dims['fwhm'] if not self.constant_fwhm_shift else 1)
        
        # if it's a list of strings, assume they refer to the variables else assume it's a boolean for all variables
        
        if type(extrapolate_fwd) in (list, tuple) and len(extrapolate_fwd) == 1 and extrapolate_fwd[0].isdigit():
            extrapolate_fwd = int(extrapolate_fwd[0])

        self.extrapolate_fwd = extrapolate_fwd
       
        if self.fix_fwhm and 'fwhm':
            _remove(self.xtrack_fwd_inputs, 'fwhm')

        if patchwise_fwhm_shift:
            _add(self.patch_fwd_inputs, 'fwhm')
            _remove(self.xtrack_fwd_inputs, 'fwhm')

        # if pxwise_fwhm_shift:
        #     _add(self.px_fwd_inputs, 'fwhm')
        #     _remove(self.xtrack_fwd_inputs, 'fwhm')

        if pxwise_wvl_shift:
            _remove(self.xtrack_fwd_inputs, 'CW')
            _add(self.px_fwd_inputs, 'CW')

        if pxwise_fwhm_shift:
            _remove(self.xtrack_fwd_inputs, 'fwhm')
            _add(self.px_fwd_inputs, 'fwhm')

        if patchwise_wvl_shift:
            _add(self.patch_fwd_inputs, 'CW')
            _remove(self.xtrack_fwd_inputs, 'CW')

        if geo_sourcewise:
            _add(self.source_fwd_inputs, 'parm1')
            _add(self.source_fwd_inputs, 'parm2')

            _remove(self.patch_fwd_inputs, 'parm1')
            _remove(self.patch_fwd_inputs, 'parm2')

        if fwhm_sourcewise:
            _add(self.source_fwd_inputs, 'fwhm')
            _remove(self.xtrack_fwd_inputs, 'fwhm')

        idparam_dim = self.sensor_dims['CW'] * ('CW' in self.predicted_fwd_inputs) * \
                                    (~np.any(['CW' in v for v in (self.px_fwd_inputs, self.patch_fwd_inputs)])) \
                             + self.sensor_dims['fwhm'] * ('fwhm' in self.predicted_fwd_inputs)  * \
                                    (~np.any(['fwhm' in v for v in (self.px_fwd_inputs, self.patch_fwd_inputs)])) \
                             + 1 * ('h1alt' in self.predicted_fwd_inputs) 
       
        self.model_type = model_type
        if model_type == 'mlp':
            kwargs['out_nonlin'] = None

            self.inp_norm = InputNorm(dim_in, windows=False)
            self.model = nn.Sequential(_MLP(dim_in=dim_in, dim_out=fwd_dim_in, *args, **kwargs),
                                       ScaledSigmoid(*allowed_fwd_range))

            if kwargs['out_mode'] == 'windows':
                self.model = ApplyOnImage(self.model)

        elif model_type == 'enc':
            # kwargs['shared.out_nonlin'] = nn
            self.inp_norm = InputNorm(dim_in)
            self.dim_in = dim_in + self.with_ids * (self.ids_mode in self.IDS_MODE_TO_ENC) * self.id_size
            self.enc = SharedInLayer(dim_in=self.dim_in, *args, **kwargs)

            enc_dim_out = len(self.enc.out_wvls) + self.pass_meta_vars_to_dec * len(self.meta_vars) \
                          + self.with_ids * (self.ids_mode in self.IDS_MODE_TO_DEC) * self.id_size
            
            self.enc_dim_out = enc_dim_out
            self.enc_dim_out_orig = len(self.enc.out_wvls) 
            self.dim_in_dec_px_patch = enc_dim_out + idparam_dim * self.pass_idparam_to_enc

            # rho f drho_dl
            kwargs['out_nonlin'] = None

            dim_out = len(self.px_fwd_inputs)
            if not self.constant_cw_shift and 'CW' in self.px_fwd_inputs:
                dim_out += self.sensor_dims['CW'] - 1

            if not self.constant_fwhm_shift and 'fwhm' in self.px_fwd_inputs:
                dim_out += self.sensor_dims['fwhm'] - 1
            
            self.dec = ApplyOnImage(nn.Sequential(_MLP(dim_in=self.dim_in_dec_px_patch, dim_out=dim_out,
                                                       *args, **kwargs),
                                                  ScaledSigmoid(*allowed_fwd_range)))

            # atmo and geo vars
            if len(self.patch_fwd_inputs) > 0:

                dim_out = len(self.patch_fwd_inputs)
                if not self.constant_cw_shift and 'CW' in self.patch_fwd_inputs:
                    dim_out += self.sensor_dims['CW'] - 1

                if not self.constant_fwhm_shift and 'fwhm' in self.patch_fwd_inputs:
                    dim_out += self.sensor_dims['fwhm'] - 1
                
                self.dec_atm_geo = nn.Sequential(_MLP(dim_in=self.dim_in_dec_px_patch, dim_out=dim_out,
                                                      *args, **kwargs),
                                                 ScaledSigmoid(*allowed_fwd_range))

            else:
                self.dec_atm_geo = None

            # xtrack vars
            if self.with_xtrack_vars and len(self.xtrack_fwd_inputs) > 0:

                dim_out = len(self.xtrack_fwd_inputs)
                if not self.constant_cw_shift and 'CW' in self.xtrack_fwd_inputs:
                    dim_out += self.sensor_dims['CW'] - 1

                if not self.constant_fwhm_shift and 'fwhm' in self.xtrack_fwd_inputs:
                    dim_out += self.sensor_dims['fwhm'] - 1
                
                dim_in = enc_dim_out if not self.xtrack_vars_on_off_nadir \
                            else len(self.meta_vars) if self.pass_meta_vars_to_xtrack_on_off_nadir \
                            else 0

                if self.xtrack_off_nadir_concat_embedding:
                    id_size = self.id_size + self.xtrack_off_nadir_embedding_size

                elif not self.xtrack_off_nadir_additive_embedding:
                    id_size = self.id_size + 1

                else:
                    id_size = self.id_size

                dim_in += id_size if self.xtrack_vars_on_source_id and self.xtrack_vars_source_id_trainable \
                    else self.xtrack_vars_on_source_id

                dim_in += self.pass_dist_to_xtrack_vars_on_off_nadir
                
                _kwargs = copy.deepcopy(kwargs)
                _kwargs['dropout'] = 0 if self.xtrack_vars_on_off_nadir else _kwargs['dropout']
                
                fwd_range = allowed_fwd_range if allowed_xtrack_fwd_range is None else allowed_xtrack_fwd_range
                self.allowed_xtrack_fwd_range = fwd_range
                self.dec_xtrack = nn.Sequential(_MLP(dim_in=dim_in, dim_out=dim_out,
                                                     *args, **_kwargs),
                                                ScaledSigmoid(*fwd_range))

                if self.xtrack_vars_on_off_nadir:
                    self.dec_xtrack = ApplyOnImage(self.dec_xtrack)

            else:
                self.dec_xtrack = None

            if self.with_atrack_vars and len(self.atrack_fwd_inputs) > 0:

                dim_out = len(self.atrack_fwd_inputs)
                if not self.constant_cw_shift and 'CW' in self.atrack_fwd_inputs:
                    dim_out += self.sensor_dims['CW'] - 1

                if not self.constant_fwhm_shift and 'fwhm' in self.atrack_fwd_inputs:
                    dim_out += self.sensor_dims['fwhm'] - 1

                dim_in = enc_dim_out if not self.atrack_vars_on_dist else 1
                dim_in += self.id_size if self.atrack_vars_on_source_id and self.atrack_vars_source_id_trainable \
                    else self.atrack_vars_on_source_id

                _kwargs = copy.deepcopy(kwargs)
                _kwargs['dropout'] = 0 if self.atrack_vars_on_dist else _kwargs['dropout']

                self.dec_atrack = nn.Sequential(_MLP(dim_in=dim_in, dim_out=dim_out,
                                                     *args, **_kwargs),
                                                ScaledSigmoid(*allowed_fwd_range))

                if self.atrack_vars_on_dist:
                    self.dec_atrack = ApplyOnImage(self.dec_atrack)

            else:
                self.dec_atrack = None

            if len(self.source_fwd_inputs) > 0:
                dim_in = self.id_size
                dim_out = len(self.source_fwd_inputs)

                _kwargs = split_kwargs(kwargs, 'dec_source', update_dic=True)
                _kwargs['dropout'] = 0 
                self.dec_source = nn.Sequential(_MLP(dim_in=dim_in, dim_out=dim_out,
                                                     *args, **_kwargs),
                                                ScaledSigmoid(*allowed_fwd_range))

                self.dec_source = ApplyOnImage(self.dec_source)
            
            else:
                self.dec_source = None

            if self.global_fwhm_shift:
                shift_estimation = Param(1, init=0)
                m = ParamPredictor(shift_estimation,
                                   param_ranges=dict(fwhm=allowed_fwd_range),
                                   with_bn=False)
                self.fwhm_estimator = ApplyOnImage(m)

    def on_load_checkpoint(self, checkpoint, device):
        if self.load_ids_from_checkpoint:
            self.ids, self.data_source_id_dict = \
                        self.load_ids_from_checkpoint(data_source_id_dict=self.data_source_id_dict, 
                                                      ids=self.ids, device=device, checkpoint=checkpoint)

    def load_ids_from_checkpoint(self, checkpoint, data_source_id_dict=None, ids=None, device=None):
        with torch.no_grad():
            if not type(checkpoint) is dict:
                ckpt = torch.load(checkpoint, map_location=self.device if device is None else device)

            else:
                ckpt = checkpoint

            loaded_ids = torch.nn.Parameter(ckpt['state_dict'][f'model.ids'].to(self.device), requires_grad=True)

            loaded_data_source_id_dict = dict(
                [(ckpt['state_dict'][f'model.data_source_ids'][i].item(), i) for i in range(ids.shape[0])])

            # if need to include passed data_source_ids
            if ids is not None and data_source_id_dict is not None:
                assert ids.shape[1] == loaded_ids.shape[1]

                d_ids = list(data_source_id_dict.keys())
                loaded_d_ids = loaded_data_source_id_dict.keys()
                override_ids = set(d_ids).intersection(set(list(loaded_d_ids)))

                for k in override_ids:
                    ids[data_source_id_dict[k]][:] = loaded_ids[loaded_data_source_id_dict[k]]

            else:
                data_source_id_dict = loaded_data_source_id_dict
                ids = loaded_ids
                
            return ids, data_source_id_dict

    def get_ids(self, x=None, reduce_to_first_n=None, **kwargs):
        # TODO: this is the same function as in _SFMNN, define this in superclass
        window_shape = (x.shape[0], self.id_size, x.shape[2], x.shape[3])
        
        if reduce_to_first_n is not None:
            map_ = dict([(v, int(str(v)[:reduce_to_first_n])) for v in self.data_source_id_dict.keys()])
            keys = np.unique(list(map_.values()))
            ids = torch.stack([torch.stack([self.ids[self.data_source_id_dict[v]] 
                                            for v in self.data_source_id_dict.keys() if map_[v] == k]).mean(axis=0)
                               for k in keys])

            data_source_id_dict = dict([(v, list(keys).index(map_[v])) for i, v in enumerate(self.data_source_id_dict.keys())])

        else:
            ids = self.ids
            data_source_id_dict = self.data_source_id_dict

        if self.ids_nn is None:
            ids = torch.stack([ids[data_source_id_dict[int(p)]]
                               for p in kwargs['source_id']], dim=0)
            ids = ids[..., None, None].expand(window_shape)

        else:
            ids = self.ids_nn(x)

        return ids

    def simulate_atmo(self):
        return None

    def stack_params_for_fwd(self, params, exclude=None):
        # if self.transformer is not None:
        #    params['rho'] = 0.5 * torch.ones(params['rho'].shape, device=params['rho'].device)
        #    params['rho_slope'] = params['rho']

        if exclude is None:
            exclude = []

        prms = []
        for name in self.fwd_inputs:
            if name in exclude:
                continue

            if name == 'CW' and not self.constant_cw_shift:
                prms.append(params['CW'].mean(1, keepdim=True))

            elif name == 'fwhm' and not self.constant_fwhm_shift:
                prms.append(params['fwhm'].mean(1, keepdim=True))

            else:
                prms.append(params[name])
        
        return torch.cat(prms, dim=1)

    def simulate_ats(self, pred, R_detach=False, f_detach=False, T14_detach=False, cut_to_fwdnn_bands=True, **kwargs):
        params = self.to_fwd_scale(pred)

        if R_detach:
            params['rho'] = params['rho'].detach()

        if f_detach:
            params['f'] = params['f'].detach()

        if T14_detach:
            params['h2ostr'] = params['h2ostr'].detach()
            params['neg_AOT'] = params['neg_AOT'].detach()

        params_ = self.stack_params_for_fwd(params)
        out = self.fwd.forward(params_, fwhms=params['fwhm'], cws=params['CW'])

        # cut to spectral_fwdnn_bands
        if self.spectral_fwdnn_bands is not None and cut_to_fwdnn_bands:
            out = select(out, windows=self.spectral_fwdnn_bands, axis=1)

        if self.with_spectral_deriv:
            out = differ(out, axis=1)

        return out, None, None

    def simulate_toc(self):
        return None

    def rho_model(self, wvl, rho_params):
        rho0, rho_slope, e = rho_params.transpose(1, 0)
        return (rho0 + rho_slope * (wvl - 740) + 0.5 * rho_slope * (e - 1) / (780 - 740) * (wvl - 740) ** 2).unsqueeze(
            1)

    def to_dict(self, vect, var_names, lens_per_name):
        params = dict()

        l_ = dict([(name, 1) for name in var_names])
        l_.update(dict([(k, v) for k, v in lens_per_name.items() if k in var_names]))

        i = 0
        for var, len_ in l_.items():
            params[var] = vect[:, i:i + len_]
            i += len_

        return params

    def mask_input(self, inp, spectrum, **kwargs):
        inp, mask_, kwargs = zero_nonfinite(inp, **kwargs)
        
        if self.mask_dark_px:
            #mask = torch.where(torch.max(inp[:, :self.spectral_inp_len], dim=1)[0] < 30)
            #mask_[mask] = True
            
            mask = torch.where(spectrum[:, self.band780nm] < spectrum[:, self.band680nm])
            mask_[mask] = True

        return inp, mask_, kwargs

    def forward(self, x, **kwargs):
        
        x, mask, kwargs = self.mask_input(x, **kwargs)
        x = self.inp_norm(x)

        # HyPlant to FluoMap dictionary of meta_info names

        ids = None
        if self.with_ids:
            ids = self.get_ids(x, **kwargs)

        if self.with_ids and self.ids_mode in self.IDS_MODE_TO_ENC:
            x_ = torch.cat([x, ids], dim=1)
        else:
            x_ = x

        enc_ = self.enc(x_)

        mvs = x[:, self.spectral_inp_feat_len:]
        if self.pass_meta_vars_to_dec:
            enc = torch.cat((enc_, mvs), dim=1)

        else:
            enc = enc_

        if self.with_ids and self.ids_mode in self.IDS_MODE_TO_DEC:
            enc = torch.cat([enc, ids], dim=1)

        mean_along_enc = enc.mean(axis=-1)
        mean_across_enc = enc.mean(axis=-2)

        params = dict()

        if self.with_xtrack_vars and self.dec_xtrack is not None:
            if not self.xtrack_vars_on_off_nadir:
                shape_[1] = len(self.xtrack_fwd_inputs)
                batch_dim = mean_along_enc.shape[0]
                spatial_dim = mean_along_enc.shape[-1]
                channel_dim = len(self.xtrack_fwd_inputs)

                mean_along_enc = mean_along_enc.permute(0, 2, 1).flatten(start_dim=0, end_dim=1)
                x_xtrack = self.dec_xtrack(mean_along_enc)
                x_xtrack = x_xtrack.reshape(batch_dim, spatial_dim, channel_dim).permute(0, 2, 1).unsqueeze(
                    -1).expand(shape_)

            elif self.xtrack_vars_on_source_id:
                if self.xtrack_vars_source_id_trainable:
                    ids = self.get_ids(x, reduce_to_first_n=self.xtrack_id_reduce_to_n, **kwargs)
                else:
                    ids = kwargs['source_id'][:, None, None, None].expand(kwargs['off_nadir'].shape)

                x_track_inp = ids
                if self.pass_meta_vars_to_xtrack_on_off_nadir:
                    x_track_inp = torch.cat([x_track_inp, mvs], dim=1)

                if self.pass_dist_to_xtrack_vars_on_off_nadir:
                    x_track_inp = torch.cat([x_track_inp, kwargs['dist']], dim=1)

                if self.xtrack_off_nadir_additive_embedding:
                    emb = self.get_sinusoid_pos_embedding(kwargs['off_nadir'], ids.shape[1])
                    x_track_inp = torch.cat([x_track_inp[:, :ids.shape[1]] + emb, 
                                             x_track_inp[:, ids.shape[1]:]], 
                                            axis=1)

                elif self.xtrack_off_nadir_concat_embedding:
                    emb = self.get_sinusoid_pos_embedding(kwargs['off_nadir'], self.xtrack_off_nadir_embedding_size)
                    x_track_inp = torch.cat([x_track_inp, emb], axis=1)

                else:
                    x_track_inp = torch.cat([kwargs['off_nadir'], x_track_inp], dim=1)

                x_xtrack = self.dec_xtrack(x_track_inp)

            else:
                x_xtrack = self.dec_xtrack(kwargs['off_nadir'])

            params.update(self.to_dict(x_xtrack, self.xtrack_fwd_inputs, self.param_dim_out))

        if self.with_atrack_vars and self.dec_atrack is not None:
            if not self.atrack_vars_on_dist:
                shape_[1] = len(self.atrack_fwd_inputs)
                batch_dim = mean_across_enc.shape[0]
                spatial_dim = mean_across_enc.shape[-1]
                channel_dim = len(self.atrack_fwd_inputs)

                mean_across_enc = mean_across_enc.permute(0, 2, 1).flatten(start_dim=0, end_dim=1)
                x_atrack = self.dec_atrack(mean_across_enc)
                x_atrack = x_atrack.reshape(batch_dim, spatial_dim, channel_dim).permute(0, 2, 1).unsqueeze(
                    -2).expand(shape_)

            elif self.atrack_vars_on_source_id:
                if self.atrack_vars_source_id_trainable:
                    ids = self.get_ids(x, **kwargs)
                else:
                    ids = kwargs['source_id'][:, None, None, None].expand(kwargs['dist'].shape)

                x_atrack = self.dec_atrack(torch.cat([kwargs['dist'], ids], dim=1))

            else:
                x_atrack = self.dec_atrack(kwargs['dist'])

            params.update(self.to_dict(x_atrack, self.atrack_fwd_inputs, self.param_dim_out))

        if self.dec_source is not None:
            ids = self.get_ids(x, **kwargs)
            x_source = self.dec_source(ids)

            params.update(self.to_dict(x_source, self.source_fwd_inputs, self.param_dim_out))

        if self.fix_fwhm:
            if self.global_fwhm_shift:
                params['fwhm'] = self.fwhm_estimator(x)['fwhm']

            else:
                params['fwhm'] = self.to_fwd_scale(
                    dict(fwhm=torch.ones(params['f'].shape, device=params['f'].device) \
                            * self.fixed_fwhm))['fwhm']

        if self.pass_idparam_to_enc:
            enc_dec_px_patch = enc
            for var in ('CW', 'fwhm', 'h1alt'):
                 if var in params:
                    enc_dec_px_patch = torch.cat((enc_dec_px_patch, params[var]), dim=1)

        else:
            enc_dec_px_patch = enc

        x_px = self.dec(enc_dec_px_patch)
        params.update(self.to_dict(x_px, self.px_fwd_inputs, self.param_dim_out))

        mean_enc = enc_dec_px_patch.mean(axis=(-2, -1))
        shape_ = list(params['f'].shape)

        if self.dec_atm_geo is not None:
            shape_[1] = len(self.patch_fwd_inputs)

            x_patch = self.dec_atm_geo(mean_enc)[..., None, None].expand(*shape_)
            params.update(self.to_dict(x_patch, self.patch_fwd_inputs, self.param_dim_out))

        if self.pass_meta_vars_to_fwd:
            params.update(self.to_fwd_scale(
                dict([(key, var) for key, var in kwargs.items() if key in self.meta_vars_to_fwd])))

        pred = self.to_physical_scale(params)

        # expand CW and fwhm if they are not freely estimated
        for var, is_constant_shift in dict(CW=self.constant_cw_shift, fwhm=self.constant_fwhm_shift).items():
            if not is_constant_shift and self.sensor_modes[var] != 'free':

                wvls  = self.fwd.out_wvls - self.fwd.out_wvls[self.fwd.out_wvls.shape[0] // 2]
                offset = pred[var][:, [-1]]

                var_range = self.fwd.ffsir_dict[self.FWD_INP_PARAM_NAME_DICT_INV[var]]

                for i in range(1, self.sensor_dims[var]):
                    range_ = (var_range[1] - var_range[0]) / (self.fwd.out_wvls[-1] - self.fwd.out_wvls[0]) ** i  * self.non_constant_sensor_mode_range
                    range_ = [-range_, range_]

                    #coef =  2 * (params[var][:, -i-1] - 0.5) * range_
                    coef = range_[0] + (range_[1] - range_[0]) * params[var][:, -i-1]
                    offset = torch.einsum('j, i... -> ij...', wvls ** i, coef) + offset

                pred[var] = offset
                params[var] = self.to_fwd_scale({var: pred[var]})[var]

            pred.update(pred_fwd_scale=params)

            # adapt f to be f(760)
            pred['lfluo760'] = np.exp(-.5 * (760 - 737) ** 2 / 20 ** 2) * pred['f']

            rho_params = torch.cat([pred['rho'], pred['rho_slope'], pred['e']], axis=1)
            for label, wvl in dict(rho740=741.7, rho760=760, rho750=752.2, rho755=755, rho775=775.2, rho745=744.4,
                                   rho780=780.5).items():
                pred[label] = self.rho_model(wvl, rho_params)

        return dict(pred=pred, enc_=enc_, enc=enc, mask=mask)

    def get_sinusoid_pos_embedding(self, pos, size):
        """ Make Sinusoid Encoding Table

            Args:
                num_tokens (int): number of tokens
                token_len (int): length of a token

            Returns:
                (torch.FloatTensor) sinusoidal position encoding table
        """

        def get_position_angle_vec(_pos):
            return torch.concatenate([2 * np.pi * _pos / 5000 ** (j / size)
                                      for j in range(size)], axis=1)

        with torch.no_grad():
            sinusoid_table = torch.concatenate([get_position_angle_vec(pos[[i]])
                                                    for i in range(pos.shape[0])],
                                                axis=0)#.transpose(1, 0)
            sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
            sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table

    @classmethod
    def add_model_specific_args(cls, parser, prepend=None, **kwargs):
        parser = _MLP.add_model_specific_args(parser, prepend=prepend, **kwargs)
        parser = _MLP.add_model_specific_args(parser, prepend=add_prepend(prepend, 'dec_source'), **kwargs)
        parser = SharedInLayer.add_model_specific_args(parser, prepend=prepend, **kwargs)

        parser_spec = dict([
            ('allowed_fwd_range', dict(type=float, default=None, nargs='*')),
            ('allowed_xtrack_fwd_range', dict(type=float, default=None, nargs='*')),

            ('fwd_model_file', dict(type=str)),
            ('fwd_corrector_model_file', dict(type=str, default=None)),
            ('fwd_sensitivity_corrector_model_file', dict(type=str, default=None)),

            ('model_type', dict(type=str, default='mlp')),
            ('pass_meta_vars_to_fwd', dict(type=int, default=1)),
            ('pass_meta_vars_to_dec', dict(type=int, default=0)),
            ('pass_meta_vars_to_fwd_select', dict(type=str, default=None, nargs='*')),
            ('pass_idparam_to_enc', dict(type=int, default=0)),

            ('poly_emulator_transform', dict(type=int, default=0)),
            ('sigmoid_emulator_transform', dict(type=int, default=0)),
            ('physical_emulator_transform', dict(type=int, default=0)),

            ('fixed_rho_in_ef', dict(type=int, default=0)),
            ('only_slope', dict(type=int, default=0)),

            ('poly_emulator_transform_bounds', dict(type=float, nargs='*', default=None)),
            ('emulator_transform_bounds', dict(type=float, nargs='*', default=None)),

            ('enc_to_transformer', dict(type=int, default=0)),

            ('spectral_fwdnn_bands', dict(type=int, default=None, nargs='*')),

            ('with_ids', dict(type=int, default=0)),
            ('ids_mode', dict(type=str, default='to_enc_dec')),
            ('id_size', dict(default=8, type=int)),
            ('init_ids_from', dict(default=None, nargs='*', type=int)),
            ('load_ids_from_ckpt', dict(default=False, type=int)),
            ('load_ids_mode', dict(default=None, type=str)),

            ('fix_fwhm', dict(default=False, type=int)),
            ('fixed_fwhm', dict(default=False, type=float)),

            ('pxwise_wvl_shift', dict(default=False, type=int)),
            ('pxwise_fwhm_shift', dict(default=False, type=int)),

            ('geo_sourcewise', dict(default=False, type=int)),
            ('fwhm_sourcewise', dict(default=False, type=int)),

            ('with_xtrack_vars', dict(default=False, type=int)),
            ('xtrack_vars_on_off_nadir', dict(default=False, type=int)),
            ('xtrack_vars_on_source_id', dict(default=True, type=int)),
            ('xtrack_vars_source_id_trainable', dict(default=False, type=int)),
            ('xtrack_id_reduce_to_n', dict(default=None, type=int)),
            ('xtrack_off_nadir_additive_embedding', dict(default=False, type=int)),
            ('xtrack_off_nadir_concat_embedding', dict(default=False, type=int)),
            ('xtrack_off_nadir_embedding_size', dict(default=8, type=int)),

            ('pass_meta_vars_to_xtrack_on_off_nadir', dict(default=False, type=int)),
            ('pass_dist_to_xtrack_vars_on_off_nadir', dict(default=False, type=int)),

            ('with_atrack_vars', dict(default=False, type=int)),
            ('atrack_vars_on_dist', dict(default=False, type=int)),
            ('atrack_vars_on_source_id', dict(default=True, type=int)),
            ('atrack_vars_source_id_trainable', dict(default=False, type=int)),

            ('extrapolate_fwd', dict(default=False, type=str, nargs='+')),

            ('constant_cw_shift', dict(default=True, type=int)),
            ('constant_fwhm_shift', dict(default=True, type=int)),
            ('non_constant_sensor_mode', dict(default='free', type=str)),
            ('non_constant_cw_mode', dict(default='free', type=str)),
            ('non_constant_fwhm_mode', dict(default='free', type=str)),
            ('non_constant_sensor_mode_range', dict(default=1, type=float)),

            ('mask_dark_px', dict(default=0, type=int))

        ])

        parser = add_model_specific_args(parser, parser_spec=parser_spec, prepend=prepend, **kwargs)
        return parser

    def to_physical_scale(self, params, extrapolate=None, **kwargs):
        rescaled_params = dict()
        for param, val in params.items():
            if param in self.FWD_INP_PARAM_NAME_DICT_INV.keys():
                
                extrapolate_ = self.extrapolate_fwd if extrapolate is None else extrapolate
                extrapolate_ = extrapolate_ if type(extrapolate_) in (int, bool) else param in extrapolate_

                # if val == 'fwhm' and not self.constant_fwhm_shift or val == 'CW' and not self.constant_cw_shift:
                rsc_param = self.fwd.to_physical_scale(self.FWD_INP_PARAM_NAME_DICT_INV[param], val,
                                                       extrapolate=extrapolate_)

                if rsc_param is not None:
                    rescaled_params[param] = rsc_param

        return rescaled_params

    def to_fwd_scale(self, params, extrapolate=None, **kwargs):
        scaled_params = dict()
        for param, val in params.items():

            extrapolate_ = self.extrapolate_fwd if extrapolate is None else extrapolate
            extrapolate_ = extrapolate_ if type(extrapolate_) in (int, bool) else param in extrapolate_

            if param in self.FWD_INP_PARAM_NAME_DICT_INV.keys():
                sc_param = self.fwd.to_fwd_scale(self.FWD_INP_PARAM_NAME_DICT_INV[param], val,
                                                 extrapolate=extrapolate_)

                if sc_param is not None:
                    scaled_params[param] = sc_param

        return scaled_params


class FWDNN(SFMNN):
    _FITTING_MODEL = _FWDNN

    def __init__(self, with_label_reg=False, label_reg_labels=None, label_reg_mults=None, mask_h2alt=False, mask_ql=False,
                 mask_cloud=False, with_border_reg=False, border_reg_weight=1, with_shift_reg=False, shift_reg_weight=False,
                 border_reg_excluded_vars=None, encoding_unlearn_xtrack=False, load_discr_on_init=False, mask_h2alt_0=False,
                 encoding_unlearn_xtrack_weight=1, params_unlearn_xtrack=False, shift_reg_exclude=None, 
                 with_smile_corr_reg=False, smile_corr_reg_weight=1, off_nadir_label_norm=None, with_consistency_patch_reg=False, 
                 consistency_patch_reg_weight=1, *args, **kwargs):

        super(FWDNN, self).__init__(*args, **kwargs)
        self.input_stack_order = to_ffs_names(self.input_stack_order)
        self.with_label_reg = with_label_reg
        self.label_reg_labels = label_reg_labels
        self.label_reg_mults = label_reg_mults

        self.with_border_reg = with_border_reg
        self.border_reg_weight = border_reg_weight
        self.border_reg_excluded_vars = border_reg_excluded_vars if border_reg_excluded_vars is not None else []

        self.with_shift_reg = with_shift_reg
        self.shift_reg_weight = shift_reg_weight
        self.shift_reg_exclude = shift_reg_exclude if shift_reg_exclude is not None else []

        self.with_smile_corr_reg = with_smile_corr_reg
        self.smile_corr_reg_weight = smile_corr_reg_weight

        self.with_consistency_patch_reg = with_consistency_patch_reg
        self.consistency_patch_reg_weight = consistency_patch_reg_weight

        self.mask_h2alt = mask_h2alt
        self.mask_h2alt_0 = mask_h2alt_0
        self.mask_cloud = mask_cloud
        self.mask_ql = mask_ql

        self.encoding_unlearn_xtrack = encoding_unlearn_xtrack
        self.encoding_unlearn_xtrack_weight = encoding_unlearn_xtrack_weight
        self.encoding_unlearn_xtrack_configs = [args, kwargs]

        self.params_unlearn_xtrack = params_unlearn_xtrack
        
        self.off_nadir_label_norm = off_nadir_label_norm  
        if off_nadir_label_norm is None:
            self.off_nadir_label_norm = [512, 1024]
  
        if load_discr_on_init:
            self.set_up_discr()

    def on_load_checkpoint(self, checkpoint, *args, **kwargs):
        super(FWDNN, self).on_load_checkpoint(checkpoint, *args, **kwargs)

        self.model.on_load_checkpoint(checkpoint, device=self.device)

    def set_up_discr(self):
        self.automatic_optimization = False

        if self.encoding_unlearn_xtrack or self.params_unlearn_xtrack:

            if self.encoding_unlearn_xtrack:
                self._optimizer_prefixes = [None, 'discr0__']

                args, kwargs = self.encoding_unlearn_xtrack_configs 
                #enc_dim = self.model.enc.model.model.out.pre_modules[0].weight.shape[0]
                dim_in = self.model.enc_dim_out_orig
                
            if self.params_unlearn_xtrack:
                self._optimizer_prefixes = [None, 'discr0__']

                args, kwargs = self.encoding_unlearn_xtrack_configs 
                dim_in = len(self.model.px_fwd_inputs) + \
                         len(self.model.patch_fwd_inputs) + \
                         len(self.model.atrack_fwd_inputs)

            kwargs['out_nonlin'] = None
            self.discr0__enc_to_xtrack = ApplyOnImage(nn.Sequential(_MLP(dim_in=dim_in, 
                                                                             dim_out=1,
                                                                             *args, **kwargs),
                                                                        ScaledSigmoid(-0.1, 1.1))).to(self.device)

    def _rename(self, data):
        ret = to_ffs_names(data)
        return ret

    def label_reg(self, pred, **kwargs):
        l = 0
        for key, mult in zip(self.label_reg_labels, self.label_reg_mults):
            if key in pred and key in kwargs:
                l += ((pred[key] - kwargs[key]) ** 2).mean() * mult

        return l

    def loss(self, pred, y, inp, sif, enc, optimizer_idx=0, mode='train', *args, **kwargs):

        if self.encoding_unlearn_xtrack or self.params_unlearn_xtrack:
            if not hasattr(self, 'discr0__enc_to_xtrack'):
                self.set_up_discr()

        if self._optimizer_prefixes[optimizer_idx] is None:
            l, f_ats, bg_ats, y_ats, y, mask = super(FWDNN, self).loss(pred, y, inp, sif, 
                                                                       enc, mode=mode,
                                                                       optimizer_idx=optimizer_idx, 
                                                                       *args, **kwargs)

            if self.with_label_reg:
                l += self.label_reg(pred, **kwargs)

            if self.with_border_reg:
                l += self.border_loss(pred, **kwargs)

            if self.with_shift_reg:
                l += self.shift_reg(pred, **kwargs)

            if self.encoding_unlearn_xtrack or self.params_unlearn_xtrack:
                l -= self.unlearn_off_nadir(discr=self.discr0__enc_to_xtrack, pred=pred, **kwargs) \
                        * self.encoding_unlearn_xtrack_weight
            
            # this loss must come last since pred is changed in place
            if self.with_smile_corr_reg:
                l += self.smile_corr_loss(pred, **kwargs)

            if self.with_consistency_patch_reg:
                l += self.consistency_patch_reg(enc, pred, **kwargs)

        elif self._optimizer_prefixes[optimizer_idx] == 'discr0__':
            l = self.unlearn_off_nadir(discr=self.discr0__enc_to_xtrack, pred=pred, **kwargs)

            f_ats, bg_ats, y_ats, mask = [None] * 4
                
        else:
            raise NotImplementedError(f'Loss for optimizer prefix {optimizer_idx} not known')

        return l, f_ats, bg_ats, y_ats, y, mask

    def apply_discr(self, discr, pred, **kwargs):
        if self.encoding_unlearn_xtrack:
            dinp = kwargs['enc_']

        elif self.params_unlearn_xtrack:
            dinp = torch.cat([pred[key] for set_ in 
                                [self.model.px_fwd_inputs, 
                                 self.model.patch_fwd_inputs, 
                                 self.model.atrack_fwd_inputs] 
                             for key in set_], axis=1)

        return discr(dinp)
     
    def unlearn_off_nadir(self, discr, pred, **kwargs):
        est = self.apply_discr(discr, pred, **kwargs).flatten()
        label = (kwargs['off_nadir'].flatten() + self.off_nadir_label_norm[0]) / self.off_nadir_label_norm[1]
        
        xy = torch.stack([est, label])
        pxy = histogram2d(xy[0].unsqueeze(0), xy[1].unsqueeze(0), 
                    torch.linspace(0, 1, 100, device=est.device), bandwidth=torch.tensor(0.1))

        return 1 - mutual_information(pxy)

    def smile_corr_loss(self, pred, **kwargs):
        pred['CW'] = torch.zeros(pred['CW'].shape, device=pred['CW'].device)
        pred['fwhm'] = torch.zeros(pred['fwhm'].shape, device=pred['fwhm'].device) 

        y_corrected, _, _ = self.model.simulate_ats(pred, **kwargs, cut_to_fwdnn_bands=True)
        return ((y_corrected - kwargs['smile_corr']) ** 2).mean() * self.smile_corr_reg_weight

    def shift_reg(self, pred, **kwargs):
        res = torch.stack([((pred['pred_fwd_scale'][var] -
                             self.model.fwd.to_fwd_scale(self.model.fwd.cw_key, torch.tensor(0)))**2).mean()
                            for var in ['CW', 'fwhm'] if not var in self.shift_reg_exclude]).mean()
        return res * self.shift_reg_weight

    def border_loss(self, pred, **kwargs):
        return torch.mean(torch.stack([((var - 1) ** 2 * 100 * (var > 1) + var ** 2 * (var < 0) * 100).mean(1)     
                          for var in pred['pred_fwd_scale'].values() if var not in self.border_reg_excluded_vars])) * self.border_reg_weight

    def clone_dict(self, dic):
        return dict([(k, v.clone()) if torch.is_tensor(v) else (k, self.clone_dict(v)) for k, v in dic.items()])

    def consistency_loss(self, inp, f_ats=None, pred=None, consistency_under_eval=True,
                         consistency_on_toc=False, mode='train', **kwargs):

        lo, hi = search_spectral_window(*self.at_sensor_wvls[[0, -1]], where=self.in_wvls)[0]
        hi += 1
        window = slice(lo, hi)

        y_ats, _, _ = self.model.simulate_ats(pred, cut_to_fwdnn_bands=False, **kwargs)

        shape_ = pred['f'].shape
        f_new = torch.rand(np.prod(shape_), device=inp.device).reshape(shape_)
        df = f_new - pred['pred_fwd_scale']['f']
        
        pred_new = self.clone_dict(pred)
        pred_new.update(self.model.to_physical_scale(dict(f=f_new)))

        y_ats_new, _, _ = self.model.simulate_ats(pred_new, cut_to_fwdnn_bands=False,
                                                  **kwargs)
        dLe = (y_ats_new - y_ats)  # .detach()

        if self.model.spectral_fwdnn_bands is not None:
            dLe = select(dLe, windows=self.model.spectral_fwdnn_bands, axis=1)

        inp_copy = inp.clone()
        inp_copy[:, window] = inp[:, window] + dLe

        pred2 = self.forward(inp_copy, **kwargs)['pred']
        const_ = ((pred2['pred_fwd_scale']['f'] - (pred['pred_fwd_scale']['f'] + df)) ** 2).squeeze()

        keep_constant = []

        if self.R_consistency:
            keep_constant += ['rho', 'rho_slope', 'e']

        if self.atmo_consistency:
            keep_constant += ['h2ostr', 'neg_AOT']

        if self.geo_consistency:
            keep_constant += ['parm1', 'parm2', 'h2alt']

        if self.sensor_consistency:
            keep_constant += ['fwhm', 'CW']

        for key in keep_constant:
            const_ += ((pred2['pred_fwd_scale'][key] - pred['pred_fwd_scale'][key]) ** 2).mean(1)

        if self.consistency_f_weighted:
            const_ *= pred['pred_fwd_scale']['f'].squeeze().detach()

        return const_
    
    def consistency_patch_reg(self, enc, pred, **kwargs):
        enc = enc.transpose(1, 0).flatten(start_dim=1).transpose(1, 0)
        x_patch = self.model.dec_atm_geo(enc)
        new_patch_vars = self.model.to_dict(x_patch, self.model.patch_fwd_inputs, self.model.param_dim_out)

        l = 0
        
        for var in new_patch_vars.keys():
            l += (new_patch_vars[var] - pred['pred_fwd_scale'][var].squeeze()) ** 2

        return l * self.consistency_patch_reg_weight

    def mask_loss(self, preds, loss=None, premask=None, **kwargs):
        if premask is None:
            mask = None
        else:
            mask = premask
        
        if self.mask_cloud:
            mask_cl = kwargs['cloud'].squeeze()
 
            if mask is None:
                mask = mask_cl

            else:
                mask = torch.logical_or(mask, mask_cl)

        if self.mask_ql:
            ql_keys = [k for k  in kwargs.keys() if k.startswith('ql_mask')]
            mask_ql = logical_or(*[kwargs[key].squeeze() for key in ql_keys])

            if mask is None:
                mask = mask_ql

            else:
                mask = torch.logical_or(mask, mask_ql)

        if self.mask_h2alt:
            mask_h = torch.logical_or(preds['h2alt'] > self.model.fwd.ffsir_dict[\
                                         self.model.FWD_INP_PARAM_NAME_DICT_INV['h2alt']][1], 
                                      preds['h2alt'] < self.model.fwd.ffsir_dict[\
                                         self.model.FWD_INP_PARAM_NAME_DICT_INV['h2alt']][0] ).squeeze()

            if mask is None:
                mask = mask_h

            else:
                mask = torch.logical_or(mask, mask_h)
        
        if self.mask_h2alt_0:
            mask_h = (preds['h2alt'] <= 0).squeeze()

            if mask is None:
                mask = mask_h

            else:
                mask = torch.logical_or(mask, mask_h)

        if loss is not None and mask is not None:
            mask = torch.logical_or(mask, torch.isnan(loss).squeeze())

        elif loss is not None:
            mask = torch.isnan(loss).squeeze()


        return mask

    @classmethod
    def add_model_specific_args(cls, parser, prepend=None, **kwargs):
        parser = super(FWDNN, cls).add_model_specific_args(parser, prepend=prepend, **kwargs)

        parser_spec = dict([
            ('with_label_reg', dict(type=int, default=False)),
            ('label_reg_labels', dict(type=str, nargs='*', default=None)),
            ('label_reg_mults', dict(type=float, default=None, nargs="*")),

            ('with_border_reg', dict(type=int, default=False)),
            ('border_reg_weight', dict(type=float, default=1)),
            ('border_reg_excluded_vars', dict(type=str, default=None, nargs='*')),

            ('with_shift_reg', dict(type=int, default=False)),
            ('shift_reg_weight', dict(type=float, default=1)),
            ('shift_reg_exclude', dict(type=str, default=None, nargs='*')),

            ('with_smile_corr_reg', dict(type=int, default=False)),
            ('smile_corr_reg_weight', dict(type=float, default=1)),

            ('with_consistency_patch_reg', dict(type=int, default=False)),
            ('consistency_patch_reg_weight', dict(type=float, default=1)),
            
            ('mask_h2alt', dict(type=int, default=False)),
            ('mask_h2alt_0', dict(type=int, default=False)),
            ('mask_cloud', dict(type=int, default=False)),
            ('mask_ql', dict(type=int, default=False)),

            ('encoding_unlearn_xtrack', dict(type=int, default=False)),
            ('encoding_unlearn_xtrack_weight', dict(type=float, default=1)),

            ('params_unlearn_xtrack', dict(type=int, default=False)),

            ('load_discr_on_init', dict(type=int, default=False)),
            ('off_nadir_label_norm', dict(type=int, default=None, nargs=2))

        ])

        parser = add_model_specific_args(parser, parser_spec=parser_spec, prepend=prepend, **kwargs)
        return parser
