import h5py

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import pytorch_lightning as pl
import numpy as np

from torch.utils.data._utils.collate import default_collate

from fluomapper.utils.func.interp1d import Interp1d, Interpolate
import copy
import pandas as pd
import xarray as xr

from fluomapper.utils.data import search_spectral_window, select
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

import faiss

from fluomapper.data.desis.desis import DESISModule, DESISMetaVarDataModule
import pickle as pkl

from functools import partial


class HDFArray(object):
    
    DESIS_L2A_WVLS = np.array([741.9, 744.4, 747. , 749.7, 752.2, 755.0, 775.2, 777.9, 780.5])
    DESIS_L2A_WVL_NAMES = np.array([740, 745, 747, 749, 750, 755, 775, 777, 780])
    
    def __init__(self, hdf_groups, loaded_keys, wvls=None, query=None, allowed_lsensor_range=(0, 100)):
        

        """

        :param hdf_group: root group to which loaded_keys refers
        :param loaded_keys: keys that are loaded to memory, all other __getitem__ calls will read from disk
        """
        self.keys = loaded_keys
        self.hdf_groups = hdf_groups
        self.wvls = self.hdf_groups[0]['srf_cw'][0]

        self.special_keys = {'rho': ('rho', lambda x: x[:, [0]]),
                             'rho_slope': ('rho', lambda x: x[:, [1]]),
                             'e': ('rho', lambda x: x[:, [2]]),
                             'rho740':  ('rho', lambda x: self.rho_model(741.9, x[:].transpose())),
                             'rho760': ('rho', lambda x: self.rho_model(760, x[:].transpose())),
                             'rho750': ('rho', lambda x:  self.rho_model(752.2, x[:].transpose())),
                             'rho775': ('rho', lambda x:  self.rho_model(775.2, x[:].transpose())),
                             'rho755': ('rho', lambda x:  self.rho_model(755.0, x[:].transpose())),
                             'rho745': ('rho', lambda x:  self.rho_model(744.4, x[:].transpose())),
                             'rho780': ('rho', lambda x:  self.rho_model(780.5, x[:].transpose())),
                             'lfluo740': ('lfluo', lambda x: self.f_model(741.9, x[:].transpose())),
                             'lfluo760': ('lfluo', lambda x: self.f_model(760, x[:].transpose())),
                             'lfluo750': ('lfluo', lambda x: self.f_model(752.2, x[:].transpose())),
                             'lfluo775': ('lfluo', lambda x: self.f_model(775.2, x[:].transpose())),
                             'lfluo780': ('lfluo', lambda x: self.f_model(780.5, x[:].transpose())),
                             'rho_full': ('rho', lambda x: self.rho_model(self.wvls, x[:].transpose())),
                             'fluo_full': ('lfluo', lambda x: self.f_model(self.wvls, x[:].transpose())),
                             'aoi': ('parm2', lambda x: x),
                             'dem': ('h2alt', lambda x: x),
                             'sza': ('parm2', lambda x: x),
                             'f': ('lfluo', lambda x: x[:, [0]]),
                             'CW': ('cw_shift',  lambda x: x),
                             'fwhm': ('fwhm_shift',  lambda x: x),
                             'neg_AOT': ('vis', lambda x: x)
                            }

        #apparent reflectance
        self.interpolator = Interpolate(x=torch.from_numpy(self.wvls), xnew=torch.from_numpy(self.DESIS_L2A_WVLS))
        for i, (name, wvl) in enumerate(zip(self.DESIS_L2A_WVL_NAMES, self.DESIS_L2A_WVLS)):
            self.special_keys['app_rho'+str(name)] = (['rho', 'lfluo', 'eg0_conv'], partial(self.app_rho_model, index=i))

        self.special_keys['app_rho_full'] = (['rho', 'lfluo', 'eg0_conv'], partial(self.app_rho_model, index=None))

        # define rho input params
        #for i, wvl in enumerate([741.9, 752.2, 762.9, 772.6, 780.5]):
        #    self.special_keys[f'r{i}'] = ('rho', lambda x: self.rho_model(wvl, x[:].transpose()))

        self.saved_vars = pd.DataFrame()
        
        print('SPECIAL_KEYS', self.special_keys.keys())
        for key in self.keys:
            if key in self.special_keys:
                key_, key_func = self.special_keys[key]
                val = np.concatenate([np.array(key_func(*self.group_getter(hdf_group, key_)))
                                      for hdf_group in self.hdf_groups], axis=0)

            else:
                val = np.concatenate([np.array(hdf_group[key]) 
                                      for hdf_group in self.hdf_groups], axis=0)

            # type conversion
            # no byte strings
            if val.dtype == 'O':
                val = val.astype('<U32')
            
            if len(val.squeeze().shape) == 1:
                self.saved_vars[key] = val.squeeze()

            else:
                setattr(self, key, val)

        self.pd_keys = list(self.saved_vars.keys())
        self.non_pd_keys = list(set(self.keys) - set(self.pd_keys))

        self.saved_vars = self.saved_vars.reset_index(drop=True)
        print('QUERY', query)
        if query is not None:
            self.saved_vars = self.saved_vars.query(query)
            
            for key in self.non_pd_keys:
                setattr(self, key, getattr(self, key)[self.saved_vars.index])

            self.saved_vars = self.saved_vars.reset_index(drop=True)
    
        # filter band indices
        if allowed_lsensor_range is not None and 'lsensor' in self.keys:
            bad_inds = np.unique(np.where(np.logical_or(self.lsensor > allowed_lsensor_range[1], self.lsensor < allowed_lsensor_range[0]))[0])
            
            for key in self.non_pd_keys:
                setattr(self, key, np.delete(getattr(self, key), bad_inds, axis=0))

            self.saved_vars = self.saved_vars.drop(index=bad_inds)

    def group_getter(self, group, keys):
        if type(keys) is str:
            keys = [keys]

        ret = []
        for key in keys:
            ret.append(group[key])

        return ret

    def f_model(self, wvl, params):
        if not hasattr(wvl, '__iter__'):
            return np.exp(-.5 * (wvl - 737) ** 2/ 20**2) * params[0]

        else:
            return np.exp(-.5 * (wvl[None, :] - 737) ** 2/ 20**2) * params[0][:, None]

    def rho_model(self, wvl, params):
        if not hasattr(wvl, '__iter__'):
            rho0, rho_slope, e, lo, up = params
            return rho0 + rho_slope * (wvl - lo) + 0.5 * rho_slope * (e - 1) / (up - lo) * (wvl - lo) ** 2

        else:
            rho0, rho_slope, e, lo, up = params
            rho0, rho_slope, e, lo, up = rho0[:, None], rho_slope[:, None], e[:, None], lo[:, None], up[:, None]
            return rho0 + rho_slope * (wvl[None, :] - lo) + 0.5 * rho_slope * (e - 1) / (up - lo) * (wvl - lo) ** 2

    def app_rho_model(self, rho_params, fluo_params, eg0, index=None):        
        if index is not None:
            eg0 = self.interpolator(torch.from_numpy(eg0[:])).numpy()[:, index] * 10

            wvl = self.DESIS_L2A_WVLS[index]
            rho = self.rho_model(wvl, rho_params[:].transpose())
            lf = self.f_model(wvl, fluo_params[:].transpose())

        else:
            eg0 = eg0[:] * 10
            rho = self.rho_model(self.wvls, rho_params[:].transpose())
            lf = self.f_model(self.wvls, fluo_params[:].transpose())

        app_rho = rho + lf * np.pi / eg0
        return app_rho

    def __getitem__(self, key):
        if key in self.pd_keys:
            return self.saved_vars[key].values
        
        elif key in self.non_pd_keys:
            return getattr(self, key)

        else:
            raise Exception(f'variable {key} is not loaded')
            if key in self.special_keys:
                key_, key_func = self.special_keys[key]
                return [key_func(hdf_group[key_]) for hdf_group in self.hdf_groups]

            else:
                return [hdf_group[key_] for hdf_group in self.hdf_groups]

    def __len__(self):
        return getattr(self, self.keys[0]).shape[0]

    def get(self, key, item=None):
        if item is None:
            return self[key]

        else:
            return self[key][item]


class NoiseModule(object):
    def __init__(self, wvls=None, additive_noise=0, meta_info_noise=None, meta_info_noise_per_px=True, rho_slope_distort=False, rho_pca_distort=False, rho_l2a_distort=None,
            rho_flat_distort=False, rho_l2a_distort_poly=False, meta_info_independent_noise=1, meta_info_dep_noise=None, apparent_refl_noise=0):

        self.wvls = wvls

        self.additive_noise = additive_noise

        self.meta_info_noise = meta_info_noise
        self.meta_info_dep_noise = meta_info_dep_noise
        self.meta_info_noise_per_px = meta_info_noise_per_px 
        self.meta_info_independent_noise = meta_info_independent_noise
        self.apparent_refl_noise = apparent_refl_noise

        self.rho_slope_distort = rho_slope_distort
        self.rho_pca_distort = rho_pca_distort
        self.rho_l2a_distort = rho_l2a_distort if rho_l2a_distort != 'none' else None
        self.rho_l2a_distort_poly = rho_l2a_distort_poly
        self.rho_flat_distort = rho_flat_distort

        if self.rho_pca_distort:
            self.mean_rho = torch.from_numpy(np.load('/home/user/.../refl_distort_mean.npy')).requires_grad_(False)
            self.mean_rho_wvls = torch.from_numpy(np.load('/home/user/.../refl_distort_mean_wvls.npy')).requires_grad_(False)
            self.rho_pca_distort_window = None
            self.left_window = None

        if self.rho_l2a_distort is not None:
            desis_dm = DESISMetaVarDataModule(paths=[self.rho_l2a_distort], num_workers=1, meta_vars=['rho'],
                                              spectral_window_wvl=[0, 2000],  # spectral_window_wvl=[700, 800], 
                                              include_wvl_info=True, spectral_window=None, load=True, load_train=False, 
                                              train_files=[0], val_files=[0],
                                              batch_size=1, reader_batch_size=1, val_frac=1.0, out_mode="single_spectra", 
                                              return_wvl=True, shuffle_reading=False)
            desis_dm.prepare_data()
            desis_dm.setup()

            self.l2a = desis_dm.val_dset.readers[1].post_process(0, desis_dm.val_dset.readers[1].sources[0]).numpy().transpose(2, 1, 0) / 10
            self.l2a = torch.from_numpy(self.l2a).flatten(start_dim=1).transpose(1, 0)
            self.l2a = self.l2a[torch.where(torch.all(self.l2a > 0, axis=-1))]
            self.l2a_wvls = np.array([ 401.9, 404.1, 406.7, 409.3, 411.7, 414.3, 416.8, 419.4, 421.9,
                                       424.5, 427.2, 429.8, 432.4, 434.9, 437.4, 439.9, 442.4, 444.9,
                                       447.7, 450.3, 452.8, 455.5, 458. , 460.7, 463.1, 465.7, 468.2,
                                       470.7, 473.3, 475.8, 478.5, 481.1, 483.7, 486.3, 488.9, 491.4,
                                       493.8, 496.4, 499.2, 501.6, 504.2, 506.8, 509.5, 512. , 514.6,
                                       517. , 519.6, 522. , 524.6, 527.1, 529.7, 532.2, 534.8, 537.5,
                                       540. , 542.5, 545.1, 547.7, 550.3, 552.9, 555.5, 558. , 560.5,
                                       563. , 565.7, 568.3, 570.9, 573.5, 576. , 578.5, 581.1, 583.5,
                                       586.1, 588.7, 591.3, 593.9, 596.5, 598.9, 601.5, 604.1, 606.6,
                                       609.2, 611.7, 614.3, 616.8, 619.4, 622. , 624.6, 627. , 629.6,
                                       632.1, 634.6, 637.2, 639.7, 642.2, 644.9, 647.5, 649.9, 652.5,
                                       655.1, 657.6, 660.1, 662.6, 665.2, 667.7, 670.3, 673.1, 675.7,
                                       678.3, 680.8, 683.4, 685.8, 688.3, 690.8, 693.4, 696. , 698.8,
                                       701.3, 703.9, 706.7, 709.3, 711.6, 713.9, 716.3, 718.8, 721.4,
                                       724.1, 726.7, 729.4, 731.9, 734.2, 736.8, 739.4, 741.9, 744.4,
                                       747. ,
                                       749.7, 752.2, 755. , 757.7, 760.3, 762.9, 764.8, 767.5,
                                       770.2, 772.6, 775.2, 777.9, 780.5, 783. , 785.5, 788.2, 790.4,
                                       793.1, 795.8, 798.2, 801. , 804. , 806.5, 809.1, 811.6, 814.2,
                                       816.8, 819.6, 822.7, 824.2, 827.1, 829.1, 832.1, 834.8, 836.6,
                                       839.9, 841.9, 844.6, 847.7, 849.8, 852.4, 855.2, 857.8, 860.2,
                                       862.8, 865.4, 867.9, 870.5, 873.1, 875.6, 878.6, 881.4, 883. ,
                                       885.3, 888. , 890.8, 893.9, 895.9, 898.3, 901.1, 903.7, 905.8,
                                       908.5, 911.5, 914.7, 916.4, 918.3, 920.8, 923.7, 927. , 929.6,
                                       931.7, 934.4, 937.2, 939.3, 941.8, 944.7, 947.2, 949.4, 951.8,
                                       954.2, 957.2, 959.5, 962.2, 965.3, 968.1, 970.3, 972.9, 975.9,
                                       978.5, 979.9, 981.9, 984.8, 988.8, 991.6, 993.1, 995.6, 997.8,
                                       999.5])
            
            self.l2a_interp = self._interpolate_l2a(self.l2a, self.l2a_wvls)

            self.l2a_740_ind = search_spectral_window(740, where=self.l2a_wvls)[0][0]
            self.l2a_750_ind = search_spectral_window(750, where=self.l2a_wvls)[0][0]

            self.l2a_slope_window = search_spectral_window(750, 775, where=self.l2a_wvls)
    
    def _interpolate_l2a(self, l2a, l2a_wvls):
        spectral_window_o2a = (750, 767)
        spectral_window_o2a = search_spectral_window(*spectral_window_o2a, where=l2a_wvls)

        spectral_window_non_o2a = (0, 755, 770, 100000)
        spectral_window_non_o2a = search_spectral_window(*spectral_window_non_o2a, where=l2a_wvls)

        refl2 = select(l2a, windows=spectral_window_non_o2a, axis=-1)
        desis_wvls2 = select(l2a_wvls, windows=spectral_window_non_o2a, axis=-1)

        o2a_wvls = select(l2a_wvls, windows=spectral_window_o2a, axis=-1)
        interp = interp1d(desis_wvls2, refl2, kind='linear')
        interp_ = interp(o2a_wvls)

        refl3 = torch.cat([l2a[:, :spectral_window_non_o2a[0][-1]], 
                           torch.from_numpy(interp_), 
                           l2a[:, spectral_window_non_o2a[0][-1] + interp_.shape[-1]:]], axis=-1).float()

        return refl3

    def apply_noise(self, vars_, dset, item):
        if self.rho_slope_distort or self.rho_pca_distort or self.rho_l2a_distort or self.rho_flat_distort:
            cw_shifts = torch.from_numpy(np.atleast_1d(dset.get('cw_shift', item)))
            rho_slope = torch.from_numpy(np.atleast_1d(dset.get('rho_slope', item)))
            lambda0 = 740
            rho = torch.from_numpy(np.atleast_1d(dset.get('rho', item)))

            if self.rho_pca_distort:
                max_, min_ = 5, 1
                new_rho = self.mean_rho * (torch.rand(1) * (max_ - min_) + min_) 
                
                if self.rho_pca_distort_window is None:
                    self.rho_pca_distort_window = search_spectral_window(self.wvls[0], self.wvls[-1],
                                                                         where=self.mean_rho_wvls)
                
                new_rho = select(new_rho, windows=self.rho_pca_distort_window, inclusive=True) 
                mult = new_rho / (rho + rho_slope * (self.wvls + cw_shifts - lambda0))

                if 'rho_slope' in vars_:
                    slope, bias = np.polyfit(self.wvls, new_rho, deg=1)
                    vars_['rho_slope'] = slope * torch.ones(vars_['rho_slope'].shape)

                if 'rho' in vars_ or 'rho750' in vars_:
                    if self.left_window is None:
                        self.left_window = search_spectral_window(740, 755, where=self.wvls)

                    wvls_, new_rho_ = select(self.wvls, windows=self.left_window), select(new_rho,
                                                                                          windows=self.left_window)

                    slope, bias = np.polyfit(wvls_, new_rho_, deg=1)
                    vars_['rho'] = bias + 740 * slope * torch.ones(vars_['rho'].shape)
                    vars_['rho750'] = bias + 750 * slope * torch.ones(vars_['rho'].shape)

            if self.rho_slope_distort:
                slope_mult = torch.rand(1) * self.rho_slope_distort
                slope_mult = max(1, slope_mult)
                                              
                mult = (rho + rho_slope * (self.wvls + cw_shifts - lambda0) * slope_mult) /\
                       (rho + rho_slope * (self.wvls + cw_shifts - lambda0))

                if 'rho_slope' in vars_:
                    vars_['rho_slope'] *= slope_mult

                if 'rho_full' in vars_:
                    vars_['rho_full'] = rho + rho_slope * (self.wvls - lambda0) * slope_mult

                #if 'rho750' in vars_:
                #    vars_['rho750'] = rho + rho_slope * 10 * slope_mult

            #if self.rho_curv_distort:
                #slope_ = torch.rand(1) * 0.004
                #curv = np.poly1d([-3.70061151e-02,  1.52349191e-06])[slope]

                #new_rho = np.poly(
                #pass

            if self.rho_l2a_distort:
                rand_ind = np.random.randint(self.l2a_interp.shape[0])
                new_rho = self.l2a_interp[rand_ind]
                
                interp = interp1d(self.l2a_wvls, new_rho, kind='linear', fill_value='extrapolate')
                wvls_cw_sh = self.wvls + cw_shifts.cpu().numpy()
                new_rho_in_wvls_cw_sh = interp(wvls_cw_sh)
                new_rho_in_wvls = interp(self.wvls)
                #new_rho_in_wvls_cw_sh /= new_rho_in_wvls_cw_sh[len(new_rho_in_wvls_cw_sh) // 2] 
                
                if self.rho_l2a_distort_poly:
                    fit_ = np.polyfit(self.wvls, new_rho, deg=5)
                    new_rho_in_wvls_cw_sh = np.poly1d(fit_)(wvls_cw_sh)

                sim_rho = rho + rho_slope * (self.wvls + cw_shifts - lambda0)
                #sim_rho /= sim_rho[len(sim_rho) // 2]

                mult = new_rho_in_wvls_cw_sh / sim_rho 
                #mult = np.clip(mult, a_min=0, a_max=6) 

                if 'rho' in vars_ or 'rho750' in vars_:
                    shape = vars_['rho'].shape if 'rho' in vars_ else vars_['rho750'].shape
                    vars_['rho'] = new_rho[self.l2a_740_ind] * torch.ones(shape)
                    vars_['rho750'] = new_rho[self.l2a_750_ind] * torch.ones(shape)

                if 'rho_slope' in vars_:
                    wvls_, new_rho_ = select(self.l2a_wvls, windows=self.l2a_slope_window), select(new_rho, windows=self.l2a_slope_window)
                    slope, bias = np.polyfit(wvls_, new_rho_, deg=1)
                    #slope = (new_rho_[-1] - new_rho_[0]) / 15
                    vars_['rho_slope'] = slope * torch.ones(vars_['rho_slope'].shape)

                if 'rho_full' in vars_:
                    vars_['rho_full'] = torch.from_numpy(new_rho_in_wvls_cw_sh).float()
                
            if self.rho_flat_distort:
                mult = rho / (rho + rho_slope * (self.wvls + cw_shifts - lambda0))

                if 'rho_slope' in vars_:
                    vars_['rho_slope'] = torch.zeros(vars_['rho_slope'].shape) 

                #if 'rho_full' in vars_:
                #    vars_['rho_full'] = torch.einsum('ij, j', torch.ones(vars_['rho_full'].shape), rho.mean(axis=1))

            vars_['obs'] = torch.einsum('j..., j -> j...', vars_['obs'], mult)

        if self.additive_noise:
            vars_['obs'] = vars_['obs'] + torch.randn(*vars_['obs'].shape, device=vars_['obs'].device) \
                               * torch.sqrt(vars_['obs'] * self.additive_noise)

        if self.apparent_refl_noise:
            rand = torch.randn(1, device=vars_['obs'].device) * self.apparent_refl_noise

            for key in [key for key in vars_ if key.startswith('rho')]:
                vars_[key] += rand * vars_['lfluo' + key[len('rho'):]]

        if self.meta_info_noise is not None:

            if self.meta_info_independent_noise in (0, 2):
                key_ = list(self.meta_info_noise.keys())[0]
                non_indep_noise_var = torch.randn(*vars_[key_].shape, device=vars_[key_].device)

            for key in self.meta_info_noise.keys():
                vars_[key+'_nonoise'] = vars_[key].clone()

                if self.meta_info_noise_per_px:
                    if self.meta_info_independent_noise in (1, 2):
                         vars_[key] = vars_[key] + torch.randn(*vars_[key].shape, device=vars_[key].device) \
                                        * self.meta_info_noise[key]

                    if self.meta_info_independent_noise == 0:
                        vars_[key] = vars_[key] + non_indep_noise_var * self.meta_info_noise[key]

                    if self.meta_info_independent_noise == 2:
                        vars_[key] = vars_[key] + non_indep_noise_var * self.meta_info_dep_noise[key] 

                else:
                    vars_[key] = vars_[key] + (torch.randn(vars_[key].shape[0], device=vars_[key].device) * var)[:, None, None]

        return vars_


class SimulationDset(Dataset):
    def __init__(self, database_path=None, indices=None, load=False, meta_info=None, return_wvl=False,
                 shuffle=False, with_interpolation=False, additive_noise=False, query=None,
                 meta_info_noise_keys=None, meta_info_noise_vals=None, meta_info_noise_per_px=True, rho_slope_distort=False,
                 meta_info_exclude=None, rho_pca_distort=False, rho_l2a_distort=None, label='lfluo760',
                 spectral_window_obs_wvl=None, out_mode='single_spectra', out_shape=10,
                 rho_flat_distort=False, reader_batch_size=1, from_pkl=None, kneighbours_on_setup=False, 
                 enforce_nbrs_dist_zero=False, pass_vars=None, meta_info_independent_noise=1, 
                 meta_info_dep_noise_vals=None, apparent_refl_noise=0, **kwargs):

        """do not open hdf5 here!!"""

        super(SimulationDset, self).__init__()
        self.database_path = database_path

        if type(self.database_path) is str:
            self.database_path = [self.database_path]

        # determine if it is a DESIS or HyPlant module
        if type(self.database_path) is list and 'DESIS' in self.database_path[0]:
            self.sensor = 'DESIS'

        else:
            self.sensor = 'HyPlant'

        self.load = load
        self.kneighbours_on_setup = kneighbours_on_setup
        self.enforce_nbrs_dist_zero = enforce_nbrs_dist_zero 

        self.indices = np.array(indices) if indices is not None else None

        self.out_mode = out_mode
        self.out_shape = out_shape
        self.reader_batch_size = reader_batch_size
        
        self.gen_spec = dict(return_wvl=return_wvl, shuffle=shuffle)
        
        if meta_info is None or 'none' in meta_info:
            meta_info = []

        self.pass_vars = pass_vars if pass_vars is not None else []

        self.meta_info_exclude = meta_info_exclude
        if self.meta_info_exclude is None:
            self.meta_info_exclude = []

        self.label = label
        if type(self.label) is str:
            self.label = [self.label]

        self.from_pkl = from_pkl
        if not self.from_pkl:        
            if self.out_mode != 'windows':
                self.vars = ['lsensor'] + self.label + [m for m in meta_info if m not in self.meta_info_exclude] + self.pass_vars  # all of those are passed in batch
                self.all_vars = copy.deepcopy(self.vars) + self.meta_info_exclude + ['srf_cw', 'cw_shift']  # all of these are loaded to memory

            else:
                self.vars = ['lsensor'] + self.label + [m for m in meta_info if m not in self.meta_info_exclude]
                self.atmo_vars = ['bckzen', 'h2ostr', 'hgpf', 'iph', 'o3str', 'parm1',
                                  'parm2', 'obszen', 'tilt', 'vis', 'h2alt', 'h1alt']

                #self.nbrs = NearestNeighbors(n_neighbors=self.out_shape ** 2 - 1, algorithm='ball_tree')
                self.kneighbours = self.out_shape ** 2 - 1
                
                self.all_vars = list(set(copy.deepcopy(self.vars) + self.meta_info_exclude 
                                     + ['srf_cw', 'cw_shift'] + self.atmo_vars))  # all of these are loaded to memory
            
            self.with_interpolation = with_interpolation
            if self.with_interpolation:
                # make sure we load cw shift to memory as well
                self.all_vars += ['cw_shift']
                self.interpolator = None
               
            # this is set when the dataset is opened
            self.sensor_wvls = None
            self.query = query
            if type(query) is list:
                self.query = ' '.join(query)
            self.query = self.query.replace('__', '"') if self.query is not None else None
            
            self.spectral_window_obs_wvl = spectral_window_obs_wvl

        else:
            with open(self.database_path, 'rb') as f:
                self.inp_data, self.out_data, self.sim_wvls = pkl.load(f)

                # TODO: quick fix, okayish for HyPlant
                self.vars = ['obs'] + [m for m in meta_info if m not in self.meta_info_exclude]
                self.sensor_wvls = self.sim_wvls

        meta_info_noise = dict(list(zip(meta_info_noise_keys, meta_info_noise_vals))) if meta_info_noise_keys is not None \
                          and meta_info_noise_vals is not None else None

        meta_info_dep_noise = dict(list(zip(meta_info_noise_keys, meta_info_dep_noise_vals))) if meta_info_noise_keys is not None \
                          and meta_info_dep_noise_vals is not None else None

        self.noise = NoiseModule(wvls=None, additive_noise=additive_noise,
                                 meta_info_noise=meta_info_noise,
                                 meta_info_noise_per_px=meta_info_noise_per_px,
                                 rho_slope_distort=rho_slope_distort,
                                 rho_pca_distort=rho_pca_distort,
                                 rho_l2a_distort=rho_l2a_distort,
                                 rho_flat_distort=rho_flat_distort,
                                 meta_info_independent_noise=meta_info_independent_noise,
                                 meta_info_dep_noise=meta_info_dep_noise,
                                 apparent_refl_noise=apparent_refl_noise)

        self.f_wvl = np.array([760])

    def open_hdf5(self):
        
        ds = []
        for path in self.database_path:
            hdf5_file = h5py.File(path, 'r')
            dataset = hdf5_file[list(hdf5_file.keys())[0]]
            ds.append(dataset)

        dataset = HDFArray(ds, self.all_vars, query=self.query)
        self.sim_wvls = torch.from_numpy(dataset['srf_cw'][0]).requires_grad_(False).float() #- 0.22

        if self.out_mode == 'windows':
            atmo_arr = np.stack([np.array(dataset.get(var)).astype(float)
                                 for var in self.atmo_vars])
            
            self.atmo_arr_mean = atmo_arr.mean(axis=1)[:, None]
            self.atmo_arr_std =  atmo_arr.std(axis=1)[:, None]

            atmo_arr_norm = (atmo_arr - self.atmo_arr_mean) / (self.atmo_arr_std + 1e-3)
            self.atmo_arr_norm = atmo_arr_norm.transpose().astype('float32')

            #self.nbrs = self.nbrs.fit(self.atmo_arr_norm[self.indices])
            nbrs_data = self.atmo_arr_norm[self.indices]
            self.nbrs = faiss.index_factory(self.atmo_arr_norm.shape[-1], "IVF4000,Flat")
            self.nbrs.train(nbrs_data[np.random.randint(0, len(nbrs_data), min(len(nbrs_data), 10000))]) # 156000
            self.nbrs.add(nbrs_data)

            if self.kneighbours_on_setup:
                # self.nbrs_dists, self.nbrs_inds = self.nbrs.kneighbors(self.atmo_arr_norm[self.indices])
                self.nbrs_dists, self.nbrs_inds = self.nbrs.search(nbrs_data, self.kneighbours)
                
                if self.enforce_nbrs_dist_zero:
                    for i, row in enumerate(self.nbrs_inds):
                        if np.any(self.nbrs_dists[i] > 0):
                            mask = np.where(self.nbrs_dists[i] > 0)
                            self.nbrs_inds[i][mask] = self.nbrs_inds[i][:len(mask[0])] 

        if self.sensor == 'DESIS':
            self.sensor_wvls = torch.from_numpy(np.array([744.4, 747. , 749.7, 752.2, 755. ,
                                                          757.7, 760.3, 762.9, 764.8, 767.5,
                                                          770.2, 772.6, 775.2])).requires_grad_(False)
        else:
            self.sensor_wvls = self.sim_wvls

        self.noise.wvls = self.sim_wvls

        if self.spectral_window_obs_wvl is not None:
            self.spectral_window = search_spectral_window(*self.spectral_window_obs_wvl,
                                                          where=self.sensor_wvls if self.with_interpolation
                                                          else self.sim_wvls) 
        else:
            self.spectral_window = None

        return hdf5_file, dataset

    def setup(self): 
        if not self.from_pkl:
            self.hdf5_file, self.dataset = self.open_hdf5()

    def __getitem__(self, item, dset=None, is_absolute_item=False, return_dists=False, **kwargs):

        gen_spec = self.gen_spec.copy()
        gen_spec.update(kwargs)
        
        # if database is loaded from pickle
        if self.from_pkl:
            vars_ = dict([(key, self.inp_data[key][item]) for key in self.vars])
            y = self.out_data[item].unsqueeze(0)

            vars_ = self.noise.apply_noise(vars_, dset, item)

            if gen_spec['return_wvl']:
                return (vars_, self.sensor_wvls), (y, self.f_wvl)
 
            return vars_, y
        
        ####################################################################
        # if not continue reading from hdf5 file
        if not hasattr(self, 'hdf5_file'):
            self.hdf5_file, self.dataset = self.open_hdf5()
        
        if dset is None:
            dset = self.dataset

        # get absolute index
        if not is_absolute_item:
            item_ = self.indices[item]

        else:
            item_ = item
            #item = list(self.indices).index(item)

        if self.out_mode != 'windows':
            vars_ = dict([(key, torch.from_numpy(np.atleast_1d(dset.get(key, item_))).float())
                          for key in self.vars])

        else:
            if self.kneighbours_on_setup:
                dists, neighbours = self.nbrs_dists[item], self.nbrs_inds[item]
            else:
                # dists, neighbours = self.nbrs.kneighbors(self.atmo_arr_norm[[item_]])
                dists, neighbours = self.nbrs.search(self.atmo_arr_norm[[item_]].astype(np.float32), self.kneighbours)

            neighbours = np.array(list(neighbours.squeeze()) + [item])
            abs_neighbours_ = self.indices[neighbours]

            vars_ = dict([(key,
                           torch.from_numpy(np.atleast_1d(self.dataset.get(key, abs_neighbours_))).float()\
                           .reshape(self.out_shape, self.out_shape, -1).permute(2, 0, 1))
                          for key in self.vars])

            if return_dists:
                vars_['dists'] = torch.from_numpy(dists)
        
        y = []
        for label in self.label:
            y.append(vars_[label])
            #del vars_[label]
        y = torch.tensor(y)

        vars_['obs'] = vars_['lsensor']
        del vars_['lsensor']

        vars_ = self.noise.apply_noise(vars_, dset, item_)
        vars_, y = self.unit_transform(vars_, y)

        if self.with_interpolation:
            simulated_wvls = (self.sim_wvls + torch.from_numpy(np.atleast_1d(dset.get('cw_shift', item_))[:, None])).squeeze().float()
            vars_['obs'] = Interp1d()(simulated_wvls, vars_['obs'], self.sensor_wvls).squeeze().numpy().astype(np.float32)

            #if 'rho_full' in vars_:
            #    vars_['rho_full'] = Interp1d()(simulated_wvls, vars_['rho_full'], self.sensor_wvls).squeeze().numpy().astype(np.float32)

        if self.spectral_window is not None:
            vars_['obs'] = select(vars_['obs'], windows=self.spectral_window, axis=-1)

        if not type(vars_['obs']) is torch.Tensor:
            vars_['obs'] = torch.from_numpy(vars_['obs']).float()
    
        if gen_spec['return_wvl']:
            return (vars_, self.sensor_wvls), (y, self.f_wvl)
        
        # print('y SHAPE pre', y.shape)

        # TODO: this is just a temporary solution to multiple parameters in rho and lfluo
        for key, item in vars_.items():
            if key in ('obs', 'cw_shift', 'fwhm_shift'):
                continue

            # if len(item) > 1:
            #    vars_[key] = item[[0]]

        #if len(y) > 1:
        #    y = y[[0]]

        # print('y SHAPE post', y.shape)

        return vars_, y

    def __len__(self):
        if not self.from_pkl:
            if self.indices is None:
                if not hasattr(self, 'hdf5_file'):
                    self.hdf5_file, self.dataset = self.open_hdf5()
                    return len(self.dataset['lsensor'])

                else:
                    return self.dataset['lsensor'].shape[0]

            return len(self.indices)

        else:
            return len(self.out_data)

    def unit_transform(self, batch, y):
        batch['obs'] *= 10
        
        for key in batch.keys():
            batch[key] = batch[key].float()

        return batch, y.float()

    @classmethod
    def add_argparse_args(self, parser, *args, **kwargs):
        parser.add_argument('--load', type=int, default=0)
        parser.add_argument('--from_pkl', type=int, default=0)
        parser.add_argument('--kneighbours_on_setup', type=int, default=0)
        parser.add_argument('--enforce_nbrs_dist_zero', type=int, default=0)

        parser.add_argument('--with_interpolation', type=int, default=0)
        parser.add_argument('--meta_info', type=str, default=None, nargs="+")
        parser.add_argument('--pass_vars', type=str, default=None, nargs="+")
        parser.add_argument('--meta_info_exclude', type=str, default=None, nargs="+")
        parser.add_argument('--additive_noise', type=float, default=0)
        parser.add_argument('--meta_info_noise_keys', type=str, default=None, nargs="*")
        parser.add_argument('--meta_info_noise_vals', type=float, default=None, nargs="*")
        parser.add_argument('--meta_info_dep_noise_vals', type=float, default=None, nargs="*")
        parser.add_argument('--meta_info_independent_noise', type=int, default=1)
        parser.add_argument('--apparent_refl_noise', type=float, default=0)
        parser.add_argument('--meta_info_noise_per_px', type=int, default=True)
        parser.add_argument('--rho_slope_distort', type=float, default=0)
        parser.add_argument('--rho_pca_distort', type=int, default=0)
        parser.add_argument('--rho_flat_distort', type=int, default=0)
        parser.add_argument('--rho_l2a_distort', type=str, default=None)
        parser.add_argument('--rho_l2a_distort_poly', type=int, default=True)
        parser.add_argument('--query', type=str, default=None, nargs='+')
        parser.add_argument('--spectral_window_obs_wvl', type=float, default=None, nargs="+")

        parser.add_argument('--out_shape', type=int, default=17)
        parser.add_argument('--out_mode', type=str, default='single_spectra')

        parser.add_argument('--label', type=str, default='lfluo760', nargs="+") 

        return parser


class SyncDset(Dataset):
    def __init__(self, dataset_paths, meta_info_vars=None, indices=None, load=True):
        super(SyncDset, self).__init__()

        self.meta_info_vars = meta_info_vars if meta_info_vars is not None else []
        self.indices = indices
        self.load = load

        self.dataset_paths = dataset_paths
        if type(dataset_paths) is str:
            self.dataset_paths = [dataset_paths]

        if load:
            self._load()

    def pkl_load(self, f):
        with open(f, 'rb') as f:
            return pkl.load(f)

    def _load(self):
        self.obs = []
        self.f = []
        self.meta_info = {}
        for path in self.dataset_paths:
            dset = self.pkl_load(path)
            self.obs.append(dset['obs'] * 10)
            self.f.append(dset['f'])

            for key in dset.keys():
                if key not in ('obs', 'f'):
                    val = dset[key]
                    
                    if type(val) in (float, int):
                        val = np.ones(len(dset['f'])) * val

                    if key in self.meta_info and key in self.meta_info_vars:
                        self.meta_info[key].append(val)

                    elif key in self.meta_info_vars:
                        self.meta_info[key] = [val]

        self.obs = torch.from_numpy(np.concatenate(self.obs, axis=1).transpose(1, 0)).float()
        self.f = torch.from_numpy(np.concatenate(self.f)).float()

        for key in self.meta_info.keys():
            self.meta_info[key] = torch.from_numpy(np.concatenate(self.meta_info[key])).float()

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)

        else:
            len_ = 0
            for path in self.dataset_paths:
                len_ += len(self.pkl_load(path)['f'])

            return len_

    def __getitem__(self, item, *args, **kwargs):
        item = self.indices[item]
        d_ = dict(obs=self.obs[item])
        d_.update({k: torch.atleast_1d(self.meta_info[k][item]) for k in self.meta_info.keys()})
        
        return d_, torch.atleast_1d(self.f[item])

    def setup(self):
        pass

    def prepare_data(self):
        if self.load and not hasattr(self, 'obs'):
            self._load()


class SimulationDmodule(pl.LightningDataModule):
    def __init__(self, val_frac, database_path=None, shuffle_dataset=False, batch_size=16, persistent_workers=True, pin_memory=False,
                 num_workers=1, random_seed=42, val_database_path=None, sync_dataset_paths=None, *args, **kwargs):
        super(SimulationDmodule, self).__init__()

        self.database_path = database_path
        self.val_database_path = val_database_path if val_database_path is not None else database_path
        print('PATHS', self.database_path, self.val_database_path)

        self.sync_dataset_paths = sync_dataset_paths

        if val_database_path is None:
            dset = self.get_dset(database_path=self.database_path, *args, **kwargs)
            indices = list(range(len(dset)))
            split = int(np.floor(val_frac * len(dset)))
            if shuffle_dataset:
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            self.train_indices, self.val_indices = indices[split:], indices[:split]

        else:
            dset = self.get_dset(database_path=self.database_path, *args, **kwargs)
            self.train_indices = list(range(len(dset)))

            dset = self.get_dset(database_path=self.val_database_path, *args, **kwargs)
            self.val_indices = list(range(len(dset)))

        if sync_dataset_paths is not None:
            sync_dset = SyncDset(sync_dataset_paths, meta_info_vars=kwargs['meta_info'], load=False)
            indices = list(range(len(sync_dset)))
            split = int(np.floor(val_frac * len(sync_dset)))
            if shuffle_dataset:
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            self.sync_train_indices, self.sync_val_indices = indices[split:], indices[:split]

        else:
            self.sync_train_indices, self.sync_val_indices = None, None

        self.batch_size = batch_size
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.dset_kwargs = kwargs

    def get_dloader(self, shuffle=False, mode='train', **kwargs):
        if mode == 'train':
            dset = self.trn_dset

        elif mode == 'val':
            dset = self.val_dset

        else:
            raise NotImplementedError()

        sampler_kwargs = dict(batch_size=self.batch_size,
                              shuffle=shuffle)

        other_kwargs = dict(persistent_workers=self.persistent_workers and self.num_workers > 0,
                            pin_memory=self.pin_memory,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn,)

        dl = DataLoader(dset, **sampler_kwargs, **other_kwargs)

        return dl

    def setup(self, stage='fit'):
        if stage in ("val", "fit") or stage is None:
            ## VALIDATION ##
            limit_patches = None
            self.val_dset = self.get_dset(limit_patches=limit_patches,
                                          **self.dset_kwargs, **self.val_data)

            self.val_dset.setup()

        if stage == "fit" or stage is None:
            ## TRAIN ##
            limit_patches = None

            self.trn_dset = self.get_dset(limit_patches=limit_patches,
                                          **self.dset_kwargs, **self.trn_data)

            self.trn_dset.setup()

        if stage == "test" or stage is None:
            raise NotImplementedError()

    def get_dset(self, *args, sync_dataset_paths=None, meta_info=None, sync_indices=None, **kwargs):
        if sync_indices is None:
            return SimulationDset(*args, meta_info=meta_info, **kwargs)

        else:
            return DatasetMerger([SimulationDset(*args, meta_info=meta_info, **kwargs),
                                  SyncDset(dataset_paths=sync_dataset_paths,
                                           meta_info_vars=meta_info, indices=sync_indices)])

    def prepare_data(self):
        self.trn_data, self.val_data, self.tst_data = self.assemble_data()

    def assemble_data(self, *args, **kwargs):
        return (dict(indices=self.train_indices, database_path=self.database_path,
                     sync_dataset_paths=self.sync_dataset_paths, sync_indices=self.sync_train_indices),
                dict(indices=self.val_indices, database_path=self.val_database_path,
                     sync_dataset_paths=self.sync_dataset_paths, sync_indices=self.sync_val_indices),
                dict([]))

    @staticmethod 
    def collate_fn(*args, **kwargs):
        return default_collate(*args, **kwargs)

    def train_dataloader(self, shuffle=True, **kwargs):
        return self.get_dloader(mode='train', shuffle=True, **kwargs)

    def val_dataloader(self, **kwargs):
        return self.get_dloader(mode='val', shuffle=False, **kwargs)

    def test_dataloader(self, **kwargs):
        return self.get_dloader(mode='test', shuffle=False, **kwargs)

    @classmethod
    def add_argparse_args(cls, parser, *args, **kwargs):
        parser = SimulationDset.add_argparse_args(parser, *args, **kwargs)
        parser.add_argument('--database_path', type=str, nargs="*", default=None)
        parser.add_argument('--val_database_path', type=str, nargs="*", default=None)

        parser.add_argument('--sync_dataset_paths', type=str, nargs="*", default=None)

        parser.add_argument('--num_workers', default=1, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--pin_memory', default=0, type=int)
        parser.add_argument('--persistent_workers', default=0, type=int)
        parser.add_argument('--val_frac', default=0.5, type=float)
        parser.add_argument('--shuffle_dataset', default=0, type=int)
        return parser


class DatasetMerger(Dataset):
    def __init__(self, dsets):
        super(DatasetMerger, self).__init__()
        self.dsets = dsets
        self.len_per_dset = [len(dset) for dset in self.dsets]
        self.cumsum = np.r_[0, np.cumsum(self.len_per_dset)]

    def __len__(self):
        return np.sum(self.len_per_dset)

    def __getitem__(self, item, **kwargs):
        dset_ind = np.where(item < self.cumsum)[0][0] - 1
        ind = item - self.cumsum[dset_ind]
        ret = self.dsets[dset_ind].__getitem__(ind, **kwargs)
        
        return ret

    def setup(self):
        for dset in self.dsets:
            dset.setup()

    def prepare_data(self):
        for dset in self.dsets:
            dset.prepare_data()

