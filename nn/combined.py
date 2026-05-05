import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from fluomapper.utils.lightning_base import _Module
from fluomapper.utils.nn import InputNorm, ScaledSigmoid, ApplyOnImage
from fluomapper.utils.gaussian import GaussianSmoothing
from fluomapper.utils.func.interp1d import Interpolate
from fluomapper.utils.run import split_kwargs, _add, _remove, make_iterable
from fluomapper.utils.data import select, to_ffs_names

from ui import get_hparams, get_model, update_namespace

from fluomapper.nn.SFMNN.SFMNN import SFMNN
from fluomapper.nn.FWDNN.FWDNN import FWDNN
from fluomapper.nn.simulation_mlp.mlp import _MLP

import numpy as np
import copy

from types import SimpleNamespace


def _get_model(run_id, mlflow_backend_uri, hparams=None, freeze=True, load_ckpt='best'):
    reload_args = dict(reload_ckpt=load_ckpt, run_id=run_id, mlflow_backend_uri=mlflow_backend_uri)
    
    hparams_orig = get_hparams(run_id, mlflow_backend_uri)
    if hparams is not None:
        hparams = update_namespace(hparams_orig, hparams, ignore_none=True)
        #hparams['model'] = hparams_orig.model
        #if 'reload_args' in hparams:
        #    del hparams['reload_args']

        #hparams = SimpleNamespace(**hparams)

    else:
        hparams = hparams_orig

    net, _ = get_model(hparams=hparams, reload_args=SimpleNamespace(**reload_args))
    
    if freeze:
        net = net.requires_grad_(False).eval()

    return net


class _Combined(_Module):
    _PHYSICS_NN = None
    _RESIDUAL_NN = None

    # maps p_network to init_network
    __MAP__ = dict(lfluo760=('f', 1 / (np.exp(-.5 * (760 - 737) ** 2 / 20**2))),
                   rho740=('rho', 1))

    def __init__(self, init_network_run_id, init_network_mlflow_backend_uri, p_network_run_id=None, 
                 p_network_mlflow_backend_uri=None, freeze_p_network=False, allowed_res_range=(-0.1, 0.1),
                 res_range_params=('f',), meta_exclude_p_network=None, init_enc_to_res=False, p_enc_to_res=False,
                 pass_p_params_to_init=None, param_residual=True, inp_to_res=True, preds_to_res=True,
                 ndvi_to_res=False, with_residual_reg=False, residual_reg_weight=1e-2, residual_no_rho=False,
                 p_network_load='best', init_network_load='best', init_network_spectral_window=None,
                 with_res_ids=False, residual_no_sensor=False, res_clamp=True, off_nadir_to_res=False, 
                 id_to_res=False, pred_init_to_res=True, *args, **kwargs):
        
        new_kwargs = {}
        for key, val in kwargs.items():
            if key.startswith('override') and val is not None or not key.startswith('override'):
                new_kwargs[key] = val
        kwargs = new_kwargs

        super(_Combined, self).__init__(*args, **kwargs)
        
        self.p_network_is_reloaded = p_network_run_id is not None

        meta_exclude_p_network = [] if meta_exclude_p_network is not None else meta_exclude_p_network 
        self.p_meta_vars = copy.deepcopy(kwargs['meta_vars'])
        self.p_meta_info = copy.deepcopy(kwargs['meta_info'])
        _remove(self.p_meta_vars, meta_exclude_p_network)
        _remove(self.p_meta_info, meta_exclude_p_network)

        pkwargs = copy.deepcopy(kwargs)
        pkwargs['meta_vars'] = self.p_meta_vars
        pkwargs['meta_info'] = self.p_meta_info
        
        if not self.p_network_is_reloaded:
            pkwargs.update(split_kwargs(pkwargs, 'override_p', update_dic=False))
            self.p_network = self._PHYSICS_NN(*args, **pkwargs)
        
        else:
            if p_network_mlflow_backend_uri is None:
                p_network_mlflow_backend_uri = init_network_mlflow_backend_uri
            
            p_override = dict([(key, pkwargs[key]) for key in ['batch_shape', 'in_wvls', 
                                                              'out_wvls', 'data_source_ids', 
                                                              'data_sources', 'meta_vars', 
                                                              'meta_info', 'meta_vars_exclude']])
            p_override.update(split_kwargs(pkwargs, 'override_p', update_dic=False))
                        
            self.freeze_p_network = freeze_p_network
            self.p_network = _get_model(p_network_run_id, p_network_mlflow_backend_uri, 
                                        hparams=p_override, 
                                        freeze=freeze_p_network,
                                        load_ckpt=p_network_load)

        self.p_network.set_up_discr()
        self.automatic_optimization = self.p_network.automatic_optimization

        self.init_network = _get_model(init_network_run_id, init_network_mlflow_backend_uri, load_ckpt=init_network_load)
        self.init_network.return_enc = init_enc_to_res

        self.init_enc_to_res = init_enc_to_res
        self.p_enc_to_res = p_enc_to_res
        self.inp_to_res = inp_to_res
        self.preds_to_res = preds_to_res
        self.ndvi_to_res = ndvi_to_res
        self.off_nadir_to_res = off_nadir_to_res
        self.id_to_res = id_to_res
        self.pred_init_to_res = pred_init_to_res

        self.mapping = dict([(var, self.__MAP__[var][0]) if var in self.__MAP__ else (var, var)
                             for var in self.init_network.label])
        self.mapping_norms = dict([(var, self.__MAP__[var][1]) if var in self.__MAP__ else (var, 1)
                                   for var in self.init_network.label])
        self.inv_mapping = dict([(key, ikey) for ikey, key in self.mapping.items()])
        self.init_labels = [key for ikey, key in self.mapping.items()]

        self.res_range_params = res_range_params
        self.pass_p_params_to_init = pass_p_params_to_init if pass_p_params_to_init is not None else []

        assert not self.init_enc_to_res or (self.init_enc_to_res and hasattr(self.init_network, 'enc_dim'))

        # add observation spectral window so we can pass the full spectrum
        # the windowing is done internally to init_network
        if init_network_spectral_window is None:
            self.init_network.model_spectral_window_obs = self.p_network.pred_window
        else:
            init_network_spectral_window = [(init_network_spectral_window[i], init_network_spectral_window[i+1])
                                            for i in range(0, len(init_network_spectral_window), 2)]
            self.init_network.model_spectral_window_obs = init_network_spectral_window

        self.init_network_on_image = ApplyOnImage(self.init_network)

        # check if init_network and p_network fit together
        # # ensure allowed_res_range and res_range_params have the same dimensionality of parameters
        assert len(allowed_res_range) % 2 == 0
        assert len(allowed_res_range) // 2 == len(res_range_params)

        # # ensure the targeted labels correspond to emulator input parameters
        assert set(self.p_network.model.predicted_fwd_inputs).issuperset(set(self.res_range_params)), \
                    f'You want to initialize {res_range_params}, but p_network predicts {self.p_network.model.predicted_fwd_inputs}'

        # # ensure the p_network params for passing to init correspond to emulator input parameters
        # assert set(self.p_network.model.predicted_fwd_inputs).issuperset(set(self.pass_p_params_to_init))

        # # ensure there is a mapping between the pred_p variable name and the init_network label
        assert set(self.init_labels).issuperset(set(self.res_range_params))

        # # ensure there pass_p_params_to_init params correspond to an init input
        assert set(self.init_network.input_stack_order).issuperset(set(self.pass_p_params_to_init))
        
        self.param_residual = param_residual
        self.with_residual_reg = with_residual_reg
        self.residual_reg_weight = residual_reg_weight

        self.with_res_ids = with_res_ids

        if self.param_residual:
            self.res_dim_out = dict([(p, self.p_network.model.param_dim_out[p])
                                       if p in self.p_network.model.param_dim_out else (p, 1) 
                                     for p in self.res_range_params])
            if 'CW' in self.res_range_params:
                self.res_dim_out['CW'] = self.p_network.model.fwd.dim \
                                if not self.p_network.model.constant_cw_shift else 1

            if 'fwhm' in self.res_range_params:
                self.res_dim_out['fwhm'] = self.p_network.model.fwd.dim \
                                if not self.p_network.model.constant_fwhm_shift else 1

            dim_out = np.sum([val for p, val in self.res_dim_out.items()])

            # p_network_input + p_network output + init_network output
            dim_in = self.p_network.model.dim_in if inp_to_res else 0
            dim_in += self.p_network.model.enc_dim_out_orig if p_enc_to_res else 0

            if self.p_network.model.with_ids and self.p_network.model.ids_mode in self.p_network.model.IDS_MODE_TO_ENC and not self.with_res_ids:
                dim_in -= self.p_network.model.id_size
            
            self.p_preds_to_res = self.p_network.model.predicted_fwd_inputs
            if residual_no_rho:
                self.p_preds_to_res = [var for var in self.p_preds_to_res if var not in ['rho', 'rho_slope', 'e']]

            if residual_no_sensor:
                self.p_preds_to_res = [var for var in self.p_preds_to_res if var not in ['CW', 'fwhm']]

            p_preds_len = len(self.p_preds_to_res) 
            for var, const in dict(CW=self.p_network.model.constant_cw_shift,   
                                   fwhm=self.p_network.model.constant_fwhm_shift).items():
                if var in self.p_preds_to_res and not const:
                    p_preds_len += self.p_network.model.fwd.dim - 1
            
            p_preds_len += len(self.init_labels) if self.pred_init_to_res else 0

            dim_in += p_preds_len if preds_to_res else 0
            dim_in += self.init_network.enc_dim if init_enc_to_res else 0
            dim_in += 1 if ndvi_to_res else 0

            dim_in += self.off_nadir_to_res
            dim_in += self.p_network.model.id_size if self.id_to_res else 0

            residual_kwargs = split_kwargs(kwargs, 'residual')
            residual_kwargs['out_nonlin'] = 'none'
            self.residual_nn = ApplyOnImage(nn.Sequential(InputNorm(dim_in, windows=False),
                                                          self._RESIDUAL_NN(dim_in=dim_in, dim_out=dim_out, **residual_kwargs)
                                                         )
                                           )

            allowed_res_range = [(allowed_res_range[i], allowed_res_range[i+1])
                                 for i in range(0, len(allowed_res_range), 2)]
            if type(allowed_res_range) is tuple:
                allowed_res_range = [allowed_res_range] * dim_out

            self.scalers = nn.ModuleDict([(res_range_params[i], ScaledSigmoid(*range_))
                                         for i, range_ in enumerate(allowed_res_range)])
           
            get_range = lambda var: self.p_network.model.to_physical_scale({var: torch.tensor(self.p_network.model.allowed_fwd_range)},
                                                                            extrapolate=True)[var].tolist()
            self.res_lims = dict([(var, get_range(var)) for var in self.res_range_params])

            self.res_clamp = res_clamp

            self._optimizer_prefixes = self.p_network._optimizer_prefixes

    def prepare_batch(self, batch):
        return batch, None   #  self.p_network.prepare_batch(*args, **kwargs)
    
    def on_train_epoch_start(self):
        super(_Module, self).on_train_epoch_start() 

        self.init_network = self.init_network.requires_grad_(False).eval()

        if self.p_network_is_reloaded:
            if self.freeze_p_network:
                self.p_network = self.p_network.requires_grad_(False).eval()

            else:
                self.p_network.set_inp_norm_eval(force=True)

    def _rename(self, data):
        return self.p_model._rename(data)

    def _prepare_init_output(self, out_init):
        if type(out_init) is dict:
            pred_init, enc_init = out_init['pred'], out_init['enc']
        else:
            pred_init, enc_init = out_init, None
        
        # map labels to output
        pred_init = dict([(label, pred_init[:, [i]]) for i, label in enumerate(self.init_network.label)])
        return pred_init, enc_init

    def interpolate(self, obs, cw):
        #obs[:, self.init_network.model_spectral_window_obs] = Interp1D()(self.
        pass

    def forward(self, batch):
        # predict p_network
        x_p, y_p, meta_p = self.p_network.prepare_batch(batch)
        _pred_p = self.p_network(x_p, **meta_p)
        pred_p, enc_p = _pred_p['pred'], _pred_p['enc_']

        #if there are ids and not with_res_ids, remove ids from input
        #if self.p_network.model.with_ids and self.p_network.model.ids_mode in self.p_network.model.IDS_MODE_TO_ENC and not self.with_res_ids:
        #    x_p = x_p[:, :-self.p_network.model.id_size]

        for key in self.pass_p_params_to_init:
            batch[0][key] = pred_p[key]
        
       # if self.interpolate_init_inp:
       #     batch[0]['obs'] = self.interpolate(batch[0]['obs'], pred_p['CW'])

        # Predict with the trained init net
        x_init, y_init, meta_init = self.init_network.prepare_batch(batch, do_flatten=False)
        
        # remove source_id from kwargs as it doesn't have image shape
        # ApplyOnImage in self.init_network_image will fail otherwise
        source_id = meta_init['source_id']
        del meta_init['source_id']

        out_init = self.init_network_on_image(x_init.float())
        pred_init, enc_init = self._prepare_init_output(out_init)
        
        #print('CW_SHAPE pre', pred_p['CW'].shape)
        if self.param_residual:
            res = self.estimate_residual(x_p, pred_p, x_init, pred_init, enc_init=enc_init, enc_p=enc_p, 
                                         **meta_p)
            clamp_ = torch.clamp if self.res_clamp else lambda x, *args, **kwargs: x
            expand = lambda arr, shape: arr.expand(-1, shape[1], -1, -1) if arr.shape[1] == 1 else arr

            update = dict([(self.mapping[var], clamp_(self.mapping_norms[var] * expand(pred_init[var], val.shape) + val,
                                                           min=self.res_lims[self.mapping[var]][0], 
                                                           max=self.res_lims[self.mapping[var]][1])) 
                           for var, val in res.items()])

            pred_p.update(update)
            pred_p.update(dict([(f'res_{var}', val) for var, val in res.items()]))
            pred_p.update(dict([(f'init_{var}', pred_init[var]) for var in res.keys()]))

        else:
            pred_p.update(dict([(self.mapping[key], val) for key, val in pred_init.items() 
                                 if self.mapping[key] in self.res_range_params]))
        
        #print('CW_SHAPE post', pred_p['CW'].shape)

        # allow for extrapolation clamping
        pred_p.update(pred_fwd_scale=self.p_network.model.to_fwd_scale(pred_p))
        pred_p.update(self.p_network.model.to_physical_scale(pred_p['pred_fwd_scale']))
        
        _pred_p.update(pred=pred_p, enc=enc_p, pred_init=pred_init) 

        return _pred_p, x_p, y_p, meta_p

    def estimate_residual(self, x_p, pred_p, x_init, pred_init):
        return None

    def simulate_ats(self, *args, **kwargs):
        return self.p_network.model.simulate_ats(pred)

    def loss(self, pred, optimizer_idx=0, *args, **kwargs):
        loss, f_ats, bg_ats, y_ats, y, mask = self.p_network.loss(pred, optimizer_idx=optimizer_idx, *args, **kwargs)

        if self.with_residual_reg and optimizer_idx == 0:
            for resvar in [key for key in pred if key.startswith('res_')]:
                varname = resvar[len('res_'):]
                
                var = self.p_network.model.to_fwd_scale(dict([(varname, pred[varname])]))[varname]
                initvar = self.p_network.model.to_fwd_scale(dict([(varname, pred[f'init_{varname}'])]))[varname]

                loss += ((var - initvar) ** 2).mean() * self.residual_reg_weight

        elif optimizer_idx != 0:
            f_ats, bg_ats, y_ats = [None] * 3

        return loss, f_ats, bg_ats, y_ats, y, mask 

    def training_step(self, batch, batch_idx, *args, **kwargs):
        ypred, inp, y, meta = self.forward(batch)

        for optimizer_idx, opt in enumerate(make_iterable(self.optimizers())):
            loss, f_ats, bg_ats, y_ats, y, mask = self.loss(y=y, model=self.p_network, inp=inp,
                                                            mode='val', optimizer_idx=optimizer_idx, **meta, **ypred)
            loss = loss.to(self.device)

            # LOGGING ######################################################
            if optimizer_idx == 0:
                self.log('mse_loss', F.mse_loss(y, y_ats).cpu().detach(),
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log('loss', loss.cpu().detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

                if self.p_network.with_label_reg:
                    label_reg = self.p_network.label_reg(**ypred, **meta)
                    self.log('label_reg', label_reg.cpu().detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

            else:
                self.log('loss_' + self._optimizer_prefixes[optimizer_idx], loss.cpu().detach(),
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # BACKPROP IN MANUAL MODE ######################################################
            if not self.automatic_optimization:
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs):
        ypred, inp, y, meta = self.forward(batch)

        loss, f_ats, bg_ats, y_ats, y, mask = self.loss(y=y, model=self.p_network, inp=inp, mode='val', optimizer_idx=optimizer_idx, **meta, **ypred)
        logs = dict(val_loss_step=loss.to(self.device).cpu().detach())
        
        if optimizer_idx == 0:
            val_overall_mse_step = self.p_network._mean_reduce((y - y_ats).mean(1) ** 2, mask=mask).to(self.device)
            logs.update(val_overall_mse_step=val_overall_mse_step)

            if self.p_network.sif_focus_window is not None:
                val_sif_focus_step = self.p_network._mean_reduce(((select(signals=y_ats, windows=self.p_network.sif_focus_window, axis=1) -
                                                select(signals=y, windows=self.p_network.sif_focus_window, axis=1)) ** 2).mean(1), mask=mask).to(self.device)
                logs.update(val_sif_focus_step=val_sif_focus_step)

            if self.p_network.with_label_reg:
                label_reg = self.p_network.label_reg(**ypred, **meta)
                logs.update(label_reg=label_reg)

        self.VAL_LOGS.append(logs)

    def on_validation_epoch_end(self):
        outputs = self.VAL_LOGS

        if self.val_logger is not None:
            _ = self.val_logger.log_validation_epoch_end(outputs=outputs, model=self)
        
        for key in outputs[0].keys():
            avg_loss = torch.stack([x[key] for x in outputs]).mean()
            self.log(key.replace('_step', ''), avg_loss.item(), sync_dist=True)

        self.VAL_LOGS.clear()


    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        parser = cls._RESIDUAL_NN.add_model_specific_args(parser, 'residual', **kwargs)
        parser = cls._PHYSICS_NN.add_model_specific_args(parser, **kwargs)
        parser = cls._PHYSICS_NN.add_model_specific_args(parser, 'override_p', default_none=True, **kwargs)
        
        parser.add_argument('--init_network_run_id', type=str, required=True)
        parser.add_argument('--init_network_mlflow_backend_uri', type=str, required=True)
        parser.add_argument('--init_network_load', type=str, default='best')
        parser.add_argument('--init_network_spectral_window', type=int, default=None, nargs='*')

        parser.add_argument('--p_network_run_id', type=str, required=False)
        parser.add_argument('--p_network_load', type=str, default='best')
        parser.add_argument('--p_network_mlflow_backend_uri', type=str, required=False)
        parser.add_argument('--freeze_p_network', type=int, default=False)

        parser.add_argument('--param_residual', type=int, default=True)

        parser.add_argument('--allowed_res_range', type=float, default=(-0.1, 0.1), nargs='*')
        parser.add_argument('--res_range_params', type=str, default=['f'], nargs='*')
        parser.add_argument('--pass_p_params_to_init', type=str, nargs="*")

        parser.add_argument('--meta_exclude_p_network', type=str, default=None, nargs="*")
        parser.add_argument('--meta_exclude_init_network', type=str, default=None, nargs="*")
        
        parser.add_argument('--init_enc_to_res', type=int, default=False)
        parser.add_argument('--p_enc_to_res', type=int, default=False)

        parser.add_argument('--inp_to_res', type=int, default=True)
        parser.add_argument('--preds_to_res', type=int, default=True)
        parser.add_argument('--residual_no_rho', type=int, default=False)
        parser.add_argument('--residual_no_sensor', type=int, default=False)
        parser.add_argument('--ndvi_to_res', type=int, default=False)
        parser.add_argument('--off_nadir_to_res', type=int, default=False)
        parser.add_argument('--id_to_res', type=int, default=False)
        parser.add_argument('--pred_init_to_res', type=int, default=True)

        parser.add_argument('--with_residual_reg', type=int, default=False)
        parser.add_argument('--residual_reg_weight', type=float, default=1e-2)

        parser.add_argument('--res_clamp', type=int, default=True)

        return parser


class CombinedFWDNN(_Combined):
    _PHYSICS_NN = FWDNN
    _RESIDUAL_NN = _MLP

    def estimate_residual(self, x_p, pred_p, x_init, pred_init, enc_init, enc_p, **kwargs):
        pred_p_cat = torch.cat([pred_p['pred_fwd_scale'][var_]
                                for var_ in self.p_preds_to_res], axis=1)

        if self.pred_init_to_res:
            pred_init_cat = torch.cat([pred_init[var_] for var_ in self.init_network.label], axis=1)
            preds_ = torch.cat([pred_p_cat, pred_init_cat], axis=1)

        else:
            preds_ = pred_p_cat
        
        if self.inp_to_res and self.preds_to_res:
            res_inp = torch.cat([x_p, preds_], axis=1)

        elif self.inp_to_res and not self.preds_to_res:
            res_inp = x_p
        
        elif self.preds_to_res:
            res_inp = preds_

        else:
            res_inp = None

        if self.init_enc_to_res and res_inp is not None:
            res_inp = torch.cat([res_inp, enc_init], axis=1)

        elif self.init_enc_to_res and res_inp is None:
            res_inp = enc_init

        if self.p_enc_to_res and res_inp is not None:
            res_inp = torch.cat([res_inp, enc_p], axis=1)

        elif self.p_enc_to_res and res_inp is None:
            res_inp = enc_p

        if res_inp is None:
            raise Exception("You have configured no input ")

        if self.ndvi_to_res:
            res_inp = torch.cat([res_inp, kwargs['ndvi']], axis=1)
        
        if self.off_nadir_to_res:
            res_inp = torch.cat([res_inp, kwargs['off_nadir']], axis=1)

        if self.id_to_res:
            ids = self.p_network.model.get_ids(x=res_inp, **kwargs)#.clone().detach()
            res_inp = torch.cat([res_inp, ids], axis=1)

        res_pred = self.residual_nn(res_inp)
        res_pred = self.p_network.model.to_dict(res_pred, self.res_range_params, self.res_dim_out)
        res_pred = dict([(key, self.scalers[key](val)) for key, val in res_pred.items()])

        #res_pred = dict([(self.inv_mapping[var], self.scalers[var](res_pred[:, [i]]))
        #                 for i, var in enumerate(self.res_range_params)])

        return res_pred

