import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from fluomapper.utils.lightning_base import _Module
from fluomapper.utils.nn import lin_nonlin_bn, InputNorm
from fluomapper.utils.run import add_model_specific_args, eval_with_valid_kwargs, init_with_valid_kwargs
from fluomapper.utils.data import permute_channels, search_spectral_window, select, to_ffs_names

from fluomapper.losses.losses import contrastive_loss 

from fluomapper.config import get_avail_heads, get_avail_losses, NoneAction
import pytorch_lightning as pl


class _MLP(nn.Module):
    _avail_heads = get_avail_heads()

    def __init__(self, dim_in, dims, dim_out=1, dropout=0, rep=1, residual=False, return_hidden=False, nonlin='relu',
                 bayesian=False, out_bn=False, attention=0, attention_rep=1, attention_bn=False,
                 in_bn=False, in_attention=False, in_attention_rep=1, head=None, out_nonlin='relu', 
                 out_dropout=None, *args, **kwargs):
        super(_MLP, self).__init__()

        self.dims = dims
        
        if in_bn:
            in_bn = nn.BatchNorm1d(dim_in)

        else:
            in_bn = nn.Identity()

        if dim_in is not None:
            self.in_lin = nn.Sequential(in_bn, lin_nonlin_bn(dim_in=dim_in, dim_out=dims[0], nonlin=nonlin, bayesian=bayesian,
                                                             attention=in_attention, attention_rep=in_attention_rep))

        else:
            self.in_lin = None
        
        self.head = head
        if self.head is not None:
            model_dim_out = None
            head_dim_out = dim_out

        else:
            model_dim_out = dim_out
            head_dim_out = None

        # CREATE head and model, if there is a head, don't put an out layer in _MLP
        if self.head is not None:
            if type(self.head) is str and self.head != 'none':
                self.head = get_avail_heads()[self.head](dim_in=dims[-1], dim_out=head_dim_out, *args, **kwargs)

            elif self.head == 'none':
                self.head = None

        dropout = self._make_list(dropout)
        attention = self._make_list(attention)
        attention_rep = self._make_list(attention_rep)
        rep = self._make_list(rep)

        self.hidden = nn.Sequential(*[lin_nonlin_bn(dim_in=dim_in, dim_out=dim_out, dropout=dropout[i], nonlin=nonlin,
                                                    rep=rep[i], residual=residual, attention=attention[i],
                                                    attention_rep=attention_rep[i],
                                                    attention_bn=attention_bn, bayesian=bayesian, *args, **kwargs)
                                      for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:]))])

        if len(self.hidden) == 0:
            self.hidden = nn.Identity()

        self.return_hidden = return_hidden
        if not self.return_hidden and self.head is None:
            out_dropout = dropout[-1] if out_dropout is None else out_dropout
            self.out = lin_nonlin_bn(dim_in=dims[-1], dim_out=model_dim_out, dropout=out_dropout, bn=out_bn,
                                     bayesian=bayesian, nonlin=out_nonlin)

    def _make_list(self, inp):
        # define inp per block
        if type(inp) is tuple:
            inp = list(inp)

        if not type(inp) in (list, tuple):
            inp = [inp] * len(self.dims)

        if len(inp) != len(self.dims):
            inp += (len(self.dims) - len(inp)) * [inp[-1]]

        return inp

    def forward(self, x, **kwargs):
        if self.in_lin is not None:
            x = self.in_lin(x)

        x = self.hidden(x)
        if self.return_hidden:
            return x

        if self.head is not None:
            return self.head(x, **kwargs)

        ypred = self.out(x)
        return ypred

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', *args, **kwargs):
        parser_spec = dict(
            [('dims', dict(nargs='+', type=int, default=[10, 20])),
             ('dropout', dict(type=float, default=0, nargs='*')),
             ('out_dropout', dict(type=float, default=None)),
             ('head', dict(type=str, default=None, choices=cls._avail_heads)),
             ('rep', dict(type=int, default=1, nargs='*')),
             ('rep_with_bn', dict(type=int, default=0)),
             ('out_bn', dict(type=int, default=0)),
             ('residual', dict(type=int, default=False)),
             ('attention', dict(type=int, default=0, nargs='*')),
             ('attention_rep', dict(type=int, default=1, nargs='*')),
             ('attention_bn', dict(type=int, default=1)),
             ('nonlin', dict(type=str, default='relu')),
             ('out_nonlin', dict(type=str, default='relu')),
             ('in_bn', dict(type=int, default=0)),
             ('bn_after_relu', dict(type=int, default=1))])
        parser = add_model_specific_args(parser, parser_spec=parser_spec, prepend=prepend, **kwargs)
        return parser


class MLP(_Module):

    def __init__(self, batch_shape=None, dim_in=None, dim_out=1, out_wvls=None, in_wvls=None,
                 model_spectral_window_obs_wvl=None, intermediate_layer=False, 
                 with_ranking_reg=False, with_contrastive_reg=False, contrastive_reg_weight=0.1,
                 ranking_reg_weight=0.1, contrastive_label=None, insert_first_layer=None, 
                 insert_first_layer_dims=None, insert_first_layer_rep=3, contrastive_dist_mode='l2', 
                 last_layer_nonlin='relu', last_layer_dropout=True, return_enc=False, *args, **kwargs):

        super(MLP, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        # TRAINING params
        self.out_wvls = out_wvls
        self.in_wvls = in_wvls

        self.return_enc = return_enc
        
        if 'label' in kwargs:
            self.label = kwargs['label']
            if type(self.label) is str:
                dim_out = 1
                self.label = [self.label]
            else:
                dim_out = len(kwargs['label'])

            self.dim_out = dim_out

        else:
            self.label = None

        kwargs['meta_info'] = [] if not 'meta_info' in kwargs or kwargs['meta_info'] is None else kwargs['meta_info']
        kwargs['meta_vars'] = [] if not 'meta_vars' in kwargs or kwargs['meta_vars'] is None else kwargs['meta_vars']
        kwargs['meta_info_exclude'] = [] if not 'meta_info_exclude' in kwargs or kwargs['meta_info_exclude'] is None else kwargs['meta_info_exclude']

        self.input_stack_order = list(kwargs['meta_info']) + list(kwargs['meta_vars'])
        self.input_stack_order = [k for k in self.input_stack_order if not k in kwargs['meta_info_exclude']]

        self.model_spectral_window_obs = model_spectral_window_obs_wvl
        if model_spectral_window_obs_wvl is not None:
            window = search_spectral_window(*model_spectral_window_obs_wvl, where=self.in_wvls[0])
            if len(window) > 0:
                self.model_spectral_window_obs = window

        orig_batch_shape = batch_shape
        sample_batch = (torch.zeros(batch_shape[0], device='cpu'), torch.zeros(batch_shape[1], device='cpu'))
        batch_shape = [out.shape for out in self.prepare_batch(sample_batch, strict=False)[:2]]

        if dim_in is None and batch_shape is not None:
            dim_in = batch_shape[0][1]

        meta_info_dims = 0

        if len(orig_batch_shape[0]) >= 4:
            if 'meta_info' in kwargs and kwargs['meta_info'] is not None and not 'none' in kwargs['meta_info']:
                meta_info_dims  += len(kwargs['meta_info']) * np.prod(orig_batch_shape[0][2:])

            if 'meta_vars' in kwargs and kwargs['meta_vars'] is not None:
                meta_info_dims  += len(kwargs['meta_vars']) * np.prod(orig_batch_shape[0][2:])

        else:
            if 'meta_info' in kwargs and kwargs['meta_info'] is not None and not 'none' in kwargs['meta_info']:
                meta_info_dims += len(kwargs['meta_info'])

            if 'meta_vars' in kwargs and kwargs['meta_vars'] is not None:
                meta_info_dims += len(kwargs['meta_vars'])

        if 'rho_full' in kwargs['meta_info']:
            #print(f'HERE adding {dim_in} + {batch_shape[0][1] - 1}')
            meta_info_dims += 8  #batch_shape[0][1] - 1
        
        reduce_vars_len = len(kwargs['meta_info_exclude']) if 'meta_info_exclude' in kwargs \
                            and kwargs['meta_info_exclude'] is not None else 0
        self.meta_info_dims = int(meta_info_dims) - reduce_vars_len
        
        self.insert_first_layer = insert_first_layer
        if insert_first_layer is not None:
            self.upscale = _MLP(dim_in=insert_first_layer[1] + self.meta_info_dims, dim_out=insert_first_layer[1],
                                dims=insert_first_layer_dims, rep=insert_first_layer_rep, residual=1, out_nonlin='none')

            self.upscale_norm = InputNorm(insert_first_layer[1] + self.meta_info_dims, windows=False)
            #nlayers = int(torch.log2(torch.ceil(torch.tensor(insert_first_layer[1]/insert_first_layer[0]))))
            #self.upscale = nn.Sequential(*[nn.Sequential(torch.nn.ConvTranspose1d(
            #                int( i==0) + int((i != 0) * 100), 
            #                int((i==nlayers) + 100 *(i!=nlayers)), 
            #                kernel_size=2, stride=2, padding=1), 
            #                    #nn.ReLU(),
            #                    #nn.BatchNorm1d(int((i==nlayers) + 100 *(i!=nlayers)))
            #                    )
            #    for i in range(nlayers + 1)])

            dim_in = insert_first_layer[1]

        else:
            self.upscale = None

        dim_in += self.meta_info_dims
        self.inp_norm = InputNorm(dim_in, windows=False)
        
        self.with_ranking_reg = with_ranking_reg
        self.ranking_reg_weight = ranking_reg_weight
        self.intermediate_layer = intermediate_layer

        self.contrastive_reg_weight = contrastive_reg_weight
        self.with_contrastive_reg = with_contrastive_reg
        self.contrastive_label = contrastive_label if not type(contrastive_label) is str else [contrastive_label]
        self.contrastive_dist_mode = contrastive_dist_mode

        if not self.intermediate_layer:
            kwargs['out_nonlin'] = last_layer_nonlin
            kwargs['out_dropout'] = last_layer_dropout

            self.model = _MLP(dim_in=dim_in, dim_out=dim_out,
                              bayesian=self.bayesian_mode == 'backprop', *args, **kwargs)

        else:
            dims_ = kwargs['dims']
            dims1 = dims_[:intermediate_layer]
            dims2 = dims_[intermediate_layer:]

            if type(kwargs['dropout']) is list:
                dropout = kwargs['dropout']
                dropout1 = dropout[:intermediate_layer]
                dropout2 = dropout[intermediate_layer:]

            else:
                dropout1 = kwargs['dropout']
                dropout2 = kwargs['dropout']

            if type(kwargs['dropout']) is list:
                rep = kwargs['rep']
                rep1 = rep[:intermediate_layer]
                rep2 = rep[intermediate_layer:]

            else:
                dropout1 = kwargs['rep']
                dropout2 = kwargs['rep']

            kwargs['dims'] = dims1
            kwargs['dropout'] = dropout1
            self.enc_dim = dims2[0]
            self.model1 = _MLP(dim_in=dim_in, dim_out=self.enc_dim,
                              bayesian=self.bayesian_mode == 'backprop', *args, **kwargs)

            if self.with_contrastive_reg and self.contrastive_dist_mode == 'cos':
                self.model1 = nn.Sequential(self.model1, nn.Sigmoid())

            kwargs['dims'] = dims2
            kwargs['dropout'] = dropout2
            kwargs['out_nonlin'] = last_layer_nonlin
            kwargs['out_dropout'] = last_layer_dropout
            self.model2 = _MLP(dim_in=self.enc_dim, dim_out=dim_out,
                              bayesian=self.bayesian_mode == 'backprop', *args, **kwargs)
            
    def _rename(self, data):
        return to_ffs_names(data)

    def prepare_batch(self, batch, do_flatten=True, strict=True):
        x, y = batch

        if type(x) is dict:
            meta_info = dict([(k, v) for k, v in x.items()
                              if k != 'obs'])
            x = x['obs']

        else:
            meta_info = {}

        if type(y) is dict:
            sif_meta_info = dict([(k, v) for k, v in y.items()
                                  if k != 'obs'])

            y = y['obs']

        else:
            sif_meta_info = {}

        if x is None:
            return None

        meta_info.update(sif_meta_info)
        meta_info = self._rename(meta_info)

        # if input is an image
        if len(x.shape) >= 4:
            wvl_dim = 1

        else:
            wvl_dim = -1

        if self.model_spectral_window_obs is not None:
            x = select(x, self.model_spectral_window_obs, axis=wvl_dim)

        if wvl_dim == 1 and y is not None and do_flatten:
            x = x.flatten(start_dim=-3)

            center_px = y.shape[-1] // 2
            y = y[..., center_px, center_px]
        
        if meta_info is not None:
            xs = [x]
        
            # keys = []
            for key in self.input_stack_order:
                if not key in meta_info and strict:
                   raise Exception(f'Variable {key} is missing in batch.')

                elif not key in meta_info and not strict:
                    continue
                
                var = meta_info[key]

                if wvl_dim == 1 and do_flatten:
                    var = var.flatten(start_dim=-3)

                xs.append(var)
               # keys.append(key)
            
            # print([val.shape for val in xs])
            # print('x', keys)
            x = torch.cat(xs, dim=wvl_dim)
      
        return x, y, meta_info

    def forward(self, x, *args, **kwargs):
        
        if self.upscale is not None:
            rad = nn.functional.interpolate(x[:, :-self.meta_info_dims].unsqueeze(1), 
                                            size=self.insert_first_layer[1], mode='linear').squeeze()

            meta = x[:, -self.meta_info_dims:]

            x = torch.cat([rad, meta], dim=1) # add meta_info part

            drad = self.upscale(self.upscale_norm(x))
            rad = rad + drad
            x = torch.cat([rad, meta], dim=1) # add meta_info part

        x_normed = self.inp_norm(x)

        if self.intermediate_layer is None:
            ypred = self.model.forward(x_normed)
            return ypred

        else:
            x_features = self.model1(x_normed)
            ypred = self.model2(x_features)

            if self.return_enc:
                return dict(pred=ypred, enc=x_features)

            return ypred

    def loss(self, ypred, y, *args, inp=None, mode='val', **kwargs):
        
        loss = None
        for i in range(y.shape[1]):
            l = super(MLP, self).loss(ypred[:, i], y[:, i])

            if loss is None:
                loss = l

            else:
                loss += l

            if mode == 'train':
                if self.label is not None:
                    label = self.label[i]

                else:
                    label = f'loss_{i}'

                self.log(label, l.mean().item(), 
                        prog_bar=True, logger=True, on_step=True, 
                        on_epoch=True)
 
        if self.with_contrastive_reg:
            x_features = self.model1(self.inp_norm(inp))
            if self.contrastive_label is None:
                labels = y

            else:
                labels = [kwargs[c] for c in self.contrastive_label]

            reg = contrastive_loss(x_features, labels, dist_mode=self.contrastive_dist_mode)

            if mode == 'train':
                self.log('contrastive_reg', reg.mean().item(), 
                        prog_bar=True, logger=True, on_step=True, 
                        on_epoch=True)

            loss += self.contrastive_reg_weight * reg 

        return loss

    def ranking_loss(self, x_features, labels):
        x_features = x_features[None]
        pdist = torch.cdist(x_features, x_features)
        pdist += torch.diag(torch.ones(pdist.shape[1], device=pdist.device) * torch.inf)[None]
        ranking = torch.argmin(pdist, dim=2).squeeze()
        
        pdist_labels = torch.cdist(labels[None, :], labels[None, :])
        pdist_labels += torch.diag(torch.ones(pdist_labels.shape[1], device=pdist_labels.device) * torch.inf)[None]
        ranking_labels = torch.argmin(pdist_labels, dim=2).squeeze()

        return ((ranking - ranking_labels) ** 2).sum()

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        parser = super(MLP, cls).add_model_specific_args(parser, prepend=prepend, **kwargs)
        parser = _MLP.add_model_specific_args(parser, prepend=prepend, **kwargs)

        parser.add_argument('--with_ranking_reg', type=int, default=0)
        parser.add_argument('--ranking_reg_weight', type=float, default=0.1)

        parser.add_argument('--intermediate_layer', type=int, default=None)

        parser.add_argument('--with_contrastive_reg', type=int, default=0)
        parser.add_argument('--contrastive_reg_weight', type=float, default=0.1)
        parser.add_argument('--contrastive_label', type=str, default=None, nargs='*')
        parser.add_argument('--contrastive_dist_mode', type=str, default='l2')

        parser.add_argument('--insert_first_layer', type=int, nargs=2, default=None)
        parser.add_argument('--insert_first_layer_rep', type=int, nargs=2, default=None)
        parser.add_argument('--insert_first_layer_dims', type=int, nargs='*', default=None)

        parser.add_argument('--last_layer_nonlin', type=str, default='relu')
        parser.add_argument('--last_layer_dropout', type=float, default=None)

        return parser

    def on_fit_start(self):
        pl.seed_everything(42)

