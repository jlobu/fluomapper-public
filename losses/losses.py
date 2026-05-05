from fluomapper.utils.run import add_model_specific_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def hyplant_calibr_noise_variance(x):
    """
    Variance model of HyPlant FLUO at-sensor acquisitions

    :param x: FLUO at-sensor radiance in DNs
    :return:
    """
    # values for sigmoid
    m = 1 / 500
    dx = 10270

    # linear fits to variance
    y1 = (0.5262 * x - 67.79)
    y2 = (0.4898 * x + 1150)

    exp = torch.exp
    sigmoid = torch.nn.functional.sigmoid

    # blend between y1 and y2
    x_ = m * (x - dx)
    y = y1 * (1 - sigmoid(x_)) + y2 * sigmoid(x_)

    # add peak
    _gaussian = lambda x, m, s: exp(-(x - m) ** 2 / s ** 2)
    y = y + 2375.5 * _gaussian(x, dx, 870) + 205 * _gaussian(x, 11920, 670)

    return y


def var_weighted_mse(ypred, y):

    with torch.no_grad():
        weights = 1 - torch.softmax(hyplant_calibr_noise_variance(y * 105) / 1e4, dim=-1)

    return ((ypred - y) ** 2 * weights).mean()


class EvidentialLoss(nn.Module):
    def __init__(self, reg_weight=1, reg_type=None):
        super(EvidentialLoss, self).__init__()
        self.reg_weight = reg_weight
        self.reg_type = reg_type
        self.register_buffer("machine_epsilon", torch.tensor(np.finfo(np.float32).eps).requires_grad_(False))

    def forward(self, inputs, targets, model=None):
        targets = targets.squeeze()
        gamma = inputs[:, 0]
        alpha = inputs[:, 1]
        beta = inputs[:, 2]
        nu = inputs[:, 3]

        alpha = torch.max(1 + self.machine_epsilon, alpha)
        beta = torch.max(self.machine_epsilon, beta)
        nu = torch.max(self.machine_epsilon, nu)
        omega = 2 * beta * (1 + nu)

        nll = 0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * torch.log(nu) \
              - alpha * torch.log(omega) \
              + (alpha + 0.5) * torch.log((targets - gamma) ** 2 * nu + omega) \
              + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        # nll = - torch.distributions.StudentT(2 * alpha, gamma, (omega / (2 * nu * alpha)) ** 0.5 ).log_prob(targets)

        if self.reg_type is None or self.reg_type == 'normal':
            err = torch.abs(targets - gamma)
            reg = err * (2 * nu + alpha)
        
        elif self.reg_type == 'softmax':
            err = F.softmax(torch.abs(targets - gamma))
            reg = err * (2 * nu + alpha)

        else:
            raise NotImplementedError

        if model is not None:
            model.log('median_aleatoric_uncertainty', (beta/(alpha - 1)).median(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            model.log('median_epistemic_uncertainty', (beta / (alpha - 1) / nu).median(), prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)

            model.log('nll', nll.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            model.log('reg', reg.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            model.log('err', err.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return (nll + self.reg_weight * reg).mean()

    @classmethod
    def add_model_specific_args(cls, parser, prepend='', **kwargs):
        parser_spec = dict([
            ('reg_weight', dict(type=float, default=1)),
            ('reg_type', dict(type=str, default=None))
        ])
        parser = add_model_specific_args(parser, parser_spec=parser_spec, prepend=prepend, ignore_overrides=False, **kwargs)
        return parser


def jensen_shannon_diff(true, pred):
    loss = F.kl_div(torch.log2(true), torch.log2(0.5 * (pred + true)), log_target=True)
    loss += F.kl_div(torch.log2(pred), torch.log2(0.5 * (pred + true)), log_target=True)

    return 0.5 * loss


def mutual_information(p_xy):

    p_x = torch.sum(p_xy, dim=1)
    p_x = torch.einsum('i..., i -> i...', p_x, 1/p_x.sum(dim=-1))
    p_x = p_x.unsqueeze(1)

    p_y = torch.sum(p_xy, dim=2)
    p_y = torch.einsum('i..., i -> i...', p_y, 1/p_y.sum(dim=-1))
    p_y = p_y.unsqueeze(2)

    logp = torch.log2(p_xy) - torch.log2(p_x) - torch.log2(p_y)
    mutual = p_xy * logp
    
    out = torch.nansum(mutual)
    if torch.isnan(out):
        return 0

    return out


def conditional_entropy(p_xy, given=0):
    sum_over = (given + 1) % 2
    p_xy += 1e-40
    log_p_y = torch.log(p_xy.sum(dim=sum_over) / p_xy.sum(dim=sum_over).sum()).unsqueeze(sum_over)
    log_p_xy = torch.log(p_xy)

    out = -torch.nansum(p_xy * (log_p_xy - log_p_y))
    
    return out


def cos_dist_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return 1 - sim_mt 


def contrastive_loss(x_features, labels, dist_mode='l2'):
    inf = torch.diag(torch.ones(x_features.shape[0], device=x_features.device) * torch.inf)
    
    if dist_mode.startswith('l2'):
        pdist = torch.cdist(x_features.unsqueeze(0), x_features.unsqueeze(0)).squeeze()

    elif dist_mode.startswith('cos'):
        pdist = cos_dist_matrix(x_features, x_features)

    else:
        raise Exception(f'Unknown dist_mode {dist_mode}')
    
    reg_ = None
    for label in labels:
        label = torch.atleast_2d(label)

        pdist_label = torch.cdist(label[None, :], label[None, :]).squeeze()
        pdist_label_min = pdist_label + inf
        pdist_label_max = pdist_label - inf
        
        if not dist_mode.endswith('cor'):
            min_labels = torch.argmin(pdist_label_min, dim=1).squeeze()
            max_labels = torch.argmax(pdist_label_max, dim=1).squeeze()
        
        if dist_mode == "l2":
            reg = (pdist[(torch.arange(len(min_labels)), min_labels)]/pdist.mean(dim=1)).sum() + \
                  (pdist[(torch.arange(len(max_labels)), max_labels)]/pdist.mean(dim=1)).sum() ** (-1)

            reg /= x_features.shape[0]

        elif dist_mode == 'cos':
            reg = (pdist[(torch.arange(len(min_labels)), min_labels)]/pdist.mean(dim=1)).sum() \
                    + ((1 - pdist[(torch.arange(len(max_labels)), max_labels)])/pdist.mean(dim=1)).sum()

            reg /= x_features.shape[0]

        elif dist_mode.endswith('cor'):
            reg = 1 - cos_dist_matrix(torch.triu(pdist, diagonal=1).flatten().unsqueeze(0), 
                                      torch.triu(pdist_label, diagonal=1).flatten().unsqueeze(0)).squeeze()

    return reg

