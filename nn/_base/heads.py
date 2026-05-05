import torch
import torch.nn as nn
import torch.nn.functional as F

from fluomapper.utils.func.cubic_spline import CubicSpline
from fluomapper.utils.nn import FixedMultiplier


class AbstractHead(nn.Module):
    def __init__(self, out_wvls=None, dim_in=None, dim_out=1, device=None, *args, **kwargs):
        super(AbstractHead, self).__init__()

        self.device = device
        self.dim_in = dim_in
        self.dim_out = dim_out

        if self.device is not None and ((type(self.device) is str and self.device == 'cpu') or self.device.type == 'cpu') \
                and torch.cuda.is_available():
            self.device = 'cuda:0'

        if out_wvls is not None:
            if len(out_wvls.shape) > 2:
                out_wvls = out_wvls[0].squeeze()
                assert len(out_wvls.shape) == 2
            
            self.out_wvls = nn.Parameter(out_wvls, requires_grad=False)

    def forward(self, *input, **kwargs):
        raise NotImplementedError('Define parametric model')


class WeightedSum(AbstractHead):
    def __init__(self, normed=True, to_sigmoid=False, with_amp=False, *args, **kwargs):
        super(WeightedSum, self).__init__(*args, **kwargs)

        self.normed = normed
        self.base_vectors = None
        
        self.to_sigmoid = to_sigmoid
        self.with_amp = with_amp

    def set_base_vectors(self, base_vectors, max=1, min=0):
        self.base_vectors = nn.Parameter(base_vectors, requires_grad=False).float()
        
        if self.to_sigmoid:
            n_weights = self.base_vectors.shape[0] + self.with_amp
            self.in_layer = nn.Sequential(nn.Linear(n_weights, n_weights), nn.BatchNorm1d(n_weights), nn.Sigmoid(),
                                          FixedMultiplier(max - min, min))

        if max is not None:
            assert self.normed
            self.base_vectors = nn.Parameter(torch.einsum('in, i -> in', self.base_vectors,
                                                          1 / (1e-3 + torch.max(self.base_vectors, dim=1)[0]),),
                                             requires_grad=False)

    def forward(self, x, eps=1e-3):
        
        if self.to_sigmoid:
            x = self.in_layer(x)

        weights = x[:, :self.base_vectors.shape[0]]

        if self.normed:
            weights = torch.einsum('bi, b -> bi', weights, 1 / (weights.sum(-1) + eps))
         
        if x.shape[1] == self.base_vectors.shape[0]:
            return torch.einsum('bi, in -> bn', weights, self.base_vectors)

        elif x.shape[1] == self.base_vectors.shape[0] + 1:
            assert self.with_amp
            amp = x[:, -1]
            return torch.einsum('bn, b -> bn', torch.einsum('bi, in -> bn', weights, self.base_vectors), amp)

        else:
            raise NotImplementedError


class PCAHead(AbstractHead):
    def __init__(self, components, mean, std, var_range=None, nr_fixed=5, variable_components=False,
                 with_mult=False, mult_bounds=None, *args, **kwargs):
        super(PCAHead, self).__init__(*args, **kwargs)

        if components is not None:
            self.components = nn.Parameter(components, requires_grad=variable_components)
        else:
            self.components = None

        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

        if var_range is not None:
            self.bounds = FixedMultiplier(var_range[1] - var_range[0], var_range[0])

        else:
            self.bounds = None

        self.with_mult = with_mult
        if mult_bounds and mult_bounds[1] - mult_bounds[0] > 0:
            self.mult_bounds = FixedMultiplier(mult_bounds[1] - mult_bounds[0], mult_bounds[0])

        else:
            self.mult_bounds = None

    def forward(self, x):
        
        with_mult = self.with_mult and self.mult_bounds is not None

        if with_mult:
            mult = x[:, [-1]]
            weights = x[:, :-1]
            
        else:
            mult = 1
            weights = x

        if with_mult:
            mult = self.mult_bounds(F.sigmoid(mult))

        if self.bounds is not None and self.components is not None:
            weights = self.bounds(F.sigmoid(weights))
        
        if self.components is not None:
            ret = mult * (self.mean + torch.einsum('ij, ni -> nj', self.components, weights) * self.std)
        else:
            ret = mult * self.mean

        return ret


class SplineHead(AbstractHead):
    def __init__(self, nr_fixed, **kwargs):
        super(SplineHead, self).__init__(**kwargs)

        xs = nn.Parameter(torch.linspace(self.out_wvls[0], self.out_wvls[-1], nr_fixed),
                          requires_grad=False)

        self.model = CubicSpline(xs, self.out_wvls)

    def forward(self, x):
        return self.model.forward(x)

class PolynomialHead(AbstractHead):
    def __init__(self, *args, **kwargs):
        super(PolynomialHead, self).__init__(*args, **kwargs)

        self.wvl_features = nn.Parameter(torch.stack([(self.out_wvls - self.out_wvls[self.out_wvls.shape[-1] // 2]) ** i
                                         for i in range(self.dim_in)], dim=-1), requires_grad=False)  # (wvls, degrees)

    def forward(self, x, **kwargs):
        features = torch.stack([x * wvl for wvl in self.wvl_features], dim=0).sum(-1).permute(1, 0)  # (batch, len(wvls))
        return features


class RSquareHead(PolynomialHead):
    def __init__(self, offset_min=0, offset_max=1, slope_min=0, slope_max=1, emin=0, emax=1, *args, **kwargs):
        super(RSquareHead, self).__init__(dim_in=3, *args, **kwargs)

        self.in_layer = nn.Sequential(nn.BatchNorm1d(3), nn.Sigmoid())
        self.offset_min = offset_min
        self.offset_max = offset_max

        self.slope_min = slope_min
        self.slope_max = slope_max

        self.emin = emin
        self.emax = emax

        self.lo = self.out_wvls[0]
        self.up = self.out_wvls[-1]

    def forward(self, x, return_params=False, *args, **kwargs):
        sigm = self.in_layer(x)

        x = torch.stack([self.offset_min + sigm[:, 0] * (self.offset_max - self.offset_min),
                         self.slope_min + sigm[:, 1] * (self.slope_max - self.slope_min),
                         self.emin + sigm[:, 2] * (self.emax - self.emin)], dim=1)

        y = torch.stack([x[:, 0], x[:, 1], 0.5 * x[:, 1] * (x[:, 2] - 1) / (self.up - self.lo)], dim=1)
        
        if not return_params:
            return super(RSquareHead, self).forward(y)

        else:
            return x


class LinearHead(PolynomialHead):
    def __init__(self, offset_min=0, offset_max=1, slope_min=0, slope_max=1, *args, **kwargs):
        super(LinearHead, self).__init__(dim_in=2, *args, **kwargs)

        self.in_layer = nn.Sequential(nn.BatchNorm1d(2), nn.Sigmoid())
        self.offset_min = offset_min
        self.offset_max = offset_max

        self.slope_min = slope_min
        self.slope_max = slope_max

    def forward(self, x):
        sigm = self.in_layer(x)
        x = torch.stack([self.offset_min + sigm[:, 1] * (self.offset_max - self.offset_min),
                         self.slope_min + sigm[:, 0] * (self.slope_max - self.slope_min)], dim=1)

        return super(LinearHead, self).forward(x)


class TwoGaussians(AbstractHead):
    def __init__(self, only=None, fixed_means=True, max_amp=None, max_sigma=None, min_amp=0, min_sigma=0,
                 max_off=None, min_off=None, off_single_param=False, *args, **kwargs):
        super(TwoGaussians, self).__init__(*args, **kwargs)

        self.PI = nn.Parameter(torch.tensor([3.1415927410125732], device=self.device, requires_grad=False), requires_grad=False)

        if not ((self.dim_in == 6 and only is None) or (self.dim_in != 3 and only in (0, 1))):
            raise Exception('TwoGaussians head needs exactly 6 or 3 parameters, you have configured %d' % self.dim_in)

        self.normed_out_wvls = nn.Parameter((self.out_wvls - self.out_wvls.mean()) / (self.out_wvls[-1] - self.out_wvls[0]), requires_grad=False)

        self.fixed_means = fixed_means
        self.fixed_means_ = None
        
        wvl_norm = self.out_wvls[-1] - self.out_wvls[0]
        self.mean1 = nn.Parameter((685 - self.out_wvls.mean()) / wvl_norm, requires_grad=False)
        self.mean2 = nn.Parameter((740 - self.out_wvls.mean()) / wvl_norm, requires_grad=False)

        self.max_amp = max_amp
        self.pre_sigmoid_amp = nn.BatchNorm1d(1)
        self.min_amp = min_amp

        self.max_sigma = max_sigma / wvl_norm if max_sigma is not None else None
        self.pre_sigmoid_sigma = nn.BatchNorm1d(1)
        self.min_sigma = min_sigma / wvl_norm if min_sigma is not None else None

        self.off_single_param = off_single_param
        if self.off_single_param:
            self.off_mean = nn.Parameter(torch.tensor(1.0).requires_grad_(True))

        self.max_off = max_off / wvl_norm if max_off is not None else None
        self.pre_sigmoid_off = nn.BatchNorm1d(1)
        self.min_off = min_off / wvl_norm if min_off is not None else None

        self.only = only

    def _gaussian_forward(self, x, default_mean=None):
        off_mean, sigma, amp = [arr.unsqueeze(0).permute(1, 0) for arr in x.permute(1, 0)]
        
        if self.fixed_means_ is None or self.fixed_means_.shape != off_mean.shape:
            self.fixed_means_ = torch.tensor(default_mean).repeat(off_mean.shape).to(x.device).requires_grad_(False)
        
        if self.off_single_param:
            off_mean = self.off_mean

        if self.max_off is not None:
            off = self.min_off + torch.relu((torch.sigmoid(self.pre_sigmoid_off(off_mean)) - 0.5) * (self.max_off - self.min_off) * 2)

        else:
            off = self.pre_sigmoid_off(off_mean)

        if self.fixed_means:
            off = 0

        mean = self.fixed_means_ + off

        if self.max_amp is not None:
            amp = self.min_amp + torch.relu((torch.sigmoid(self.pre_sigmoid_amp(amp)) - 0.5) * (self.max_amp - self.min_amp) * 2)

        else:
            amp = torch.relu(self.pre_sigmoid_amp(amp))

        if self.max_sigma is not None:
            sigma = self.min_sigma + torch.relu((torch.sigmoid(sigma) - 0.5) * (self.max_sigma - self.min_sigma) * 2)

        else:
            sigma = torch.relu(self.pre_sigmoid_sigma(sigma))

        return self.gaussian(mean, sigma, amp)

    def forward(self, x, **kwargs):

        if self.only is not None:
            if self.only == 0:
                peak1 = self._gaussian_forward(x[..., :3], default_mean=self.mean1)
                return peak1

            else:
                peak2 = self._gaussian_forward(x[..., :3], default_mean=self.mean2)
                return peak2

        else:
            peak1 = self._gaussian_forward(x[..., :3], default_mean=self.mean1)
            peak2 = self._gaussian_forward(x[..., 3:], default_mean=self.mean2)

        return peak1 + peak2

    def gauss_fcn(self, mean, sigma, amp, eps=1e-2):
        gauss = torch.exp( - ((self.normed_out_wvls - mean) ** 2) / (2 * (sigma + eps) ** 2)) * amp # / torch.sqrt(2 * self.PI) / (sigma + eps) * amp
       # with torch.no_grad():
       #     gauss = torch.einsum('ij, i -> ij', gauss, 1 / (gauss.max(dim=1)[0] + eps))
        
        #gauss = torch.einsum('ij, i -> ij', gauss, amp.squeeze())
        
        return gauss

    def gaussian(self, mean, sigma, amp, eps=1e-2):
        #gauss = torch.stack([self.gauss_fcn(m, s, a, eps=eps) for m, s, a in zip(mean, sigma, amp)])
        gauss = self.gauss_fcn(mean, sigma, amp, eps=eps)
        #print('SHAPE', gauss.shape, )
        return gauss


class Evidential(AbstractHead):
    def __init__(self, *args, **kwargs):
        super(Evidential, self).__init__(*args, **kwargs)

        self.out = nn.Linear(self.dim_in, 4)

    def forward(self, x, **kwargs):

        # compute 4 vars: gamma, alpha, beta, nu = out.transpose(1, 0)
        # out = f.elu(self.out(x))
        # out[:, 1:] += 1  # so there is a min of 0 for alpha, beta, nu
        # out[:, 1] += 1  # min(alpha) >= 1

        out = F.softplus(self.out(x))
        gamma, alpha, beta, nu = out.transpose(1, 0)

        alphap1 = alpha + 1
        nup1 = nu + 1

        return torch.stack([gamma, alphap1, beta, nu], dim=-1)


class ConstantHead(nn.Module):
    def __init__(self, return_):
        super(ConstantHead, self).__init__()

        self.return_ = nn.Parameter(return_, requires_grad=False)

    def forward(self, x, **kwargs):
        if type(x) is tuple:
            x, y = x

        if len(x.shape) == 4:
            return self.return_[..., None, None]

        return self.return_
