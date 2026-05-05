import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from fluomapper.utils import data as da

from scipy.interpolate import interp1d
from scipy.signal.windows import gaussian
import copy


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


class GaussianSmoothing(nn.Module):
    """
    Implementation taken from
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8

    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, sigma, kernel_size=None, keep_dim=True, kernel_size_factor=None, variable_kernel=False,
                 dim=2):
        super(GaussianSmoothing, self).__init__()

        self.dim = dim
        self.sigma = sigma  # if sigma is not numbers.Number else [sigma] * dim
        self.kernel_size = kernel_size  # if kernel_size is not numbers.Number else [kernel_size] * dim

        if kernel_size_factor is not None:
            self.kernel_size = self.sigma * kernel_size_factor

        self.kernel_size, self.sigma = self._format(self.kernel_size, self.sigma, self.dim)

        self.channels = channels

        self.kernel_size_factor = kernel_size_factor
        self.keep_dim = keep_dim

        self.variable_kernel = variable_kernel

        self._meshgrids = None

        if not variable_kernel:
            self.weights, self.conv, self.padding = self.construct_kernel(self.kernel_size, self.sigma, self.dim)
            self.weights = nn.Parameter(self.weights, requires_grad=False)

    def _format(self, kernel_size, sigma, dim):
        kernel_size = [kernel_size] * dim
        kernel_size = [round_up_to_odd(k) for k in kernel_size]

        sigma = [sigma] * dim

        return kernel_size, sigma

    def construct_kernel(self, kernel_size, sigma, dim, device='cpu', recalc_meshgrids=False):

        if self._meshgrids is None or recalc_meshgrids:
            meshgrids = torch.meshgrid(
                [
                    torch.arange(size, dtype=torch.float32, device=device)
                    for size in kernel_size
                ]
            )

            self._meshgrids = meshgrids

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = None
        for size, std, mgrid in zip(kernel_size, sigma, self._meshgrids):
            mean = (size - 1) / 2
            std = torch.tensor(std).reshape(*([-1] + dim * [1]))
            factor = 1 / (std * math.sqrt(2 * math.pi)) * \
                     torch.exp(- (mgrid - mean) ** 2 / 2 / std ** 2)

            if kernel is None:
                kernel = factor

            else:
                kernel *= factor

        kernel = torch.einsum('o..., o -> o...', kernel, 1 / torch.sum(kernel, dim=tuple(range(1, dim + 1))))

        kernel = kernel[:, None, None].repeat(1, self.channels, self.channels,
                                              *[1] * (kernel.dim() - 1))  # add channel in and out

        if dim == 1:
            conv = F.conv1d
        elif dim == 2:
            conv = F.conv2d
        elif dim == 3:
            conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        if self.keep_dim:
            padding = [int(k // 2) for k in kernel_size]

        else:
            padding = [0 for k in kernel_size]

        return kernel, conv, padding

    def _construct_weights(self, fwhm_shift, device):
        sigma = [s + fwhm_shift / 2.35 for s in self.sigma]
        # kernel_size = [self.kernel_size_factor * int(s) for s in self.sigma]
        weights, conv_, padding = self.construct_kernel(self.kernel_size,
                                                        sigma, self.dim,
                                                        device=device)
        return weights, conv_, padding

    def forward(self, input, **kwargs):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        if not self.variable_kernel:
            weights, conv_, padding = self.weights, self.conv, self.padding

        else:
            fwhm_shift = kwargs.get('fwhm_shift', 0)
            weights, conv_, padding = self._construct_weights(fwhm_shift, device=input.device)

        conv = self._convolve(conv_, input, weights, padding=tuple(padding))

        return conv

    def _convolve(self, conv_, input, weights, padding, *args, **kwargs):
        do_squeeze = False
        if len(input.shape) == 2:
            do_squeeze = True
            input = input[:, None]

        if not weights.shape[0] == input.shape[0]:
            assert weights.shape[0] == 1

            weights = weights.repeat(input.shape[0], *[1] * (weights.dim() - 1))

        N, cin = input.shape[:2]
        cout = weights.shape[1]

        input_ = input.view(1, -1, *input.shape[2:])
        weights_ = weights.view(-1, *weights.shape[2:])

        conv = torch.conv1d(input_, weights_, groups=N, padding=padding, *args, **kwargs)
        conv = conv.view(N, cout, *[input.shape[i] - self.kernel_size[i - 2] + 1 + 2 * padding[i - 2] for i in
                                    range(2, len(input.shape))])

        if do_squeeze:
            conv = conv.squeeze(1)

        return conv


class Resampler(object):
    """
    Convolving and Resampling without backpropagation
    """

    def __init__(self, wvl, new_wvls, fwhm=0.33, acc=5, unif_resample_method='linear', sigma_considered=2):

        self.wvl = wvl
        self.new_wvls = new_wvls
        sampling_interval = min(set(np.diff(self.new_wvls)))

        self.unif_resample_method = unif_resample_method

        self._unif_grid = np.arange(max(self.wvl[0], new_wvls[0] + sampling_interval),
                                    min(self.wvl[-1], new_wvls[-1] - sampling_interval),
                                    sampling_interval / acc)

        self.sampling_interval = sampling_interval / acc

        if type(fwhm) in (float, int) and fwhm == 0:
            self.mode = 'delta_peak'
            self.cw_inds = np.array(da.search_spectral_window(*new_wvls, where=self.wvl, pairs=False))

        elif fwhm == 'box':
            self.mode = 'box'

            self._reduction_stops = [0]
            diffs = np.diff(new_wvls)
            mid_diffs = np.r_[diffs / 2, sampling_interval / 2]

            for i, w in enumerate(new_wvls):
                inds = np.where(w + mid_diffs[i] > self._unif_grid)[0]
                if len(inds) == 0:
                    pass

                else:
                    ind = inds[-1]
                    self._reduction_stops.append(ind)

            self._reduction_stops = np.array(self._reduction_stops)

        # if we want a regular sampling
        elif type(fwhm) is float:
            self.mode = 'convolve'

            self.weights = [self._get_weights(fwhm, sigma_considered=sigma_considered)] * len(new_wvls)

            self.cw_inds = np.array(da.search_spectral_window(*new_wvls, where=self._unif_grid, pairs=False))

        # else it is assumed fwhm is array of fwhm values for gaussian convolutions per band
        else:
            self.mode = 'convolve'

            self.weights = [self._get_weights(f, sigma_considered=sigma_considered)
                            for f in fwhm]

            self.cw_inds = np.array(da.search_spectral_window(*new_wvls, where=self._unif_grid, pairs=False))

    def _get_weights(self, fwhm, sigma_considered=1):
        sigma = fwhm / 2 / np.sqrt(2 * np.log(2))

        sigma_considered_nm = sigma * sigma_considered

        n = int(np.ceil(sigma_considered_nm / self.sampling_interval))
        std = sigma / self.sampling_interval
        weights = gaussian(2 * n + 1, std=std)
        weights /= weights.sum()

        return weights

    def resample(self, signals, axis=-1):
        if self.mode == 'delta_peak':
            return self._delta_peak_resample(signals)

        if type(signals) is torch.Tensor:
            pass

        signals = interp1d(x=self.wvl,
                           y=signals,
                           axis=axis,
                           kind=self.unif_resample_method)(self._unif_grid)

        if self.mode == 'box':
            return self._box_resample(signals)

        elif self.mode == 'convolve':
            return self._convolve_resample(signals)

        else:
            raise NotImplementedError

    def _delta_peak_resample(self, signals):
        signals = signals[..., self.cw_inds]
        return signals

    def _convolve_resample(self, signals):
        ret = []

        for i, cw_ind in enumerate(self.cw_inds):
            n = int((len(self.weights[i]) - 1) / 2)

            lo = max(0, cw_ind - n)
            hi = min(signals.shape[-1], cw_ind + n + 1)

            diff_lo = - (cw_ind - n) if cw_ind - n < 0 else 0
            diff_hi = - (cw_ind + n + 1 - signals.shape[-1]) if cw_ind + n + 1 > signals.shape[-1] else signals.shape[
                -1]

            signal_window = signals[..., lo: hi]

            weights = copy.deepcopy(self.weights[i][diff_lo: diff_hi])
            weights /= weights.sum()

            ret.append((signal_window * weights).sum(-1))

        ret = np.stack(ret, axis=-1)
        return ret

    def _box_resample(self, signals):
        resampled_signal = np.add.reduceat(signals,
                                           indices=self._reduction_stops,
                                           axis=-1)

        resampled_signal = np.einsum('...n, n -> ...n',
                                     resampled_signal,
                                     1 / np.diff(np.r_[self._reduction_stops,
                                                       len(signals)]))

        # don't include last started bin
        resampled_signal = resampled_signal[..., :-1]

        return resampled_signal
