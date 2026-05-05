import torch
import torch.nn as nn


# copied from here:
# https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
# Cubic Hermite Splines


def h_poly_helper(tt):
    A = torch.tensor([[1, 0, -3, 2],
                      [0, 1, -2, 1],
                      [0, 0, 3, -2],
                      [0, 0, -1, 1]],
                     dtype=tt[-1].dtype)
    return [sum(A[i, j] * tt[j] for j in range(4)) for i in range(4)]


def h_poly(t):
    tt = [None for _ in range(4)]
    tt[0] = 1
    for i in range(1, 4):
        tt[i] = tt[i - 1] * t
    return h_poly_helper(tt)


def H_poly(t):
    tt = [None for _ in range(4)]
    tt[0] = t
    for i in range(1, 4):
        tt[i] = tt[i - 1] * t * i / (i + 1)
    return h_poly_helper(tt)


def interp(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    I = torch.searchsorted(x[1:], xs)
    dx = (x[I + 1] - x[I])
    hh = h_poly((xs - x[I]) / dx)
    return hh[0] * y[I] + hh[1] * m[I] * dx + hh[2] * y[I + 1] + hh[3] * m[I + 1] * dx


def integ(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    I = torch.searchsorted(x[1:], xs)
    Y = torch.zeros_like(y)
    Y[1:] = (x[1:] - x[:-1]) * ((y[:-1] + y[1:]) / 2 + (m[:-1] - m[1:]) * (x[1:] - x[:-1]) / 12)
    Y = Y.cumsum(0)
    dx = (x[I + 1] - x[I])
    hh = H_poly((xs - x[I]) / dx)
    return Y[I] + dx * (hh[0] * y[I] + hh[1] * m[I] * dx + hh[2] * y[I + 1] + hh[3] * m[I + 1] * dx)


class CubicSpline(nn.Module):
    def __init__(self, x, out_x, mode='interp'):
        super(CubicSpline, self).__init__()

        self.x = x
        self.out_x = out_x

        self.mode = mode

    def forward(self, batch):
        if self.mode == 'interp':
            ret = torch.stack([interp(self.x, b, self.out_x) for b in batch])

        else:
            raise NotImplementedError

        return ret
