from functools import partial

import numpy as np
from skimage.measure import compare_psnr as psnr, compare_ssim as ssim


EPSILON = 1e-10


class AverageMeter:
    def __init__(self, callback=None):
        super().__init__()
        if callback is not None:
            self.compute = callback
        self.reset()

    def compute(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            raise NotImplementedError

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, *args, n=1):
        self.val = self.compute(*args)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class PSNR(AverageMeter):
    __name__ = 'PSNR'
    def __init__(self, **configs):
        super().__init__(callback=partial(psnr, **configs))


class SSIM(AverageMeter):
    __name__ = 'SSIM'
    def __init__(self, **configs):
        super().__init__(callback=partial(ssim, **configs))


class RMSE(AverageMeter):
    __name__ = 'RMSE'
    def __init__(self):
        super().__init__(None)
    def compute(self, x1, x2):
        return np.sqrt(np.mean((x1-x2)**2))


# Adapted from
# Adapted from https://github.com/berk95kaya/Spectral-Estimation/blob/3c634de1ba196e3429e299c25bfee35f440c7e27/forecasting_metrics.py
# which is adapted from https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
def _error(actual, predicted):
    """ Simple error """
    return actual - predicted


def _naive_forecasting(actual, seasonality = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual, predicted, benchmark = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) /\
               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


class MRAE(AverageMeter):
    __name__ = 'MRAE'
    def __init__(self):
        super().__init__(None)
    def compute(self, predicted, actual):
        return np.mean(np.abs(_relative_error(actual, predicted, None)))