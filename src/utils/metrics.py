from functools import partial

import numpy as np
import torch
from skimage.measure import compare_psnr as psnr, compare_ssim as ssim
from .psnr import PSNR as TENSOR_PSNR
from .ssim import SSIM as TENSOR_SSIM


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

class PSNR_TENSOR(AverageMeter):
    __name__ = 'PSNR_TENSOR'
    def __init__(self, **configs):
        psnr = TENSOR_PSNR(**configs)
        super().__init__(callback=psnr)


class SSIM(AverageMeter):
    __name__ = 'SSIM'
    def __init__(self, **configs):
        super().__init__(callback=partial(ssim, **configs))

class SSIM_TENSOR(AverageMeter):
    __name__ = 'SSIM_TENSOR'
    def __init__(self, **configs):
        ssim = TENSOR_SSIM(**configs)
        super().__init__(callback=ssim)

class RMSE(AverageMeter):
    __name__ = 'RMSE'
    def __init__(self):
        super().__init__(None)
    def compute(self, x1, x2):
        x1, x2 = x1.astype('float'), x2.astype('float')
        return np.sqrt(np.mean((x1-x2)**2))

class RMSE_TENSOR(AverageMeter):
    __name__ = 'RMSE_TENSOR'
    def __init__(self):
        super().__init__(None)
    def compute(self, x1, x2):
        assert x1.shape != 4, 'Input images must 4-d tensor.'
        assert x1.type() == x2.type(), 'Input images must 4-d tensor.'
        assert x1.shape == x2.shape, 'Input images must have the same dimensions.'
        x1, x2 = x1.type(torch.float), x2.type(torch.float)

        return torch.mean(torch.sqrt(torch.mean((x1 - x2)**2, dim=[1, 2, 3])))


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

def _naive_forecasting_tensor(actual, seasonality = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:, :, :-seasonality]


def _relative_error_tensor(actual, predicted, benchmark = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[:, :, seasonality:], predicted[:, :, seasonality:]) /\
               (_error(actual[:, :, seasonality:], _naive_forecasting_tensor(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


class MRAE_TENSOR(AverageMeter):
    __name__ = 'MRAE_TENSOR'
    def __init__(self):
        super().__init__(None)
    def compute(self, predicted, actual):
        assert predicted.shape != 4, 'Input images must 4-d tensor.'
        assert predicted.type() == actual.type(), 'Input images must 4-d tensor.'
        assert predicted.shape == actual.shape, 'Input images must have the same dimensions.'
        predicted, actual = predicted.type(torch.float), actual.type(torch.float)

        return torch.mean(torch.abs(_relative_error_tensor(actual, predicted, None)))




if __name__ =='__main__':
    from skimage import io
    import torchvision
    # true1 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\clear\clear\1.png')
    # true2 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\clear\clear\2.png')
    # true3 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\clear\clear\3.png')
    # img1 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\hazy\hazy\1_1_0.90179.png')
    # img2 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\hazy\hazy\2_1_0.99082.png')
    # img3 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\hazy\hazy\3_1_0.88255.png')

    true1 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\clear\clear\1.png').astype(np.float)/255
    true2 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\clear\clear\2.png').astype(np.float)/255
    true3 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\clear\clear\3.png').astype(np.float)/255
    img1 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\hazy\hazy\1_1_0.90179.png').astype(np.float)/255
    img2 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\hazy\hazy\2_1_0.99082.png').astype(np.float)/255
    img3 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\hazy\hazy\3_1_0.88255.png').astype(np.float)/255

    psnr = PSNR()
    ssim = SSIM(multichannel=True)
    mrae = MRAE()
    rmse = RMSE()

    psnr.update(true1, img1)
    ssim.update(true1, img1)
    mrae.update(true1, img1)
    rmse.update(true1, img1)

    description1 = 'numpy_version  PSNR:{} SSIM{} MRAE{} RMSE{}'.format(psnr.val, ssim.val, mrae.val, rmse.val)
    print(description1)




    # data_range 0-1
    np2tensor = torchvision.transforms.ToTensor()
    true1_t = np2tensor(true1).unsqueeze(0)
    img1_t = np2tensor(img1).unsqueeze(0)
    true2_t = np2tensor(true2).unsqueeze(0)
    img2_t = np2tensor(img2).unsqueeze(0)
    true3_t = np2tensor(true3).unsqueeze(0)
    img3_t = np2tensor(img3).unsqueeze(0)

    # data_range 0-255
    # true1_t = torch.Tensor(true1).permute(2, 0, 1).unsqueeze(0)
    # img1_t = torch.Tensor(img1).permute(2, 0, 1).unsqueeze(0)
    # true2_t = torch.Tensor(true2).permute(2, 0, 1).unsqueeze(0)
    # img2_t = torch.Tensor(img2).permute(2, 0, 1).unsqueeze(0)
    # true3_t = torch.Tensor(true3).permute(2, 0, 1).unsqueeze(0)
    # img3_t = torch.Tensor(img3).permute(2, 0, 1).unsqueeze(0)

    true_batch = torch.cat((true1_t, true2_t, true3_t))
    img_batch = torch.cat((img1_t, img2_t, img3_t))

    psnr_t = PSNR_TENSOR(data_range=1)
    ssim_t = SSIM_TENSOR(data_range=1)
    mrae_t = MRAE_TENSOR()
    rmse_t = RMSE_TENSOR()

    psnr_t.update(true1_t, img1_t)
    ssim_t.update(true1_t, img1_t)
    mrae_t.update(true1_t, img1_t)
    rmse_t.update(true1_t, img1_t)

    description2 = 'tensor_version  PSNR:{} SSIM{} MRAE{} RMSE{}'.format(psnr_t.val, ssim_t.val, mrae_t.val, rmse_t.val)
    print(description2)

    psnr.update(true2, img2)
    ssim.update(true2, img2)
    mrae.update(true2, img2)
    rmse.update(true2, img2)

    psnr.update(true3, img3)
    ssim.update(true3, img3)
    mrae.update(true3, img3)
    rmse.update(true3, img3)

    description3 = 'numpy_batch_version  PSNR:{} SSIM{} MRAE{} RMSE{}'.format(psnr.avg, ssim.avg, mrae.avg, rmse.avg)
    print(description3)

    psnr_t.reset()
    ssim_t.reset()
    mrae_t.reset()
    rmse_t.reset()

    psnr_t.update(true_batch, img_batch)
    ssim_t.update(true_batch, img_batch)
    mrae_t.update(true_batch, img_batch)
    rmse_t.update(true_batch, img_batch)

    description4 = 'tensor_batch_version  PSNR:{} SSIM{} MRAE{} RMSE{}'.format(psnr_t.val, ssim_t.val, mrae_t.val, rmse_t.val)
    print(description4)









