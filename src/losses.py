import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from https://github.com/berk95kaya/Spectral-Estimation/blob/3c634de1ba196e3429e299c25bfee35f440c7e27/estimator.py
class SmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, n_chns):
        a = torch.ones(n_chns)*2
        b = torch.ones(n_chns-1)*-1
        A = self._tridiag(b, a, b)
        A[0,:] = 0
        return torch.sum(torch.pow(torch.matmul(A.to(x.device), x), 2))

    @staticmethod
    def _tridiag(a, b, c, k1=-1, k2=0, k3=1):
        return torch.diag(a, k1) + torch.diag(b, k2) + torch.diag(c, k3)
