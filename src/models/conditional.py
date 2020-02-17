import torch
import torch.nn as nn
from models.estimator import Estimator
from models.residual_hyper_inference import ResidualHyperInference


__ENTRANCE__ = 'Conditional'

class Conditional(nn.Module):
    def __init__(self, n_in, n_out, n_resblocks, ckp_est=None, ckp_inf=None):
        super().__init__()
        self.estimator = Estimator(n_in, 93)
        self.residual_hyper_inference = ResidualHyperInference(96, n_out, n_resblocks)

        # load pre-trained weights
        if ckp_est and ckp_inf:
            ckp_est = torch.load(ckp_est)
            ckp_inf = torch.load(ckp_inf)
            state_dict_est = ckp_est.get('state_dict', ckp_est)
            state_dict_inf = ckp_inf.get('state_dict', ckp_inf)
            self.estimator.load_state_dict(state_dict_est)
            self.residual_hyper_inference.load_state_dict(state_dict_inf)



    def forward(self, x):
        sens = self.estimator(x)    # (31, 3)
        x = torch.cat([x, sens.view(1, -1, 1, 1).repeat(x.size(0), 1, *x.shape[2:])], dim=1)    #(x, 96, ?,?)
        y = self.residual_hyper_inference(x)

        return y


if __name__ == '__main__':
    model = Conditional(3, 31, 2)
    temp = torch.Tensor(3, 3, 256, 256)
    state_dict = model.state_dict()
    # print(state_dict)
    print(state_dict.keys())
    output = model(temp)
    print(output.shape)
