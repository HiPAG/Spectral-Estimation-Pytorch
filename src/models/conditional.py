import torch
import torch.nn as nn
import sys
import os
# print(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.estimator import Estimator
from models.residual_hyper_inference import ResidualHyperInference

class Conditional(nn.Module):
    def __init__(self, n_in, n_out, n_resblocks):
        super().__init__()
        self.estimator = Estimator(n_in, 93)
        self.residual_hyper_inference = ResidualHyperInference(96, n_out, n_resblocks)

    def load_checkpoint(self, *state_dict):
        self.estimator.load_state_dict(state_dict[0])
        self.residual_hyper_inference.load_state_dict(state_dict[1])

    def forward(self, x):
        sens = self.estimator(x)    # (31, 3)
        x = torch.cat([x, sens.view(1, -1, 1, 1).repeat(x.size(0), 1, *x.shape[2:])], dim=1)    #(x, 96, ?,?)
        y = self.residual_hyper_inference(x)

        return y



if __name__ == '__main__':
    model = Conditional(3, 31, 2)
    temp = torch.Tensor(3, 3, 256, 256)
    output = model(temp)
    print(output.shape)
