import torch
from models.conditional import Conditional

if __name__ == '__main__':
    model = Conditional(3, 31, 2)
    temp = torch.Tensor(3, 3, 256, 256)
    output = model(temp)
    print(output.shape)