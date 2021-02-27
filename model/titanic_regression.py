import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import random
class TitanicModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size )
        # self.relu1 = nn.ReLU()
        # self.fc3 = nn.Linear(3, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        # out = self.relu1(out)
        # out = self.fc3(out)
        return out


if __name__ == '__main__':
    input_dim = 8
    hidden_dim = 4
    output_dim = 1
    torch.manual_seed(1111)
    np.random.seed(1111)
    random.seed(1111)
    model = TitanicModel(input_dim, hidden_dim, output_dim)
    print(list(model.parameters())[0].size())
    print(list(model.parameters())[1].size())
    # print(list(model.parameters())[2].size())

    example = torch.rand(1, 8)
    out = model(example)
    print(out)
    out = out.view(-1)
    print(out)



