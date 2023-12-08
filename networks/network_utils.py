import torch.nn as nn
import torch
import numpy as np

class SinLayer(nn.Module):
    def __init__(self, fin, fout, omega, is_first):
        super().__init__()
        self.linear = nn.Linear(fin, fout)
        self.omega = omega

        # TODO
        # if is_first:
        #     self.omega = omega
        # else:
        #     self.omega = 50
        print(is_first, self.omega)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if is_first:
                    torch.nn.init.uniform_(m.weight, -1 / fin, 1 / fin)
                else:
                    torch.nn.init.uniform_(m.weight,
                        -np.sqrt(6 / fin) / self.omega,
                        np.sqrt(6 / fin) / self.omega,)
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class SinCosLayer(nn.Module):
    def __init__(self, fin, fout, omega, is_first):
        super().__init__()
        self.linear = nn.Linear(fin, fout)
        self.omega = omega

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if is_first:
                    torch.nn.init.uniform_(m.weight, -1 / fin, 1 / fin)
                else:
                    torch.nn.init.uniform_(m.weight,
                        -np.sqrt(6 / fin) / self.omega,
                        np.sqrt(6 / fin) / self.omega,)
    def forward(self, x):
        arg = self.omega * self.linear(x)
        return torch.sin(arg), torch.cos(arg)
    
class CosLayer(nn.Module):
    def __init__(self, fin, fout, omega, is_first):
        super().__init__()
        self.linear = nn.Linear(fin, fout)
        self.omega = omega

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if is_first:
                    torch.nn.init.uniform_(m.weight, -1 / fin, 1 / fin)
                else:
                    torch.nn.init.uniform_(m.weight,
                        -np.sqrt(6 / fin) / self.omega,
                        np.sqrt(6 / fin) / self.omega,)
    def forward(self, x):
        return torch.cos(self.omega * self.linear(x))
    
class LinearLayer(nn.Module):
    def __init__(self, fin, fout, activation='none', init='default'):
        super().__init__()
        self.linear = nn.Linear(fin, fout)
        if activation == 'none':
            self.act = nn.Identity()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Activation '{activation}' does not exit in LinearLayer!")

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if init == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight)
                elif init == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight)
                else:
                    torch.nn.init.uniform_(m.weight, -1 / fin, 1 / fin)

    def forward(self, x):
        return self.act(self.linear(x))
