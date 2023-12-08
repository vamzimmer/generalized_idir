import torch

def get_norm(norm_type, fin):
    if norm_type == 'in':
        return torch.nn.InstanceNorm3d(fin)
    elif norm_type == 'bn':
        return torch.nn.BatchNorm3d(fin)
    elif norm_type == 'none':
        return None
    else:
        raise ValueError
    
def get_activation(activation_type, fin):
    if activation_type == 'relu':
        return torch.nn.ReLU(inplace=True)
    elif activation_type == 'leaky_relu':
        return torch.nn.LeakyReLU(inplace=True)
    elif activation_type == 'none':
        return torch.nn.Identity()
    elif activation_type == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation_type == 'tanh':
        return torch.nn.Tanh()
    else:
        raise ValueError

class ConvBlock(torch.nn.Module):
    def __init__(self, fin, fout, kernel_size=3, stride=1, padding=1, padding_mode='zeros', norm_type='in', activation_type='relu'):
        super().__init__()
        self.conv = torch.nn.Conv3d(fin, fout, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    padding_mode=padding_mode)
        self.norm = get_norm(norm_type, fout)
        self.act = get_activation(activation_type, fout)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm != None:
            x = self.norm(x)
        x = self.act(x)
        return x

class ConvTBlock(torch.nn.Module):
    def __init__(self, fin, fout, kernel_size=3, stride=1, padding=1, output_padding=0, padding_mode='zeros', norm_type='in', activation_type='relu'):
        super().__init__()
        self.conv = torch.nn.ConvTranspose3d(fin, fout, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    output_padding=output_padding,
                                    padding_mode=padding_mode)
        self.norm = get_norm(norm_type, fout)
        self.act = get_activation(activation_type, fout)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm != None:
            x = self.norm(x)
        x = self.act(x)
        return x