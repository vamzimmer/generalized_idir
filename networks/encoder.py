import torch
from networks.encoder_utils import *

class GlobalAveragePool3d(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim = (2,3,4))

class ResidualBlock(torch.nn.Module):
    def __init__(self, fin, fout, kernel_size=3, stride=1, padding=1, norm_type='in', activation_type='relu'):
        super().__init__()
        self.block1 = ConvBlock(fin, fout, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, activation_type=activation_type)
        self.block2 = ConvBlock(fout, fout, kernel_size=kernel_size, stride=1, padding=padding, norm_type=norm_type, activation_type=activation_type)
        self.activation_out = get_activation('none', fout)
        if stride == 1:
            self.downsample = None
        else:    
            norm = get_norm(norm_type, fout)
            self.downsample = torch.nn.Sequential(torch.nn.Conv3d(fin, fout, kernel_size=1, stride=stride), norm)


    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        if self.downsample != None:
            x = self.downsample(x)
        return self.activation_out(x + out)

class BasicEncoder(torch.nn.Module):
    def __init__(self, input_shape, feature_dim=32, num_layers=4, output_dim=128, norm_type='in', activation_type='leaky_relu'):
        super().__init__()

        kwargs = {'norm_type': norm_type,
                  'activation_type': activation_type,
                  'kernel_size': 3,
                  'padding': 1,
                  'stride': 1}
        
        # Define the convolution blocks in the encoder
        self.layers = []
        self.layers.append(ConvBlock(2, feature_dim, **kwargs))
        self.layers.append(torch.nn.MaxPool3d(4))

        for i in range(0, num_layers-1):
            self.layers.append(ConvBlock(feature_dim*2**i, feature_dim*2**(i+1), **kwargs))
            self.layers.append(torch.nn.MaxPool3d(2))

        self.layers = torch.nn.Sequential(*self.layers)

        # Compute the output shape of the convolutional layers
        with torch.no_grad():
            self.enc_input_dim = feature_dim*2**(num_layers-1)
            self.enc_shape = self.layers(torch.zeros(1, 2, *input_shape)).numel()

        # Define the linear layer to compute the latent features
        self.linear = torch.nn.Linear(self.enc_shape, output_dim)
        self.output_activation = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.InstanceNorm3d, torch.nn.GroupNorm)):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # Apply convolution blocks
        x = self.layers(x)
        # Flatten
        x = x.view(x.shape[0], -1)
        # Apply linear layer
        x = self.linear(x)
        # TODO apply sigmoid?
        # x = self.output_activation(x)
        return x


class ResidualEncoder(torch.nn.Module):
    def __init__(self, feature_dim=15, output_dim=128, norm_type='in', activation_type='leaky_relu'):
        super().__init__()

        kwargs = {'norm_type': norm_type,
                  'activation_type': activation_type,
                  'kernel_size': 3,
                  'padding': 1}
        
        self.layer_in = ConvBlock(2, feature_dim, kernel_size=7, stride=2, padding=3, activation_type=activation_type, norm_type=norm_type)

        self.residual_blocks = []
        self.residual_blocks.append(ResidualBlock(feature_dim, feature_dim, stride=2, **kwargs))
        self.residual_blocks.append(ResidualBlock(feature_dim, feature_dim, stride=1, **kwargs))
        self.residual_blocks.append(ResidualBlock(feature_dim, feature_dim*2, stride=2, **kwargs))
        self.residual_blocks.append(ResidualBlock(feature_dim*2,feature_dim*2, stride=1, **kwargs))
        self.residual_blocks.append(ResidualBlock(feature_dim*2,feature_dim*4, stride=2, **kwargs))
        self.residual_blocks.append(ResidualBlock(feature_dim*4,feature_dim*4, stride=1, **kwargs))
        self.residual_blocks = torch.nn.Sequential(*self.residual_blocks)

        self.layer_out = torch.nn.Conv3d(feature_dim*4, output_dim, kernel_size=1, stride=1, padding=0)
        #self.layer_out = ConvBlock(feature_dim*4, output_dim, kernel_size=1, stride=1, padding=0, activation_type='none', norm_type=norm_type)
        self.global_pool = GlobalAveragePool3d()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.InstanceNorm3d, torch.nn.GroupNorm)):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer_in(x)
        x = self.residual_blocks(x)
        x = self.layer_out(x)
        x = self.global_pool(x)
        return x
