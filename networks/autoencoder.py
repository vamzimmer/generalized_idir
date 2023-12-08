import torch
from torch.nn import functional as F
from networks.encoder_utils import *

def reparametrize(mu, logvar, training=True):
    r"""Reparametrization trick used in variational inference for autoencoders
        (Sample z from latent space using mu and logvar)

    This is used for the parametrisation of the latent variable vector so that it becomes differentiable

    https://arxiv.org/pdf/1312.6114.pdf


    .. math::
        x = \mu + \epsilon, \quad \epsilon \sim N(0,logvar)


    Args:
        mu (tensor) Input expectation values
        logvar (tensor) Input log-variance values
        training (bool) If true the returned value is the expectation augmented with random variable of log-variance
          logvar, if false the expectation alone is returned
    """
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    else:
        return mu

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_shape,
                        input_dim=1,
                        feature_dim=32,
                        num_layers=4,
                        hidden_dim=128,
                        output_dim=32,
                        norm_type='in',
                        activation_type='leaky_relu',
                        activation_type_last='none',
                        fc_layers=1,
                        upsample_mode='interp',
                        checkpoint=None):

        super().__init__()

        kwargs = {'norm_type': norm_type,
                  'activation_type': activation_type,
                  'kernel_size': 3,
                  'padding': 1,
                  'stride': 1,
                  'padding_mode':'zeros'}
        kwargs_convT = {'norm_type': norm_type,
                  'activation_type': activation_type,
                  'kernel_size': 3,
                  'padding': 1,
                  'stride': 2,
                  'output_padding' : 1,
                  'padding_mode':'zeros'}

        assert fc_layers in [1, 2]
        self.fc_layers = fc_layers
        assert upsample_mode in ['convT', 'interp']
        self.upsample_mode = upsample_mode
        
        # Define convolutional encoding layers
        self.enc_layers = []
        self.enc_layers.append(ConvBlock(input_dim, feature_dim, **kwargs))
        self.enc_layers.append(torch.nn.MaxPool3d(2))

        for i in range(0, num_layers-1):
            self.enc_layers.append(ConvBlock(feature_dim*2**i, feature_dim*2**(i+1), **kwargs))
            self.enc_layers.append(torch.nn.MaxPool3d(2))
        self.enc_layers = torch.nn.Sequential(*self.enc_layers)

        # Compute the output shape of the convolutional layers
        with torch.no_grad():
            self.enc_input_dim = feature_dim*2**(num_layers-1)
            enc = self.enc_layers(torch.zeros(1, input_dim, *input_shape))
            self.enc_shape = enc.shape[1:]
            self.enc_dim = enc.numel()

        # Define fully connected encoding layers
        self.enc_fc = []
        if self.fc_layers == 1:
            self.enc_fc.append(torch.nn.Linear(self.enc_dim, output_dim))
        elif self.fc_layers == 2:
            self.enc_fc.append(torch.nn.Linear(self.enc_dim, hidden_dim))
            self.enc_fc.append(get_activation(activation_type, hidden_dim))
            self.enc_fc.append(torch.nn.Linear(hidden_dim, output_dim))
        self.enc_fc = torch.nn.Sequential(*self.enc_fc)

        # Define fully connected decoding layers
        self.dec_fc = []
        self.dec_fc.append(get_activation(activation_type, output_dim))
        if self.fc_layers == 1:
            self.dec_fc.append(torch.nn.Linear(output_dim, self.enc_dim))
        elif self.fc_layers == 2:
            self.dec_fc.append(torch.nn.Linear(output_dim, hidden_dim))
            self.dec_fc.append(get_activation(activation_type, hidden_dim))
            self.dec_fc.append(torch.nn.Linear(hidden_dim, self.enc_dim))  
        self.dec_fc = torch.nn.Sequential(*self.dec_fc)

        # Define convolutional decoding layers
        self.dec_layers = []
        if self.upsample_mode == 'interp':
            self.dec_layers.append(torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            for i in range(num_layers-1, 0, -1):
                self.dec_layers.append(ConvBlock(feature_dim*2**(i), feature_dim*2**(i-1), **kwargs))
                self.dec_layers.append(torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            self.dec_layers.append(torch.nn.Conv3d(feature_dim, input_dim, padding_mode='replicate', kernel_size=kwargs['kernel_size'], padding=kwargs['padding']))
        elif self.upsample_mode == 'convT':
            for i in range(num_layers-1, 0, -1):
                self.dec_layers.append(ConvTBlock(feature_dim*2**(i), feature_dim*2**(i-1), **kwargs_convT))
            kwargs_convT['activation_type'] = activation_type_last
            kwargs_convT['norm_type'] = 'none'
            self.dec_layers.append(ConvTBlock(feature_dim, input_dim, **kwargs_convT))

        self.dec_layers = torch.nn.Sequential(*self.dec_layers)

        if checkpoint == None:
            # Initialization
            for m in self.modules():
                if isinstance(m, torch.nn.Conv3d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation_type)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.InstanceNorm3d, torch.nn.GroupNorm)):
                    if m.weight is not None:
                        torch.nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Linear):
                    #torch.nn.init.xavier_normal_(m.weight)
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation_type)
        else:
            print(f"Load State Dict for Autoencoder from '{checkpoint}'")
            cp = torch.load(checkpoint)
            self.load_state_dict(cp['model_state_dict'])

    def encode(self, x):
        # Apply convolution blocks
        x = self.enc_layers(x)
        # Flatten
        x = x.view(x.shape[0], -1)
        # Apply linear layer
        x = self.enc_fc(x)
        return x

    def decode(self, x):
        # Apply linear layer
        x = self.dec_fc(x)
        # Reshape
        x = x.view(x.shape[0],*self.enc_shape)
        # Apply convolution blocks
        x = self.dec_layers(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    
class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, input_shape,
                        input_dim=1,
                        feature_dim=32,
                        num_layers=4,
                        hidden_dim=128,
                        output_dim=32,
                        norm_type='in',
                        activation_type='leaky_relu',
                        activation_type_last='none',
                        fc_layers=1,
                        upsample_mode='interp'):

        super().__init__()

        kwargs = {'norm_type': norm_type,
                  'activation_type': activation_type,
                  'kernel_size': 3,
                  'padding': 1,
                  'stride': 1,
                  'padding_mode':'zeros'}
        kwargs_convT = {'norm_type': norm_type,
                  'activation_type': activation_type,
                  'kernel_size': 3,
                  'padding': 1,
                  'stride': 2,
                  'output_padding' : 1,
                  'padding_mode':'zeros'}

        assert fc_layers in [1, 2]
        self.fc_layers = fc_layers
        assert upsample_mode in ['convT', 'interp']
        self.upsample_mode = upsample_mode

        self.output_dim = output_dim
        
        # Define convolutional encoding layers
        self.enc_layers = []
        self.enc_layers.append(ConvBlock(input_dim, feature_dim, **kwargs))
        self.enc_layers.append(torch.nn.MaxPool3d(2))

        for i in range(0, num_layers-1):
            self.enc_layers.append(ConvBlock(feature_dim*2**i, feature_dim*2**(i+1), **kwargs))
            self.enc_layers.append(torch.nn.MaxPool3d(2))
        self.enc_layers = torch.nn.Sequential(*self.enc_layers)

        # Compute the output shape of the convolutional layers
        with torch.no_grad():
            self.enc_input_dim = feature_dim*2**(num_layers-1)
            enc = self.enc_layers(torch.zeros(1, input_dim, *input_shape))
            self.enc_shape = enc.shape[1:]
            self.enc_dim = enc.numel()

        # Define fully connected encoding layers
        self.enc_fc = []
        if self.fc_layers == 1:
            self.enc_fc.append(torch.nn.Linear(self.enc_dim, 2*output_dim))
        elif self.fc_layers == 2:
            self.enc_fc.append(torch.nn.Linear(self.enc_dim, hidden_dim))
            self.enc_fc.append(get_activation(activation_type, hidden_dim))
            self.enc_fc.append(torch.nn.Linear(hidden_dim, 2*output_dim))
        self.enc_fc = torch.nn.Sequential(*self.enc_fc)

        # Define fully connected decoding layers
        self.dec_fc = []
        if self.fc_layers == 1:
            self.dec_fc.append(torch.nn.Linear(output_dim, self.enc_dim))
        elif self.fc_layers == 2:
            self.dec_fc.append(torch.nn.Linear(output_dim, hidden_dim))
            self.dec_fc.append(get_activation(activation_type, hidden_dim))
            self.dec_fc.append(torch.nn.Linear(hidden_dim, self.enc_dim))  
        self.dec_fc = torch.nn.Sequential(*self.dec_fc)

        # Define convolutional decoding layers
        self.dec_layers = []
        if self.upsample_mode == 'interp':
            self.dec_layers.append(torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            for i in range(num_layers-1, 0, -1):
                self.dec_layers.append(ConvBlock(feature_dim*2**(i), feature_dim*2**(i-1), **kwargs))
                self.dec_layers.append(torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            self.dec_layers.append(torch.nn.Conv3d(feature_dim, input_dim, padding_mode='replicate', kernel_size=kwargs['kernel_size'], padding=kwargs['padding']))
        elif self.upsample_mode == 'convT':
            for i in range(num_layers-1, 0, -1):
                self.dec_layers.append(ConvTBlock(feature_dim*2**(i), feature_dim*2**(i-1), **kwargs_convT))
            kwargs_convT['activation_type'] = activation_type_last
            kwargs_convT['norm_type'] = 'none'
            self.dec_layers.append(ConvTBlock(feature_dim, input_dim, **kwargs_convT))

        self.dec_layers = torch.nn.Sequential(*self.dec_layers)
        # Initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation_type)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.InstanceNorm3d, torch.nn.GroupNorm)):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                #torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation_type)

    def encode(self, x):
        # Apply convolution blocks
        x = self.enc_layers(x)
        # Flatten
        x = x.view(x.shape[0], -1)
        # Apply linear layer
        x = self.enc_fc(x)
        return x

    def decode(self, x):
        # Apply linear layer
        x = self.dec_fc(x)
        # Reshape
        x = x.view(x.shape[0],*self.enc_shape)
        # Apply convolution blocks
        x = self.dec_layers(x)
        return x

    def forward(self, x):
        z = self.encode(x)

        # for variational autoencoder:
        # reparametrization trick
        mu = z[:, :self.output_dim]
        logvar = z[:, self.output_dim:]
        # Sample z from latent space using mu and logvar
        z = reparametrize(mu, logvar)

        return self.decode(z), mu, logvar
