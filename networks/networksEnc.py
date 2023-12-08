import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from networks import encoder, autoencoder, attention
from networks.network_utils import SinLayer, CosLayer, SinCosLayer, LinearLayer

#
#   Code base modified from:
#   Wolterink et al.: Implicit Neural Representations for Deformable Image Registration, MIDL 2022
#   https://github.com/MIAGroupUT/IDIR
#

    
class Siren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, input_shape, layers,omega=10, encoder_type='none', modulation_type='none',
                 encoder_config=None, mapping_type='basic', mapping_size=16, mapping_sigma=10,
                 modulation_activation_type='relu'):
        """Initialize the network."""

        super(Siren, self).__init__()

        print("encoder type", encoder_type)
        print("modulation type", modulation_type)
        print("modulation activation type", modulation_activation_type)
        print("Omega", omega)

        #print(f"Mapping type: {mapping_type}")
        #self.mapping = mappings.BasicMapping()
        #layers[0] = 2 * dimension
        #if mapping_type == 'fourier':
        #    self.mapping = mappings.FourierMapping(f_in=dimension, mapping_size=mapping_size, sigma=mapping_sigma)

        # print(f"Fourier mapping size: {mapping_size}")
        # print(f"Fourier mapping sigma: {mapping_sigma}")

        # # adapt the channel number for the first Linear layer to the Fourier features
        # layers[0] = 2 * mapping_size
        self.encoder_type = encoder_type
        self.encoder_config = {}
        if encoder_config is not None:
            self.encoder_config = encoder_config
        # default arguments
        if "input_dim" not in self.encoder_config: self.encoder_config["input_dim"] = 1
        if "upsample_mode" not in self.encoder_config: self.encoder_config["upsample_mode"] = "convT"
        if "feature_dim" not in self.encoder_config: self.encoder_config["feature_dim"] = 32
        if "hidden_dim" not in self.encoder_config: self.encoder_config["hidden_dim"] = 128
        if "output_dim" not in self.encoder_config: self.encoder_config["output_dim"] = 64
        if "fc_layers" not in self.encoder_config: self.encoder_config["fc_layers"] = 1
        if "norm_type" not in self.encoder_config: self.encoder_config["norm_type"] = "bn"
        if "activation_type" not in self.encoder_config: self.encoder_config["activation_type"] = "leaky_relu"
        if "num_layers" not in self.encoder_config: self.encoder_config["num_layers"] = 4
        if "activation_type_last" not in self.encoder_config: self.encoder_config["activation_type_last"] = "none"
        if "activation_type_latent" not in self.encoder_config: self.encoder_config["activation_type_latent"] = "none"

        self.modulation_type = modulation_type

        self.n_layers = len(layers) - 1
        self.omega = omega

        self.activation_latent = encoder.get_activation(self.encoder_config["activation_type_latent"], self.encoder_config["output_dim"])
        print("activation latent", self.activation_latent)

        # Pure SIREN
        if self.encoder_type == 'none':
            pass # no additional things to setup

        # Learn the mapping
        # elif self.encoder_type == 'globalmapping':
        #     # Mapping
        #     layers[0] = 2 * mapping_size  # KHexp2-mapping{-pi}
        #     # layers[0] = mapping_size  # KHexp2
        #     self.mapping = mappings.LearnedMapping()
        #     # Encoder
        #     self.encoder = encoder.BasicEncoder(input_shape, feature_dim=self.encoder_config["feature_dim"], num_layers=self.encoder_config["num_layers"], output_dim=mapping_size*3)

        if self.encoder_type == 'AE':
            self.encoder = autoencoder.AutoEncoder(input_shape,
                                                   input_dim=self.encoder_config["input_dim"],
                                                   upsample_mode=self.encoder_config["upsample_mode"],
                                                   feature_dim=self.encoder_config["feature_dim"],
                                                   num_layers=self.encoder_config["num_layers"],
                                                   output_dim=self.encoder_config["output_dim"],
                                                   norm_type=self.encoder_config["norm_type"],
                                                   activation_type=self.encoder_config["activation_type"], 
                                                   activation_type_last=self.encoder_config["activation_type_last"])
            def encode(x):
                return self.encoder.encode(x)
            self.encode = encode

        elif self.encoder_type == 'AE_better':
            self.encoder = aae_pidhorskyi.AAE(z_dim=self.encoder_config["output_dim"],
                                              d=self.encoder_config["feature_dim"],
                                              cross_batch=self.encoder_config["cross_batch"],
                                              extra_layers=2)
            self.encode_latent = []

            # Compute the output shape of the convolutional layers
            with torch.no_grad():
                enc = self.encoder.encode(torch.zeros(1, self.encoder_config["input_dim"], *input_shape))
                self.enc_shape = enc.shape[1:]
                self.enc_dim = enc.numel()
            self.encode_latent.append(torch.nn.Linear(self.enc_dim, self.encoder_config["output_dim"]))
            self.encode_latent = torch.nn.Sequential(*self.encode_latent)
            def encode(x):
                z = self.encoder.encode(x)
                z = z.view(z.shape[0], -1)
                z = self.encode_latent(z)
                return z
            self.encode = encode

        elif self.encoder_type == 'VGG':
            self.encoder = vgg.VGGEncoder(layers=[11])
            # Compute the output shape of the convolutional layers
            with torch.no_grad():
                enc = self.encoder(torch.zeros(1, self.encoder_config["input_dim"], *input_shape))[0]
                self.enc_shape = enc.shape[1:]
                self.enc_dim = enc.numel()
            self.encode_latent = []
            self.encode_latent.append(LinearLayer(self.enc_dim, self.encoder_config["output_dim"], "none"))
            self.encode_latent = torch.nn.Sequential(*self.encode_latent)
            def encode(x):
                z = self.encoder(x)[0]
                z = z.view(z.shape[0], -1)
                z = self.encode_latent(z)
                return z
            self.encode = encode
        elif self.encoder_type != 'none':
            raise ValueError("Encoder type must be in ['AE', 'AE_better', 'VGG', 'none']")
        
        # Global Latent Vector
        if self.modulation_type == 'globalAM':
            # self.encoder = encoder.BasicEncoder(input_shape, feature_dim=self.encoder_config["feature_dim"], num_layers=self.encoder_config["num_layers"], output_dim=mapping_size)
            self.enc1_layers = []
            self.enc2_layers = []
            self.layers_P = []
            # Make the encoding layers
            for i in range(self.n_layers-1):
                self.enc1_layers.append(LinearLayer(mapping_size, layers[i + 1], activation=modulation_activation_type))
                #self.enc2_layers.append(HiddenSigmoidLayer(mapping_size, layers[i + 1], self.omega, i))
                self.layers_P.append(SinLayer(layers[i], layers[i + 1], self.omega, True if i == 0 else False))
            self.enc1_layers = nn.ModuleList(self.enc1_layers)
            #self.enc2_layers = nn.ModuleList(self.enc2_layers)
            self.layers_P = nn.ModuleList(self.layers_P)

        # Local Latent Vector, take stacked fixed/moving patch as input, modulated SIREN
        elif self.modulation_type == 'localAM':
            self.enc_layers = []    
            for i in range(self.n_layers-1):
                self.enc_layers.append(LinearLayer(mapping_size + layers[i] if i>0 else mapping_size, layers[i + 1], activation=modulation_activation_type))
            self.enc_layers = nn.ModuleList(self.enc_layers)

        # Separate Local Latent Vectors, Quadratur amplitude modulation
        elif self.modulation_type == 'localQAM':
            self.enc_layers_fixed = []    
            self.enc_layers_moving = []    
            self.layers_cos = []
            self.layers_sin = []
            for i in range(self.n_layers-1):
                self.enc_layers_fixed.append(LinearLayer(mapping_size + layers[i] if i>0 else mapping_size, layers[i + 1], activation=modulation_activation_type))
                self.enc_layers_moving.append(LinearLayer(mapping_size + layers[i] if i>0 else mapping_size, layers[i + 1], activation=modulation_activation_type))
                self.layers_cos.append(CosLayer(layers[i], layers[i + 1], self.omega, True if i == 0 else False))
                self.layers_sin.append(SinLayer(layers[i], layers[i + 1], self.omega, True if i == 0 else False))
            self.enc_layers_fixed = nn.ModuleList(self.enc_layers_fixed)
            self.enc_layers_moving = nn.ModuleList(self.enc_layers_moving)
            self.layers_cos = nn.ModuleList(self.layers_cos)
            self.layers_sin = nn.ModuleList(self.layers_sin)

        # Separate Local Latent Vectors, Quadratur amplitude modulation with fixed frequency response for sin/cos
        elif self.modulation_type == 'localQAMv2':
            self.enc_layers_fixed = []    
            self.enc_layers_moving = []    
            self.layers_sincos = []
            for i in range(self.n_layers-1):
                self.enc_layers_fixed.append(LinearLayer(mapping_size + layers[i] if i>0 else mapping_size, layers[i + 1], activation=modulation_activation_type))
                self.enc_layers_moving.append(LinearLayer(mapping_size + layers[i] if i>0 else mapping_size, layers[i + 1], activation=modulation_activation_type))
                self.layers_sincos.append(SinCosLayer(layers[i], layers[i + 1], self.omega, True if i == 0 else False))
            self.enc_layers_fixed = nn.ModuleList(self.enc_layers_fixed)
            self.enc_layers_moving = nn.ModuleList(self.enc_layers_moving)
            self.layers_sincos = nn.ModuleList(self.layers_sincos)

        # Local Latent Vector, take stacked fixed/moving patch as input, modulated SIREN with Attention layer
        elif self.modulation_type in ['localAttentionAM', 'localCrossAttentionAM', 'localCrossAttentionAMv2']:
            self.attn_layers = []    
            hidden_dim = 64
            for i in range(self.n_layers-1):
                self.attn_layers.append(attention.AttentionLayer(mapping_size + layers[i] if i>0 else mapping_size, hidden_dim, layers[i + 1], activation=modulation_activation_type))
            self.attn_layers = nn.ModuleList(self.attn_layers)

        elif self.modulation_type == 'SIREN+':
            self.enc_layer = LinearLayer(mapping_size, mapping_size, init='latent', activation=modulation_activation_type)
        elif self.modulation_type != 'none':
            raise ValueError(f"Modulation Type '{self.encoder_type}' does not exist!")

        if not self.modulation_type in ['localQAM', 'localQAMv2']:
            # Make the layers
            self.layers = []
            for i in range(self.n_layers-1):
                self.layers.append(SinLayer(layers[i], layers[i + 1], self.omega, True if i == 0 else False))
            self.layers = nn.ModuleList(self.layers)

        # Define last output layer
        self.layer_last = LinearLayer(layers[self.n_layers-1], layers[self.n_layers], activation='none')


    def forward(self, x, encoder_input):
        """The forward function of the network."""
        # if self.modulation_type == 'globalmapping':
        #     latent_code = self.encoder(encoder_input)
        #     x = self.mapping(x, latent_code.view(3, -1))

        if self.modulation_type == 'globalAM' and self.encoder_type != "none":
            # latent_code = self.encoder(encoder_input)
            latent_code = self.encode(encoder_input)
            # concatenate latent codes for fixed and moving images
            latent_code = torch.cat(torch.tensor_split(latent_code, 2, dim=0), -1)
            latent_code = self.activation_latent(latent_code)

            #print(encoder_input.shape, latent_code.shape)

        elif self.modulation_type in ['localAM', 'localAttentionAM'] and self.encoder_type != "none":
            latent_code = self.encode(encoder_input)
            # concatenate latent codes for fixed and moving images
            latent_code = torch.cat(torch.tensor_split(latent_code, 2, dim=0), -1)
            latent_code = self.activation_latent(latent_code)

        elif self.modulation_type == 'SIREN+':
            latent_code = self.encode(encoder_input)
            # concatenate latent codes for fixed and moving images
            latent_code = torch.cat(torch.tensor_split(latent_code, 2, dim=0), -1)
            latent_code = self.activation_latent(latent_code)
            latent_code = latent_code.repeat(x.shape[0], 1)
            latent_code = self.enc_layer(latent_code/128) # transform to [-1, 1] for coordinates
            x = torch.concatenate([x , latent_code], dim=1)

        elif self.modulation_type in ['localQAM', 'localQAMv2', 'localCrossAttentionAM', 'localCrossAttentionAMv2'] and self.encoder_type != "none":
            input_fixed, input_moving = torch.tensor_split(encoder_input, 2, dim=0)
            latent_code_fixed = self.encode(input_fixed)
            latent_code_moving = self.encode(input_moving)
            #latent_code = latent_code.view(x.shape[0], -1)
            latent_code_fixed = self.activation_latent(latent_code_fixed)
            latent_code_moving = self.activation_latent(latent_code_moving)
        
        # Start processing...
        for i in range(self.n_layers-1):
            if self.modulation_type == 'globalAM' and self.encoder_type != "none":
                xi = self.layers[i](x)
                enc1_i = self.enc1_layers[i](latent_code)         
                x_P = self.layers_P[i](x)
                #enc2_i = self.enc2_layers[i](latent_code)
                x = enc1_i * xi + (1 - enc1_i) * x_P

            elif self.modulation_type == 'localAM' and self.encoder_type != "none":
                alpha_i = self.enc_layers[i](torch.concatenate([alpha_i, latent_code], dim=1) if i > 0 else latent_code)
                xi = self.layers[i](x)
                x = alpha_i * xi

            elif self.modulation_type == 'localQAM' and self.encoder_type != "none":
                alpha_i = self.enc_layers_fixed[i](torch.concatenate([alpha_i, latent_code_fixed], dim=1) if i > 0 else latent_code_fixed)
                beta_i = self.enc_layers_moving[i](torch.concatenate([beta_i, latent_code_moving], dim=1) if i > 0 else latent_code_moving)
                xi_sin = self.layers_sin[i](x)
                xi_cos = self.layers_cos[i](x)
                x = alpha_i * xi_sin + beta_i * xi_cos

            elif self.modulation_type == 'localQAMv2' and self.encoder_type != "none":
                alpha_i = self.enc_layers_fixed[i](torch.concatenate([alpha_i, latent_code_fixed], dim=1) if i > 0 else latent_code_fixed)
                beta_i = self.enc_layers_moving[i](torch.concatenate([beta_i, latent_code_moving], dim=1) if i > 0 else latent_code_moving)
                xi_sin, xi_cos = self.layers_sincos[i](x)
                x = alpha_i * xi_sin + beta_i * xi_cos

            elif self.modulation_type == 'localAttentionAM' and self.encoder_type != "none":
                h = torch.concatenate([alpha_i, latent_code], dim=1) if i > 0 else latent_code
                alpha_i = self.attn_layers[i](h, h)
                xi = self.layers[i](x)
                x = alpha_i * xi

            elif self.modulation_type == 'localCrossAttentionAM' and self.encoder_type != "none":
                h_f = torch.concatenate([alpha_i, latent_code_fixed], dim=1) if i > 0 else latent_code_fixed
                h_m = torch.concatenate([alpha_i, latent_code_moving], dim=1) if i > 0 else latent_code_moving
                alpha_i = self.attn_layers[i](h_f, h_m)
                xi = self.layers[i](x)
                x = alpha_i * xi

            elif self.modulation_type == 'localCrossAttentionAMv2' and self.encoder_type != "none":
                h_f = torch.concatenate([alpha_i, latent_code_fixed], dim=1) if i > 0 else latent_code_fixed
                h_m = torch.concatenate([alpha_i, latent_code_moving], dim=1) if i > 0 else latent_code_moving
                alpha_i = self.attn_layers[i](h_m, h_f)
                xi = self.layers[i](x)
                x = alpha_i * xi

            else:
                x = self.layers[i](x)

        # Propagate through final layer and return the output
        #return self.layers[-1](x)
        return self.layer_last(x)
