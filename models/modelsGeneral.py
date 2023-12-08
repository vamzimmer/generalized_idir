import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
import numpy as np

from utils import general, eval_utils
#from networks import networksNaive as networksN
from networks import networks, networksEnc
from objectives import ncc
from objectives import regularizers
from models import utils
from visualization import display_utils

import SimpleITK as sitk

#
#   Code base modified from:
#   Wolterink et al.: Implicit Neural Representations for Deformable Image Registration, MIDL 2022
#   https://github.com/MIAGroupUT/IDIR
#

class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, moving_image, coordinate_tensor=None, index_tensor=None, fixed_image=None, 
        output_shape=(28, 28), dimension=0, slice_pos=0, batch_size=1900
    ):
        """Return the image-values for the given input-coordinates."""

        # Use standard coordinate tensor if none is given
        if coordinate_tensor is None:
            coordinate_tensor = self.make_coordinate_slice(
                output_shape, dimension, slice_pos
            )

        coordinate_batches = torch.tensor_split(coordinate_tensor, index_tensor.size()[0]//batch_size+1 ,dim=0)
        
        if self.patch_size is not None:
            index_batches = torch.tensor_split(index_tensor, index_tensor.size()[0]//batch_size+1 ,dim=0)

            # get patches at indices
            #TODO: for all batches in index_tensor
            print(f"Batchwise ({len(index_batches)}) displacement prediction...")
            for b in tqdm.tqdm(range(len(index_batches))):
                # print(b)

                patches = general.sample_all_patch_pairs_3d(torch.unsqueeze(torch.unsqueeze(fixed_image,0),0), torch.unsqueeze(torch.unsqueeze(moving_image,0),0), index_batches[b], self.patch_size)

                # reshape patches, depending on number of input channels for encoder
                # input_dim=1 : encoding of fixed and moving patches individually
                # input_dim=1 : simultaneous encoding of image patches
                if self.encoder_input_dim==1:
                    patch_tensor = patches.view(2*index_batches[b].size()[0],1,self.patch_size[0],self.patch_size[1],self.patch_size[2])
                else:
                    patch_tensor = patches.view(index_batches[b].size()[0],2,self.patch_size[0],self.patch_size[1],self.patch_size[2])

                output_batch = self.network(coordinate_batches[b].cuda(), patch_tensor.cuda())

                try:
                    output = torch.cat((output, output_batch.cpu().detach()), dim=0)
                except:
                    output = output_batch.cpu().detach()

        else:
            images = torch.unsqueeze(torch.stack((fixed_image, moving_image), dim=0), 1)
            
            ##########################
            # Veronika, 02.03.2023
            coordinate_batches = torch.tensor_split(coordinate_tensor, coordinate_tensor.size()[0]//batch_size+1 ,dim=0)

            print(f"Batchwise ({len(coordinate_batches)}) displacement prediction...")
                
            for b in tqdm.tqdm(range(len(coordinate_batches))):
                output_batch = self.network(coordinate_batches[b], images.cuda())

                try:
                    output = torch.cat((output, output_batch.cpu().detach()), dim=0)
                except:
                    output = output_batch.cpu().detach()


            # output = self.network(coordinate_tensor, images.cuda())

            # continue

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(output.cuda(), coordinate_tensor) #TODO stack all coordinates
        self.displacement = output
        self.transformation = coord_temp

        transformed_image = self.transform_no_add(coord_temp, moving_image=moving_image.cuda())

        ##########################
        # Veronika, 26.1.2023
        if len(output_shape)==2:
            return (
                transformed_image.cpu()
                .detach()
                .numpy()
                .reshape(output_shape[0], output_shape[1])
            )
        else:
            return (
                transformed_image.cpu()
                .detach()
                .numpy()
                .reshape(output_shape[0], output_shape[1], output_shape[2])
            )
        # ##########################

    def compute_landmarks2(self, landmarks_pre, image_size, moving_image, fixed_image):
        # print("Compute landmarks")

        # landmarks to normalized coodinate system
        scale_of_axes = [(0.5 * s) for s in image_size]
        coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

        if self.patch_size is not None:

            nr_patches = landmarks_pre.shape[0]

            # get patch coordinates with landmarks_pre as patch_centers
            patches = general.sample_all_patch_pairs_3d(torch.unsqueeze(torch.unsqueeze(fixed_image,0),0), torch.unsqueeze(torch.unsqueeze(moving_image,0),0), torch.from_numpy(landmarks_pre), self.patch_size)
            
            # reshape patches, depending on number of input channels for encoder
            # input_dim=1 : encoding of fixed and moving patches individually
            # input_dim=1 : simultaneous encoding of image patches
            if self.encoder_input_dim==1:
                patch_tensor = patches.view(2*nr_patches,1,self.patch_size[0],self.patch_size[1],self.patch_size[2])
            else:
                patch_tensor = patches.view(nr_patches,2,self.patch_size[0],self.patch_size[1],self.patch_size[2])
            
            # network
            output = self.network(coordinate_tensor.cuda(), patch_tensor.cuda())  

        else:
            images = torch.unsqueeze(torch.stack((fixed_image, moving_image), dim=0), 1)

            # network
            output = self.network(coordinate_tensor.cuda(), images.cuda())

        # back to image coordinate system
        delta = output.cpu().detach().numpy() * (scale_of_axes)

        return landmarks_pre + delta, delta 

    def compute_landmarks(self, landmarks_pre, image_size, moving_image, fixed_image, batch_size=1000, verbose=True):
        # print("Compute landmarks")

        landmarks_pre_batches = torch.tensor_split(torch.from_numpy(landmarks_pre), landmarks_pre.shape[0]//batch_size+1 ,dim=0)
        moving_image_batches = torch.tensor_split(moving_image, landmarks_pre.shape[0]//batch_size+1 ,dim=0) #TODO check if this works for batches!
        fixed_image_batches = torch.tensor_split(fixed_image, landmarks_pre.shape[0]//batch_size+1 ,dim=0)

        # landmarks to normalized coodinate system
        scale_of_axes = [(0.5 * s) for s in image_size]
        coordinate_tensor = landmarks_pre / (scale_of_axes) - 1.0
        coordinate_tensor_batches = torch.tensor_split(torch.from_numpy(coordinate_tensor).float(), coordinate_tensor.shape[0]//batch_size+1 ,dim=0)

        pbar = range(len(landmarks_pre_batches))
        if verbose:
            print(f"Batchwise ({len(landmarks_pre_batches)}) landmark displacement prediction...")
            pbar = tqdm.tqdm(range(len(landmarks_pre_batches)))
        # for b in tqdm.tqdm(range(len(landmarks_pre_batches))):
        for b in pbar:
            

            if self.patch_size is not None:

                # get patch coordinates with landmarks_pre as patch_centers
                patches = general.sample_all_patch_pairs_3d(torch.unsqueeze(torch.unsqueeze(fixed_image,0),0), torch.unsqueeze(torch.unsqueeze(moving_image,0),0), landmarks_pre_batches[b], self.patch_size)
                nr_patches = landmarks_pre_batches[b].size()[0]

                if self.encoder_input_dim==1:
                    patch_tensor = patches.view(2*nr_patches,1,self.patch_size[0],self.patch_size[1],self.patch_size[2])
                else:
                    patch_tensor = patches.view(nr_patches,2,self.patch_size[0],self.patch_size[1],self.patch_size[2])

                # network
                output_batch = self.network(coordinate_tensor_batches[b].cuda(), patch_tensor.cuda())  

            else:
                # images_batch = torch.unsqueeze(torch.stack((fixed_image_batches[b], moving_image_batches[b]), dim=0), 1)
                images = torch.unsqueeze(torch.stack((fixed_image, moving_image), dim=0), 1)

                # network
                # output_batch = self.network(coordinate_tensor_batches[b].cuda(), images_batch.cuda())
                output_batch = self.network(coordinate_tensor_batches[b].cuda(), images.cuda())

            try:
                output = torch.cat((output, output_batch), dim=0)
            except:
                output = output_batch
                
        # back to image coordinate system
        delta = output.cpu().detach().numpy() * (scale_of_axes)

        return landmarks_pre + delta, delta 



    def __init__(self, trainDataloader, validateDataloader, inferDataloader, **kwargs):
        """Initialize the learning model."""

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        self.trainDataLoader = trainDataloader
        self.valDataLoader = validateDataloader
        self.infDataLoader = inferDataloader

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.start_epoch = 0

        self.accumulated_loss = np.zeros((self.epochs, 3))
        self.accumulated_loss_val = np.zeros((self.epochs, 3))
        self.accumulated_loss_inf = np.zeros((self.epochs, 4))

        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )
        self.encoder_config = (
            kwargs["encoder_config"]
            if "encoder_config" in kwargs
            else self.args["encoder_config"]
        )
        self.modulation_activation_type = (
            kwargs["modulation_activation_type"]
            if "modulation_activation_type" in kwargs
            else self.args["modulation_activation_type"]
        )

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )

        # Parse other arguments from kwargs
        self.progress_file = (
            kwargs["progress_file"] if "progress_file" in kwargs else self.args["progress_file"]
        )
         # Plotting
        if self.progress_file is not None:
            self.plotter = utils.PlotRegTestTrainingProgress(self.progress_file)

        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )
        self.patch_size = kwargs["patch_size"] if 'patch_size' in kwargs else self.args["patch_size"]
        self.FM_mapping_type = kwargs["FM_mapping_type"] if "FM_mapping_type" in kwargs else self.args["FM_mapping_type"]
        self.FM_mapping_size = kwargs["FM_mapping_size"] if "FM_mapping_size" in kwargs else self.args["FM_mapping_size"]
        self.FM_sigma = kwargs["FM_sigma"] if "FM_sigma" in kwargs else self.args["FM_sigma"]
        
        self.encoder_type = kwargs["encoder_type"] if "encoder_type" in kwargs else self.args["encoder_type"]
        self.modulation_type = kwargs["modulation_type"] if "modulation_type" in kwargs else self.args["modulation_type"]
        self.encoder_config = kwargs["encoder_config"] if "encoder_config" in kwargs else self.args["encoder_config"]
        self.encoder_input_dim = self.encoder_config["input_dim"] if "input_dim" in self.encoder_config else self.args["encoder_input_dim"]
        self.encoder_loss = self.encoder_config["encoder_loss"] if "encoder_loss" in self.encoder_config else self.args["encoder_loss"]
        self.encoder_freeze = self.encoder_config["freeze_weights"] if "freeze_weights" in self.encoder_config else self.args["encoder_freeze"]

        if self.network_from_file is None:
            ##########################
            # Veronika, 11.2.2023
            # if self.network_type == "MLP":
            #     self.network = networks.MLP(self.layers)
            # else:
            #     self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            if self.network_type == "MLP":
                self.network = networks.MLP(self.layers)
            # elif self.network_type == "SIREN-N":
            #     self.layers[0] = self.layers[0]+2
            #     self.network = networksN.Siren(self.layers, self.weight_init, self.omega)
            elif self.network_type == 'SIREN-ENC':
                print()
                print('SIREN-ENC')
                #TODO check Omega
                mapping_size = 2*self.encoder_config["output_dim"] if self.encoder_config["input_dim"]==1 else self.encoder_config["output_dim"]
                mapping_size = mapping_size // 2 if self.modulation_type in ['localQAM', 'localQAMv2', 'localCrossAttentionAM', 'localCrossAttentionAMv2'] else mapping_size
                if self.modulation_type == 'SIREN+':
                    self.layers[0] += mapping_size
                encoder_size = self.patch_size if self.patch_size is not None else kwargs["image_shape"]
                self.network = networksEnc.Siren(encoder_size, self.layers, encoder_type=self.encoder_type, modulation_type=self.modulation_type, omega=self.omega,
                                                 encoder_config=self.encoder_config, mapping_size=mapping_size, modulation_activation_type=self.modulation_activation_type)
                # Load checkpoint for encoder
                if "checkpoint" in self.encoder_config:
                    if os.path.exists(self.encoder_config["checkpoint"]):
                        print()
                        print(f'=> loading model {self.encoder_config["checkpoint"]}')
                        checkpoint = torch.load(self.encoder_config["checkpoint"])
                        try:
                            self.network.encoder.load_state_dict(checkpoint['model_state_dict'])
                            print(f'=> loaded model {self.encoder_config["checkpoint"]}')
                        except Exception as e:
                            print(e)
                            self.network.encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
                            print(f'=> loaded model {self.encoder_config["checkpoint"]} with unexpected/missing keys')
                        
                    else:
                        print()
                        print(f'CAUTION: the model file {self.encoder_config["checkpoint"]} does not exist!')
                        print()
                
                # freeze the encoder weights
                if self.encoder_freeze:
                    print("Freeze encoder weights.")
                    self.freeze_weights_encoder()
                else:
                    print("DO NOT freeze encoder weights.")

            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            # print(self.network)
            ##########################
            if self.verbose:
                print(
                    "Network contains {} trainable parameters.".format(
                        general.count_parameters(self.network)
                    )
                )
        else:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()

        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )
        # TODO Freeze weights of encoder if we don't want to train it

        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

        # Loss for encoder if it is to be trained with the MLP
        if self.encoder_loss == 'L1':
            self.criterionEnc = nn.L1Loss()
        else:
            self.criterionEnc = nn.MSELoss()

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Parse arguments from kwargs
        # self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]
        self.use_mask = kwargs["use_mask"] if "use_mask" in kwargs else self.args["use_mask"]

        # Parse regularization kwargs
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.voxel_size = (
            kwargs["voxel_size"]
            if "voxel_size" in kwargs
            else self.args["voxel_size"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )

        self.transformation = None

        ##########################
        # Veronika, 06.2.2023
        self.possible_coordinate_tensor_general = general.make_coordinate_tensor(
                self.image_shape
        )
        self.possible_index_tensor_general = general.make_index_tensor(
                self.image_shape
        )
        ##########################

    def load_checkpoint(self, model_file):

        if not os.path.isfile(model_file):
            print(f'Provided checkpoint does not exist.')
            print(f'Model file: {model_file}')
        else:
            print(f"=> loading checkpoint '{model_file}'")
            print('loading checkpoint as dictionary')

            checkpoint = torch.load(model_file)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint["epoch"]
            self.accumulated_loss[:self.start_epoch,:] = checkpoint['acc_loss'][:self.start_epoch,:]
            self.accumulated_loss_val[:self.start_epoch,:] = checkpoint['acc_loss_val'][:self.start_epoch,:]
            self.accumulated_loss_inf[:self.start_epoch,:] = checkpoint['acc_loss_inf'][:self.start_epoch,:]

            print(f"=> loaded checkpoint '{model_file}' (epoch {self.start_epoch})")
                

    def freeze_weights_encoder(self):
        """Freeze the weights of the encoder."""
        if hasattr(self.network, "encoder"):
            print("Freeze the weights of the encoder")
            for param in self.network.encoder.parameters():
                param.requires_grad = False

    def cuda(self):
        """Move the model to the GPU."""

        # Standard variables
        self.network.cuda()

        # # Variables specific to this class
        # self.moving_image.cuda()
        # self.fixed_image.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        # self.args["mask"] = None
        # self.args["mask_2"] = None
        self.args["use_mask"] = False

        self.args["method"] = 1

        self.args["lr"] = 0.00001
        self.args["batch_size"] = 10000
        self.args["layers"] = [3, 256, 256, 256, 3]
        self.args["velocity_steps"] = 1

        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1

        self.args["jacobian_regularization"] = False
        self.args["alpha_jacobian"] = 0.05

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (200, 200, 200)
        self.args["voxel_size"] = [1.5, 1.5, 1.5]

        self.args["network"] = None

        self.args["epochs"] = 2500
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"
        self.args["progress_file"] = None

        self.args["network_type"] = "MLP"
        self.args["FM_mapping_type"] = "basic"
        self.args["FM_mapping_size"] = 16
        self.args["FM_sigma"] = 10
        self.args["patch_size"] = [8, 8, 8]
        self.args["modulation_activation_type"] = "relu"
        self.args["modulation_type"] = 'none'
        self.args["encoder_type"] = 'none'
        self.args["encoder_config"] = None
        self.args["encoder_input_dim"] = 1
        self.args["encoder_loss"] = 'L1'
        self.args["encoder_freeze"] = True

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1

    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        losses = [eval_utils.AverageMeter(), eval_utils.AverageMeter(), eval_utils.AverageMeter()]
        for i, (images, masks, _, idx) in enumerate(self.trainDataLoader):
            # print(i)

            # print(torch.min(images), torch.max(images))

            # if i>0:
            #     continue

            # print(idx)
            mask = masks[0,0,:].numpy() # fixed mask
            if self.gpu:
                images = images.cuda()

            #
            #   sample random coordinates
            #
            if self.use_mask:
                possible_coordinate_tensor = self.possible_coordinate_tensor_general[mask.flatten() > 0, :]
            else:
                possible_coordinate_tensor = self.possible_coordinate_tensor_general

            # random subset of coordinate tensors, number determined by self.batch_size
            indices = torch.randperm(
                possible_coordinate_tensor.shape[0], device="cuda"
            )[: self.batch_size]
            # indices = torch.arange(0,self.batch_size)

            coordinate_tensor = possible_coordinate_tensor[indices, :]
            coordinate_tensor = coordinate_tensor.requires_grad_(True)  

            if self.patch_size is not None:
                #
                #   sample random indices
                #
                if self.use_mask:
                    possible_index_tensor = self.possible_index_tensor_general[mask.flatten() > 0, :]
                else:
                    possible_coordinate_tensor = self.possible_coordinate_tensor_general
                    possible_index_tensor = self.possible_index_tensor_general

                index_tensor = possible_index_tensor[indices, :]

                #
                #   Extract patches with coordinates as patch centers
                #
                patches = general.sample_all_patch_pairs_3d(images[:,:1,:], images[:,1:,:], index_tensor, self.patch_size)
                
                # reshape patches, depending on number of input channels for encoder
                # input_dim=1 : encoding of fixed and moving patches individually
                # input_dim=1 : simultaneous encoding of image patches
                if self.encoder_input_dim==1:
                    patch_tensor = patches.view(2*self.batch_size,1,self.patch_size[0],self.patch_size[1],self.patch_size[2])
                else:
                    patch_tensor = patches.view(self.batch_size,2,self.patch_size[0],self.patch_size[1],self.patch_size[2])

                # patch_tensor = patch_tensor.requires_grad_(True)

                # print(coordinate_tensor.size())
                # print(patch_tensor.size())

                output = self.network(coordinate_tensor, patch_tensor)    

                # continue      
            else:
                # print(self.patch_size)

                # print(coordinate_tensor.size())
                # print(images.size())
                # images = images.permute(1,0,2,3,4)
                # print(images.size())

                output = self.network(coordinate_tensor, images.permute(1,0,2,3,4))

                # continue
            coord_temp = torch.add(output, coordinate_tensor)
            # output_add = coord_temp

            transformed_image = self.transform_no_add(coord_temp, moving_image=images[0,1,:])
            fixed_image = general.fast_trilinear_interpolation(
                # self.fixed_image,
                images[0,0,:],
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )

            # Compute the loss
            # loss += self.criterion(transformed_image, fixed_image)
            loss = self.criterion(transformed_image, fixed_image)

            # Store the value of the data loss
            if self.verbose:
                self.data_loss_list[epoch] = loss.detach().cpu().numpy()

            # Relativation of output
            # output_rel = torch.subtract(output, coordinate_tensor)
            output_rel = output

            # Regularization
            # print(self.alpha_bending)
            regloss = 0
            if self.jacobian_regularization:
                regloss = self.alpha_jacobian * regularizers.compute_jacobian_loss(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            if self.hyper_regularization:
                regloss = self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            if self.bending_regularization:
                regloss = self.alpha_bending * regularizers.compute_bending_energy(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            totalloss = loss + regloss
            losses[0].update(totalloss.item(), images.size(0))
            losses[1].update(loss.item(), images.size(0))
            # losses[2].update(regloss.item(), images.size(0))

            # # add the autoencoder loss if so desired
            # if not self.encoder_freeze:
            #     outputEnc = self.network.encoder(patch_tensor)
            #     totalloss += self.criterionEnc(outputEnc, patch_tensor)

            # Perform the backpropagation and update the parameters accordingly
            for param in self.network.parameters():
                param.grad = None
            totalloss.backward()
            self.optimizer.step()
            # self.optimizer.zero_grad()

        # Store the value of the total loss
        # if self.verbose:
            # self.loss_list[epoch] = loss.detach().cpu().numpy()

        return [l.avg for l in losses]

    def validation_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.eval()

        losses = [eval_utils.AverageMeter(), eval_utils.AverageMeter(), eval_utils.AverageMeter()]

        for i, (images, masks, _, idx) in enumerate(self.valDataLoader):

            # print(torch.min(images), torch.max(images))

            # if i>0:
            #     continue

            mask = masks[0,0,:].numpy() # fixed mask
            if self.gpu:
                images = images.cuda()

            #
            #   sample random coordinates
            #
            if self.use_mask:
                possible_coordinate_tensor = self.possible_coordinate_tensor_general[mask.flatten() > 0, :]

            else:
                possible_coordinate_tensor = self.possible_coordinate_tensor_general

            # random subset of coordinate tensors, number determined by self.batch_size
            indices = torch.randperm(
                possible_coordinate_tensor.shape[0], device="cuda"
            )[: self.batch_size]
            coordinate_tensor = possible_coordinate_tensor[indices, :]
            coordinate_tensor = coordinate_tensor.requires_grad_(True)

            if self.patch_size is not None:
                #
                #   sample random coordinates
                #
                if self.use_mask:
                    possible_index_tensor = self.possible_index_tensor_general[mask.flatten() > 0, :]

                else:
                    possible_index_tensor = self.possible_index_tensor_general

                index_tensor = possible_index_tensor[indices, :]
                #
                #   Extract patches with coordinates as patch centers
                #
                patches = general.sample_all_patch_pairs_3d(images[:,:1,:], images[:,1:,:], index_tensor, self.patch_size)

                # reshape patches, depending on number of input channels for encoder
                # input_dim=1 : encoding of fixed and moving patches individually
                # input_dim=1 : simultaneous encoding of image patches
                if self.encoder_input_dim==1:
                    patch_tensor = patches.view(2*self.batch_size,1,self.patch_size[0],self.patch_size[1],self.patch_size[2])
                else:
                    patch_tensor = patches.view(self.batch_size,2,self.patch_size[0],self.patch_size[1],self.patch_size[2])

                # patch_tensor = patch_tensor.requires_grad_(True)

                output = self.network(coordinate_tensor, patch_tensor) 

            else:
                # images = images.permute(1,0,2,3,4)

                output = self.network(coordinate_tensor, images.permute(1,0,2,3,4))

            coord_temp = torch.add(output, coordinate_tensor)
            # output_add = coord_temp

            transformed_image = self.transform_no_add(coord_temp, moving_image=images[0,1,:])
            fixed_image = general.fast_trilinear_interpolation(
                # self.fixed_image,
                images[0,0,:],
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )

            # Compute the loss
            # loss += self.criterion(transformed_image, fixed_image)
            loss = self.criterion(transformed_image, fixed_image)

            # Store the value of the data loss
            if self.verbose:
                self.data_loss_list[epoch] = loss.detach().cpu().numpy()

            # Relativation of output
            # output_rel = torch.subtract(output, coordinate_tensor)
            output_rel = output

            # Regularization
            regloss = 0
            if self.jacobian_regularization:
                regloss = self.alpha_jacobian * regularizers.compute_jacobian_loss(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            if self.hyper_regularization:
                regloss = self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            if self.bending_regularization:
                regloss = self.alpha_bending * regularizers.compute_bending_energy(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            totalloss = loss + regloss
            losses[0].update(totalloss.item(), images.size(0))
            losses[1].update(loss.item(), images.size(0))
            # losses[2].update(regloss.item(), images.size(0))

        # Store the value of the total loss
        # if self.verbose:
            # self.loss_list[epoch] = loss.detach().cpu().numpy()
        return [l.avg for l in losses]


    def test_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.eval()

        losses = [eval_utils.AverageMeter(), eval_utils.AverageMeter(), eval_utils.AverageMeter(), eval_utils.AverageMeter()]

        for i, (images, masks, landmarks, idx) in enumerate(self.infDataLoader):

            # continue

            mask = masks[0,0,:].numpy() # fixed mask
            if self.gpu:
                images = images.cuda()

            #
            #   sample random coordinates
            #
            if self.use_mask:
                possible_coordinate_tensor = self.possible_coordinate_tensor_general[mask.flatten() > 0, :]

            else:
                possible_coordinate_tensor = self.possible_coordinate_tensor_general

            # random subset of coordinate tensors, number determined by self.batch_size
            indices = torch.randperm(
                possible_coordinate_tensor.shape[0], device="cuda"
            )[: self.batch_size]
            coordinate_tensor = possible_coordinate_tensor[indices, :]
            coordinate_tensor = coordinate_tensor.requires_grad_(True)

            if self.patch_size is not None:
                #
                #   sample random coordinates
                #
                if self.use_mask:
                    possible_index_tensor = self.possible_index_tensor_general[mask.flatten() > 0, :]

                else:
                    possible_index_tensor = self.possible_index_tensor_general

                index_tensor = possible_index_tensor[indices, :]
                #
                #   Extract patches with coordinates as patch centers
                #
                patches = general.sample_all_patch_pairs_3d(images[:,:1,:], images[:,1:,:], index_tensor, self.patch_size)

                # reshape patches, depending on number of input channels for encoder
                # input_dim=1 : encoding of fixed and moving patches individually
                # input_dim=1 : simultaneous encoding of image patches
                if self.encoder_input_dim==1:
                    patch_tensor = patches.view(2*self.batch_size,1,self.patch_size[0],self.patch_size[1],self.patch_size[2])
                else:
                    patch_tensor = patches.view(self.batch_size,2,self.patch_size[0],self.patch_size[1],self.patch_size[2])

                # patch_tensor = patch_tensor.requires_grad_(True)

                output = self.network(coordinate_tensor, patch_tensor) 

            else:
                # images = images.permute(1,0,2,3,4)

                output = self.network(coordinate_tensor, images.permute(1,0,2,3,4))

            coord_temp = torch.add(output, coordinate_tensor)
            # output_add = coord_temp

            transformed_image = self.transform_no_add(coord_temp, moving_image=images[0,1,:])
            fixed_image = general.fast_trilinear_interpolation(
                # self.fixed_image,
                images[0,0,:],
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )

            # Compute the loss
            # loss += self.criterion(transformed_image, fixed_image)
            loss = self.criterion(transformed_image, fixed_image)

            # Store the value of the data loss
            if self.verbose:
                self.data_loss_list[epoch] = loss.detach().cpu().numpy()

            # Relativation of output
            # output_rel = torch.subtract(output, coordinate_tensor)
            output_rel = output

            # Regularization
            regloss = 0
            if self.jacobian_regularization:
                regloss = self.alpha_jacobian * regularizers.compute_jacobian_loss(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            if self.hyper_regularization:
                regloss = self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            if self.bending_regularization:
                regloss = self.alpha_bending * regularizers.compute_bending_energy(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
                losses[2].update(regloss.item(), images.size(0))
            totalloss = loss + regloss
            losses[0].update(totalloss.item(), images.size(0))
            losses[1].update(loss.item(), images.size(0))
            # losses[2].update(regloss.item(), images.size(0))

        
            # Compute the landmark error
            fixed_landmarks = landmarks[0][0]
            moving_landmarks = landmarks[0][1]

            tre_before = eval_utils.compute_tre(fixed_landmarks.numpy(), moving_landmarks.numpy(), self.voxel_size, self.voxel_size, fix_lms_warped=fixed_landmarks.numpy(), disp=None).mean()
            registered_landmarks, _ = self.compute_landmarks(fixed_landmarks.numpy(), self.image_shape, 
                                            moving_image=images[0,1,...], fixed_image=images[0,0,...], verbose=False)

            tre = eval_utils.compute_tre(fixed_landmarks.numpy(), moving_landmarks.numpy(), self.voxel_size, self.voxel_size, fix_lms_warped=registered_landmarks, disp=None).mean()
            losses[3].update(tre, images.size(0))


        return [l.avg for l in losses]


    def transform(
        self, transformation, moving_image, coordinate_tensor=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor

        # If no moving image is given use the standard one
        # if moving_image is None:
            # moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def transform_no_add(self, transformation, moving_image, reshape=False):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        # if moving_image is None:
            # moving_image = self.moving_image
        # print('GET MOVING')
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def fit(self, epochs=None, red_blue=False):
        """Train the network."""

        best_loss = 1e10

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # # Extend lost_list if necessary
        # if not len(self.loss_list) == epochs:
        #     self.loss_list = [0 for _ in range(epochs)]
        #     self.data_loss_list = [0 for _ in range(epochs)]

        # Perform training iterations
        for i in tqdm.tqdm(range(self.start_epoch, epochs)):
            try:
            #if 1:
                losses = self.training_iteration(i)
                self.accumulated_loss[i, :] = losses

                ##########################
                # Veronika, 17.2.2023

                if self.valDataLoader is not None:
                    losses_val = self.validation_iteration(i)
                    self.accumulated_loss_val[i, :] = losses_val
                
                    if losses_val[0]< best_loss:
                        best_loss = losses_val[0]

                        val_model_file = f'{self.save_folder}/model-best.pt'

                        torch.save({
                            'epoch': i,
                            'model_state_dict': self.network.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'network_type': self.network_type,
                            'optimizer': self.optimizer_arg.lower(),
                            'loss': self.loss_function_arg.lower(),
                            'regularizer': 'hyper' if self.args['hyper_regularization'] 
                                            else 'jacobian' if self.args['jacobian_regularization'] 
                                            else 'bending',
                            'acc_loss': self.accumulated_loss,
                            'acc_loss_val': self.accumulated_loss_val,
                            'acc_loss_inf': self.accumulated_loss_inf,
                            "best_loss": best_loss
                            }, val_model_file)
                        
                if self.infDataLoader is not None:
                    losses_inf = self.test_iteration(i)
                    self.accumulated_loss_inf[i, :] = losses_inf
                        
                if self.progress_file is not None:
                    self.plotter(self.accumulated_loss[:i,:], self.accumulated_loss_val[:i,:], self.accumulated_loss_inf[:i,:], i)
                    
                if i % 100 == 0:
                    model_file = f'{self.save_folder}/checkpoint-epochs{i}.pt'

                    torch.save({
                        'epoch': i,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'network_type': self.network_type,
                        'optimizer': self.optimizer_arg.lower(),
                        'loss': self.loss_function_arg.lower(),
                        'regularizer': 'hyper' if self.args['hyper_regularization'] 
                                        else 'jacobian' if self.args['jacobian_regularization'] 
                                        else 'bending',
                        'acc_loss': self.accumulated_loss,
                        'acc_loss_val': self.accumulated_loss_val,
                        'acc_loss_inf': self.accumulated_loss_inf
                        }, model_file)

            except RuntimeError as e:
                print('skip epoch', i, e)
                
        model_file = f'{self.save_folder}/checkpoint-epochs{epochs}.pt'

        torch.save({
            'epoch': epochs,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'network_type': self.network_type,
            'optimizer': self.optimizer_arg.lower(),
            'loss': self.loss_function_arg.lower(),
            'regularizer': 'hyper' if self.args['hyper_regularization'] 
                            else 'jacobian' if self.args['jacobian_regularization'] 
                            else 'bending',
            'acc_loss': self.accumulated_loss,
            'acc_loss_val': self.accumulated_loss_val,
            'acc_loss_inf': self.accumulated_loss_inf
            }, model_file)
        ##########################

    ##########################
    # Veronika, 26.1.2023

    def infer(self, model_file=None):

        if model_file is None:
            model_file = f'{self.save_folder}/model-best.pt'

        checkpoint = torch.load(model_file)
        self.network.load_state_dict(checkpoint['model_state_dict'])

        self.network.eval()

    ##########################

    # ##########################
    # # Veronika, 17.2.2023

    # def test_and_save(self, out_dir=None, model_file=None):

    #     self.infer(model_file)

    #     for i, (images, masks, keypoints, idx) in enumerate(self.infDataLoader):

    #         if i

    #         case_dir = f'{out_dir}/{idx}'
    #         if not os.path.exists(case_dir):
    #             os.makedirs(case_dir)

    #         # Compute landmarks
    #         registered_landmarks, _ = general.compute_landmarks(self.network, keypoints[0,0,:], image_size=self.image_shape,
    #                                                             moving_image=images[0,1,:], fixed_image=images[0,0,:])
            
    #         print(registered_landmarks)

        

    # ##########################