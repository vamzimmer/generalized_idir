import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
import numpy as np

from utils import general
from networks import networks
from objectives import ncc
from objectives import regularizers
from models import utils
from utils import eval_utils

#
#   Original code taken from 
#   Wolterink et al.: Implicit Neural Representations for Deformable Image Registration, MIDL 2022
#   https://github.com/MIAGroupUT/IDIR
#

class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, coordinate_tensor=None, output_shape=(28, 28), dimension=0, slice_pos=0, batch_size=1000000
    ):
        """Return the image-values for the given input-coordinates."""

        # Use standard coordinate tensor if none is given
        if coordinate_tensor is None:
            coordinate_tensor = self.make_coordinate_slice(
                output_shape, dimension, slice_pos
            )

        ##########################
        # Veronika, 02.03.2023
        coordinate_batches = torch.tensor_split(coordinate_tensor, coordinate_tensor.size()[0]//batch_size+1 ,dim=0)

        print(f"Batchwise ({len(coordinate_batches)}) displacement prediction...")
        for b in tqdm.tqdm(range(len(coordinate_batches))):

            output_batch = self.network(coordinate_batches[b])

            try:
                output = torch.cat((output, output_batch.cpu().detach()), dim=0)
            except:
                output = output_batch.cpu().detach()

        # output = self.network(coordinate_tensor)

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(output.cuda(), coordinate_tensor)
        self.displacement = output
        self.transformation = coord_temp

        transformed_image = self.transform_no_add(coord_temp)

        ##########################
        # Veronika, 26.1.2023
        # return (
        #         transformed_image.cpu()
        #         .detach()
        #         .numpy()
        #         .reshape(output_shape[0], output_shape[1])
        #     )
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
        ##########################

    def compute_landmarks(self, landmarks_pre, image_size):

        scale_of_axes = [(0.5 * s) for s in image_size]
        coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0
        output = self.network(coordinate_tensor.cuda())
        delta = output.cpu().detach().numpy() * (scale_of_axes)

        return landmarks_pre + delta, delta

    def __init__(self, moving_image, fixed_image, moving_landmarks=None, fixed_landmarks=None, **kwargs):
        """Initialize the learning model."""

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
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

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )

        ##################
        # Veronika 3.3.2023

        # Parse other arguments from kwargs
        self.progress_file = (
            kwargs["progress_file"] if "progress_file" in kwargs else self.args["progress_file"]
        )
         # Plotting
        if self.progress_file is not None:
            # self.plotter = utils.PlotTrainingProgress(self.progress_file)
            self.plotter = utils.PlotRegTestTrainingProgress(self.progress_file)

        ##################

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
        if self.network_from_file is None:
            if self.network_type == "MLP":
                self.network = networks.MLP(self.layers)
            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)
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
        print(self.network)

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

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Parse arguments from kwargs
        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]

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
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )

        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image

        self.moving_landmarks = moving_landmarks
        self.fixed_landmarks = fixed_landmarks

        self.transformation = None

        ##########################
        # Veronika, 06.2.2023

        # self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(
        #         self.mask, self.fixed_image.shape
        #     )
        if self.mask is not None:
            self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(
                self.mask, self.fixed_image.shape
            )
        else:
            self.possible_coordinate_tensor = general.make_coordinate_tensor(
                self.fixed_image.shape
            )
        ##########################

        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()

    def cuda(self):
        """Move the model to the GPU."""

        # Standard variables
        self.network.cuda()

        # Variables specific to this class
        self.moving_image.cuda()
        self.fixed_image.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None

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

        self.args["image_shape"] = (200, 200)

        self.args["network"] = None

        self.args["epochs"] = 2500
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"
        ##################
        # Veronika 3.3.2023
        self.args["progress_file"] = None
        ##################

        self.args["network_type"] = "MLP"

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

        losses = [0,0,0,0]

        loss = 0
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], device="cuda"
        )[: self.batch_size]
        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)

        output = self.network(coordinate_tensor)
        coord_temp = torch.add(output, coordinate_tensor)
        output = coord_temp

        transformed_image = self.transform_no_add(coord_temp)
        fixed_image = general.fast_trilinear_interpolation(
            self.fixed_image,
            coordinate_tensor[:, 0],
            coordinate_tensor[:, 1],
            coordinate_tensor[:, 2],
        )
        # Compute the loss
        loss += self.criterion(transformed_image, fixed_image)

        # Store the value of the data loss
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        # Relativation of output
        output_rel = torch.subtract(output, coordinate_tensor)

        # Regularization
        regloss = 0
        if self.jacobian_regularization:
            regloss = self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.hyper_regularization:
            regloss = self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.bending_regularization:
            regloss = self.alpha_bending * regularizers.compute_bending_energy(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        totalloss = loss + regloss
        losses[0] = totalloss.item()
        losses[1] = loss.item()
        losses[2] = regloss.item()


        # Perform the backpropagation and update the parameters accordingly
        for param in self.network.parameters():
            param.grad = None
        # loss.backward()
        totalloss.backward()
        self.optimizer.step()

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = totalloss.detach().cpu().numpy()
            # self.loss_list[epoch] = loss.detach().cpu().numpy()


        if self.moving_landmarks is not None and self.fixed_landmarks is not None:
            # Compute the landmark error

            tre_before = eval_utils.compute_tre(self.fixed_landmarks, self.moving_landmarks, [1.5,1.5,1.5], [1.5,1.5,1.5], fix_lms_warped=self.fixed_landmarks, disp=None).mean()
            registered_landmarks, _ = self.compute_landmarks(self.fixed_landmarks, self.moving_image.shape)

            tre = eval_utils.compute_tre(self.fixed_landmarks, self.moving_landmarks, [1.5,1.5,1.5], [1.5,1.5,1.5], fix_lms_warped=registered_landmarks, disp=None).mean()
            losses[3] = tre

            # print(tre_before)
            # print(tre)

        return losses


    def transform(
        self, transformation, coordinate_tensor=None, moving_image=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def transform_no_add(self, transformation, moving_image=None, reshape=False, interp='linear'):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image
        # print('GET MOVING')
        if interp=='linear':
            return general.fast_trilinear_interpolation(
                moving_image,
                transformation[:, 0],
                transformation[:, 1],
                transformation[:, 2],
            )
        elif interp=='nearest':
            return general.fast_nearest_interpolation(
                moving_image,
                transformation[:, 0],
                transformation[:, 1],
                transformation[:, 2],
            )

    def fit(self, epochs=None, red_blue=False):
        """Train the network."""

        self.accumulated_loss = np.zeros((self.epochs, 3))
        self.accumulated_loss_val = np.zeros((self.epochs, 3))
        self.accumulated_loss_inf = np.zeros((self.epochs, 4))

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]

        # Perform training iterations
        for i in tqdm.tqdm(range(epochs)):
            losses = self.training_iteration(i)

            self.accumulated_loss[i, :] = losses[:-1]
            self.accumulated_loss_inf[i,-1] = losses[-1]

            ##################
            # Veronika 3.3.2023 
            if self.progress_file is not None:
                # self.plotter(self.data_loss_list, self.loss_list, i)
                self.plotter(self.accumulated_loss[:i,:], self.accumulated_loss_val[:i,:], self.accumulated_loss_inf[:i,:], i)
            ##################


        ##########################
        # Veronika, 26.1.2023

        model_file = f'{self.save_folder}/model-inr-epochs{epochs}.pt'
        print(model_file)

        torch.save({
            'epoch': epochs,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'network_type': self.network_type,
            'optimizer': self.optimizer_arg.lower(),
            'loss': self.loss_function_arg.lower(),
            'regularizer': 'hyper' if self.args['hyper_regularization'] 
                            else 'jacobian' if self.args['jacobian_regularization'] 
                            else 'bending'
            }, model_file)
        ##########################

    ##########################
    # Veronika, 26.1.2023

    def infer(self, model_file=None):

        if model_file is None:
            model_file = f'{self.save_folder}/model-inr-epochs{self.epochs}.pt'

        checkpoint = torch.load(model_file)
        self.network.load_state_dict(checkpoint['model_state_dict'])

        self.network.eval()

    ##########################