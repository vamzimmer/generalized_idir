import numpy as np
import os
import torch
import SimpleITK as sitk

#
#   Original code taken from 
#   Wolterink et al.: Implicit Neural Representations for Deformable Image Registration, MIDL 2022
#   https://github.com/MIAGroupUT/IDIR
#


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def compute_landmarks(network, landmarks_pre, image_size, moving_image=None, fixed_image=None, encoder=False):
    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

    if moving_image is None:
        output = network(coordinate_tensor.cuda())
    elif encoder:
        encoder_input = torch.concat([fixed_image.view(1, 1, *fixed_image.shape),
                                        moving_image.view(1, 1, *moving_image.shape),],
                                        axis=1
                                        ).cuda()
        output = network(coordinate_tensor.cuda(), encoder_input)
    else:
        # get intensities of fixed and moving images
            fixed = fast_trilinear_interpolation(
                fixed_image,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )
            moving = fast_trilinear_interpolation(
                moving_image,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )
            intensities = torch.cat((fixed.unsqueeze(1), moving.unsqueeze(1)),1)
            output = network(coordinate_tensor.cuda(), intensities.cuda())

    delta = output.cpu().detach().numpy() * (scale_of_axes)

    return landmarks_pre + delta, delta


# def compute_landmarks(network, landmarks_pre, image_size):
#     scale_of_axes = [(0.5 * s) for s in image_size]

#     coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

#     output = network(coordinate_tensor.cuda())

#     delta = output.cpu().detach().numpy() * (scale_of_axes)

#     return landmarks_pre + delta, delta


def load_image_DIRLab(variation=1, folder=r"D:\Data\DIRLAB\Case"):
    # Size of data, per image pair
    image_sizes = [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    # Scale of data, per image pair
    voxel_sizes = [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    shape = image_sizes[variation]

    folder = folder + str(variation) + r"Pack" + os.path.sep

    # Images
    dtype = np.dtype(np.int16)

    with open(folder + r"Images/case" + str(variation) + "_T00_s.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_insp = data.reshape(shape)

    with open(folder + r"Images/case" + str(variation) + "_T50_s.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

    imgsitk_in = sitk.ReadImage(folder + r"Masks/case" + str(variation) + "_T00_s.mhd")

    mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)

    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Landmarks
    with open(
        folder + r"ExtremePhases/Case" + str(variation) + "_300_T00_xyz.txt"
    ) as f:
        landmarks_insp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    with open(
        folder + r"ExtremePhases/Case" + str(variation) + "_300_T50_xyz.txt"
    ) as f:
        landmarks_exp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    landmarks_insp[:, [0, 2]] = landmarks_insp[:, [2, 0]]
    landmarks_exp[:, [0, 2]] = landmarks_exp[:, [2, 0]]

    return (
        image_insp,
        image_exp,
        landmarks_insp,
        landmarks_exp,
        mask,
        voxel_sizes[variation],
    )


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output

def fast_nearest_interpolation(input_array, x_indices, y_indices, z_indices):

    # voxel displacement in image coordinate system
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    # get neighboring voxels in all three dimensions
    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # border handling
    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    # new position relative to neighboring voxels
    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    # nearest neighbor interpolation
    xN = torch.where(x>0.5, x1, x0)
    yN = torch.where(y>0.5, y1, y0)
    zN = torch.where(z>0.5, z1, z0)

    output = input_array[xN, yN, zN]
    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_slice(dims=(28, 28), dimension=0, slice_pos=0, gpu=True):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims.insert(dimension, 1)

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    if gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_masked_coordinate_tensor(mask, dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    if gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

##################################
# Veronika (20.02.2023)
def make_index_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(0, dims[i]-1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    if gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor



def sample_all_patch_pairs_3d(image1, image2, patch_centers, patch_size=(8,8,8)):

    """
        @Vasiliki: PatchLoss
    """

    sh = image1.shape #data_dict['fixed'].shape
    # print(sh)

    fixed_patches = []
    moving_patches = []
    
    # # make sure that the patch is within the image borders
    # start_x = -1
    # start_y = -1
    # start_z = -1
    # end_x = 1000
    # end_y = 1000
    # end_z = 1000

    n_patches = patch_centers.size()[0] #data_dict['fixed_centres'].size()[0]
    # print(patch_centers.size())
    # print(n_patches)
    for idx in range(n_patches):
        # print(idx)

        # # make sure that the patch is within the image borders
        # start_x = -1
        # start_y = -1
        # start_z = -1
        # end_x = 1000
        # end_y = 1000
        # end_z = 1000

        # sampling a centre, while the patch is not within the image boarders, sample another centre
        # count = 0
        # while (start_x < 0 or start_y < 0 or start_z < 0 or end_x>sh[-3] or end_y > sh[-2] or end_z > sh[-1]):

        #     if count>2:
        #         break

        #     print(start_x<0, start_y<0, start_z<0)
        #     print(end_x>sh[-3], end_y > sh[-2], end_z > sh[-1])
        #     print(start_x, start_y, start_z)
        #     print(end_x, end_y, end_z)

        #     # # here I choose one index at random, but this can be an argument of the function 
        #     # if idx is None:
        #     #     idx = np.random.randint(0, data_dict['fixed_centres'].shape[0])
            
        # the fixed centres 
        patch_centre = patch_centers[idx,:]# data_dict['fixed_centres'][idx]
        # print()
        # print(sh)
        # print(patch_centre)
        # print()

        cy, cx, cz = int(patch_centre[0].item()), int(patch_centre[1].item()), int(patch_centre[2].item())

        start_x = cx - patch_size[0] // 2
        end_x = cx + patch_size[0] // 2

        start_y = cy - patch_size[1] // 2
        end_y = cy + patch_size[1] // 2

        start_z = cz - patch_size[2] // 2
        end_z = cz + patch_size[2] // 2

        # count += 1

        # print(start_x, start_y, start_z)
        # print(end_x, end_y, end_z)

        # correct the index of the patch is not completely inside the image.
        if start_x<0: 
            end_x -= start_x
            start_x = 0
        if end_x > sh[-3]:
            start_x -= (end_x-sh[-3])
            end_x = sh[-3]
        if start_y<0: 
            end_y -= start_y
            start_y = 0
        if end_y > sh[-2]:
            start_y -= (end_y-sh[-2])
            end_y = sh[-2]
        if start_z<0: 
            end_z -= start_z
            start_z = 0
        if end_z > sh[-1]:
            start_z -= (end_z-sh[-1])
            end_z = sh[-1]

        # print(start_x, start_y, start_z)
        # print(end_x, end_y, end_z)

        # moving_patch = data_dict['moving'][:, :, start_y:end_y, start_x:end_x, start_z:end_z]
        # fixed_patch = data_dict['fixed'][:, :, start_y:end_y, start_x:end_x, start_z:end_z]
        # moving_patch = image2[:, :, start_y:end_y, start_x:end_x, start_z:end_z]
        # fixed_patch = image1[:, :, start_y:end_y, start_x:end_x, start_z:end_z]
        moving_patch = image2[:, :, start_x:end_x, start_y:end_y, start_z:end_z]
        fixed_patch = image1[:, :, start_x:end_x, start_y:end_y, start_z:end_z]

        # print(fixed_patch.size())
        # print(moving_patch.size())

        fixed_patches.append(fixed_patch.flatten())
        moving_patches.append(moving_patch.flatten())

    fixed_patches = torch.stack(fixed_patches)
    moving_patches = torch.stack(moving_patches)
    patches = torch.stack([fixed_patches, moving_patches])

    return patches


##################################