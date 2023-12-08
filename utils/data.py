import numpy as np
import os
import torch
import json
import SimpleITK as sitk
import nibabel as nib
from numpy import genfromtxt


def load_image_L2R_NLST(id, folder, mode='train'):

    # info_file = f'{folder}/NLST/NLST_dataset.json'
    # if mode == 'test':
    #     info_file = f'{folder}/NLST__testdata/NLST_dataset.json'
    # with open(info_file) as json_file:
    #     info = json.load(json_file)

    # filenames = info['training_paired_images']
    subfolder = 'NLST' if mode=='train' else 'NLST_testdata'

    # images
    fixed_file = f'{folder}/{subfolder}/imagesTr/NLST_{str(id).zfill(4)}_0000.nii.gz'
    moving_file = f'{folder}/{subfolder}/imagesTr/NLST_{str(id).zfill(4)}_0001.nii.gz'

    # masks
    fixed_mask_file = f'{folder}/{subfolder}/masksTr/NLST_{str(id).zfill(4)}_0000.nii.gz'
    moving_mask_file = f'{folder}/{subfolder}/masksTr/NLST_{str(id).zfill(4)}_0001.nii.gz'

    # landmarks
    fixed_lm_file = f'{folder}/{subfolder}/keypointsTr/NLST_{str(id).zfill(4)}_0000.csv'
    moving_lm_file = f'{folder}/{subfolder}/keypointsTr/NLST_{str(id).zfill(4)}_0001.csv'
    
    # fixed = sitk.ReadImage(fixed_file)
    # moving = sitk.ReadImage(moving_file)
    # fixed_mask = sitk.ReadImage(fixed_mask_file)
    # moving_mask = sitk.ReadImage(moving_mask_file)
    fixed = nib.load(fixed_file)
    moving = nib.load(moving_file)
    fixed_mask = nib.load(fixed_mask_file)
    moving_mask = nib.load(moving_mask_file)
    

    # to tensor
    fixed = torch.FloatTensor(fixed.get_data())
    moving = torch.FloatTensor(moving.get_data())
    # fixed = torch.FloatTensor(sitk.GetArrayFromImage(fixed))
    # moving = torch.FloatTensor(sitk.GetArrayFromImage(moving))
    # fixed = torch.FloatTensor(np.flip(sitk.GetArrayFromImage(fixed),0).copy())
    # moving = torch.FloatTensor(np.flip(sitk.GetArrayFromImage(moving),0).copy())

    fixed_mask = np.clip(fixed_mask.get_data(), 0, 1)
    moving_mask = np.clip(moving_mask.get_data(), 0, 1)
    # fixed_mask = np.clip(sitk.GetArrayFromImage(fixed_mask), 0, 1)
    # moving_mask = np.clip(sitk.GetArrayFromImage(moving_mask), 0, 1)
    # fixed_mask = np.clip(np.flip(sitk.GetArrayFromImage(fixed_mask),0), 0, 1)
    # moving_mask = np.clip(np.flip(sitk.GetArrayFromImage(moving_mask),0), 0, 1)


    fixed_landmarks = genfromtxt(fixed_lm_file, delimiter=',')
    moving_landmarks = genfromtxt(moving_lm_file, delimiter=',')

    return (
            fixed,
            moving,
            fixed_landmarks,
            moving_landmarks,
            fixed_mask,
            moving_mask
        )

def load_image_DIRLab_4DCT(variation=1, folder=r"D:\Data\DIRLAB\Case"):
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

def load_image_DIRLab_COPD(variation=1, folder=r"D:\Data\DIRLAB\COPD\Case"):
     # Size of data, per image pair
    image_sizes = [
        0,
        [121, 512, 512],
        [102, 512, 512],
        [126, 512, 512],
        [126, 512, 512],
        [131, 512, 512],
        [119, 512, 512],
        [112, 512, 512],
        [115, 512, 512],
        [116, 512, 512],
        [135, 512, 512],
    ]

    # Scale of data, per image pair
    # voxel_sizes = [
    #     0,
    #     [0.625, 0.625, 2.5],
    #     [0.645, 0.645, 2.5],
    #     [0.652, 0.652, 2.5],
    #     [0.590, 0.590, 2.5],
    #     [0.647, 0.647, 2.5],
    #     [0.633, 0.633, 2.5],
    #     [0.625, 0.625, 2.5],
    #     [0.586, 0.586, 2.5],
    #     [0.664, 0.664, 2.5],
    #     [0.742, 0.742, 2.5],
    # ]
    voxel_sizes = [
        0,
        [2.5, 0.625, 0.625],
        [2.5, 0.645, 0.645],
        [2.5, 0.652, 0.652],
        [2.5, 0.590, 0.590],
        [2.5, 0.647, 0.647],
        [2.5, 0.633, 0.633],
        [2.5, 0.625, 0.625],
        [2.5, 0.586, 0.586],
        [2.5, 0.664, 0.664],
        [2.5, 0.742, 0.742],
    ]

    shape = image_sizes[variation]
    spacing = voxel_sizes[variation]

    # Images
    dtype = np.dtype(np.int16)

    # Inspiration
    with open(f'{folder}/copd{variation}/copd{variation}_iBHCT.img', "rb") as f:
        data = np.fromfile(f, dtype)
    image_insp = data.reshape(shape)

    # Expiration
    with open(f'{folder}/copd{variation}/copd{variation}_eBHCT.img', "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

    # Load lung mask (fixed image)
    imgsitk_in = sitk.ReadImage(f'{folder}/copd{variation}/copd{variation}_iBHCT_mask.mhd')
    mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)

    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Landmarks
    iLM_file = f'{folder}/copd{variation}/copd{variation}_300_iBH_xyz_r1.txt'
    with open(iLM_file) as f:
        landmarks_insp = np.array(
            [list(map(float, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    eLM_file = f'{folder}/copd{variation}/copd{variation}_300_eBH_xyz_r1.txt'
    with open(eLM_file) as f:
        landmarks_exp = np.array(
            [list(map(float, line[:-1].split("\t")[:3])) for line in f.readlines()]
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