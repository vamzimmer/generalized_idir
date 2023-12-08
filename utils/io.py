import os
import numpy as np
import SimpleITK as sitk
from glob import glob
import pandas as pd
from openpyxl import load_workbook


def save_to_excel(names, values, measures, sheet_name, outfile):
    # save volumes in excel file
    if os.path.isfile(outfile):
        book = load_workbook(outfile)
        writer = pd.ExcelWriter(outfile, engine='openpyxl')
        writer.book = book

    N, M = values.shape

    if N > 1:
        d = {'ids': range(0, N)}
    else:
        d = {'ids': range(0, N - 2)}
    df3 = pd.DataFrame(data=d)

    df3['name'] = names

    for m in range(0, len(measures)):
        for j in range(0, N):
            if np.isinf(values[j, m]):
                values[j, m] = -1
        df3[measures[m]] = values[:, m].tolist()

    if os.path.isfile(outfile):
        df3.to_excel(writer, sheet_name=sheet_name)
        writer.save()
        writer.close()
    else:
        df3.to_excel(outfile, sheet_name=sheet_name)


def load_image_DIRLab_as_mhd(variation=1, folder='/home/veronika/Data/Reg/CRC/DIRLab/4DCT'):
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

    voxel_sizes = [
        0,
        [0.97, 0.97, 2.5],
        [1.16, 1.16, 2.5],
        [1.15, 1.15, 2.5],
        [1.13, 1.13, 2.5],
        [1.1, 1.1, 2.5],
        [0.97, 0.97, 2.5],
        [0.97, 0.97, 2.5],
        [0.97, 0.97, 2.5],
        [0.97, 0.97, 2.5],
        [0.97, 0.97, 2.5],
    ]

    shape = image_sizes[variation]
    spacing = voxel_sizes[variation]

    # Images
    dtype = np.dtype(np.int16)
    # dtype = np.dtype(np.float32)

    file_insp = glob(f"{folder}/Case{variation}Pack/Images/**T00**.img", recursive=True)[0]
    file_exp = glob(f"{folder}/Case{variation}Pack/Images/**T50**.img", recursive=True)[0]

    # Inspiration
    with open(file_insp, "rb") as f:
        data = np.fromfile(f, dtype)
    image_insp = data.reshape(shape)

    img_insp = sitk.GetImageFromArray(image_insp)
    img_insp.SetSpacing(spacing)
    img_insp = sitk.Cast(img_insp, sitk.sitkFloat32)
    sitk.WriteImage(img_insp, f'{folder}/Case{variation}Pack/Images/case{variation}_T00_s.mhd')

    # Expiration
    with open(file_exp, "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

    img_exp = sitk.GetImageFromArray(image_exp)
    img_exp.SetSpacing(spacing)
    sitk.WriteImage(img_exp, f'{folder}/Case{variation}Pack/Images/case{variation}_T50_s.mhd')


def load_image_DIRLab_COPD_as_mhd(variation=1, folder='/home/veronika/Data/Reg/CRC/DIRLAB/COPD/copd'):

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
    voxel_sizes = [
        0,
        [0.625, 0.625, 2.5],
        [0.645, 0.645, 2.5],
        [0.652, 0.652, 2.5],
        [0.590, 0.590, 2.5],
        [0.647, 0.647, 2.5],
        [0.633, 0.633, 2.5],
        [0.625, 0.625, 2.5],
        [0.586, 0.586, 2.5],
        [0.664, 0.664, 2.5],
        [0.742, 0.742, 2.5],
    ]

    shape = image_sizes[variation]
    spacing = voxel_sizes[variation]

    # Images
    dtype = np.dtype(np.int16)

    # Inspiration
    with open(f'{folder}/copd{variation}/copd{variation}_iBHCT.img', "rb") as f:
        data = np.fromfile(f, dtype)
    image_insp = data.reshape(shape)

    img_insp = sitk.GetImageFromArray(image_insp)
    img_insp.SetSpacing(spacing)
    sitk.WriteImage(img_insp, f'{folder}/copd{variation}/copd{variation}_iBHCT.mhd')

    # Expiration
    with open(f'{folder}/copd{variation}/copd{variation}_eBHCT.img', "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

    img_exp = sitk.GetImageFromArray(image_exp)
    img_exp.SetSpacing(spacing)
    sitk.WriteImage(img_exp, f'{folder}/copd{variation}/copd{variation}_eBHCT.mhd')
