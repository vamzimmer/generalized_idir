import os
from typing import List
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchio as tio
import numpy as np
import nibabel as nib
import random
import json
from . import data_utils as dutils
from utils import general
from visualization import display_utils


class NLSTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", config_file: str = "./", batch_size=1, mode='image',
                 image_shape=[224,192,224], coordinate_batch_size=100, patch_size=[8,8,8],
                 int_transforms=None, augm_transforms=None, val_augm_transforms=None):
                #  fold: str = "./", img_size: tuple = (32, 32, 32), img_spacing: tuple = (1, 1, 1)):
                #  dataset='images', use_mask=True, coordinate_batch_size=10000):
        super().__init__()
        self.data_dir = data_dir
        self.config_file = config_file
        self.batch_size = batch_size
        self.int_transform = int_transforms
        self.val_augm_transforms = val_augm_transforms
        self.augm_transforms = augm_transforms
        self.mode = mode
        
        self.image_shape = image_shape
        self.coordinate_batch_size = coordinate_batch_size
        self.patch_size = patch_size

        self.config = {}
        if config_file is not None:
            with open(config_file, 'r') as f:
                self.config=json.load(f)
        else:
            self.config['training'] = np.arange(1,81,1)
            self.config['validation'] = np.arange(81,91,1)
            self.config['testing'] = np.arange(91,101,1)

        # automatic hyperparameter saving
        self.save_hyperparameters()

    def setup(self, stage: List[str] = None):
        # This method expects a stage argument. It is used to separate setup logic for
        # trainer.{fit,validate,test,predict}. If setup is called with stage=None,
        # we assume all stages have been set-up.

        # setup data transformations
        # self.setup_transform()

        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            # get samples for subsets (train|val|test|predict)
            # if self.mode == 'patch':
            #     self.nlst_train = NLSTDataset(f'{self.data_dir}/NLST', self.config['training'], transforms=self.train_int_transform, augm_transforms=self.augm_transforms,
            #                                        image_shape=self.image_shape, coordinate_batch_size=self.coordinate_batch_size, patch_size=self.patch_size)
            #     self.nlst_val = NLSTDataset(f'{self.data_dir}/NLST', self.config['validation'], transforms=self.val_int_transform,
            #                                      image_shape=self.image_shape, coordinate_batch_size=self.coordinate_batch_size, patch_size=self.patch_size)
            # else:
            self.nlst_train = NLSTDataset(f'{self.data_dir}/NLST', self.config['training'], transforms=self.int_transform, augm_transforms=self.augm_transforms)
            self.nlst_val = NLSTDataset(f'{self.data_dir}/NLST', self.config['validation'], transforms=None, augm_transforms=self.val_augm_transforms)
        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            # get samples for subsets (train|val|test|predict)
            # if self.mode == 'patch':
            #     self.nlst_test = NLSTDataset(f'{self.data_dir}/NLST', self.config['testing'], transforms=self.val_int_transform,
            #                                       image_shape=self.image_shape, coordinate_batch_size=self.coordinate_batch_size, patch_size=self.patch_size)
            # else:
            self.nlst_test = NLSTDataset(f'{self.data_dir}/NLST', self.config['testing'], transforms=None, augm_transforms=self.val_augm_transforms)

        if stage in (None, "predict"):
            self.nlst_predict = None
            print()

    def train_dataloader(self):
        return DataLoader(self.nlst_train, batch_size=self.batch_size, num_workers=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.nlst_val, batch_size=self.batch_size, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.nlst_test, batch_size=self.batch_size, num_workers=32)

    def predict_dataloader(self):
        return DataLoader(self.nlst_predict, batch_size=self.batch_size, num_workers=32)

    def setup_transform(self):
        transforms = [
            tio.Resample(self.img_spacing, image_interpolation='linear'),
            tio.CropOrPad(target_shape=self.img_size),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean, exclude=('FixedMask', 'MovingMask')), # whitening

        ]
        augm_transforms = [
            tio.RandomAffine(image_interpolation='linear'),

        ]
        self.transforms = tio.Compose(transforms)
        self.augm_transforms = tio.Compose(augm_transforms)

    # def __len__(self):
        # return len(self.filenames)


class NLSTDataset(Dataset):
    def __init__(self, data_dir= "./", samples=None, transforms=None, augm_transforms=None):
        self.data_dir = data_dir
        self.samples = samples
        self.transforms = transforms
        self.augm_transforms = augm_transforms

    def __getitem__(self, index):
        """
        Arguments
        ---------
        index : int
            index position to return the data
        Returns
        -------
        tuple: (fixed, moving) where label is the ground truth label image
        """
        direcs = ["imagesTr", "masksTr"]
        keysF = ["Fixed", "FixedMask"]
        keysM = ["Moving", "MovingMask"]

        

        if isinstance(self.samples[index], dict):
            case_id = self.samples[index]['fixed'].replace('./imagesTr/', '').replace('_0000.nii.gz', '')


            caseF = self.samples[index]['fixed'].replace('./', '')
            caseM = self.samples[index]['moving'].replace('./', '')
            caseFM = self.samples[index]['fixed'].replace('./images', 'masks')
            caseMM = self.samples[index]['moving'].replace('./images', 'masks')
            caseFL = self.samples[index]['fixed'].replace('./images', 'keypoints').replace('.nii.gz', '.csv')
            caseML = self.samples[index]['moving'].replace('./images', 'keypoints').replace('.nii.gz', '.csv')

        else:
            case_id = f'NLST_{str(self.samples[index]).zfill(4)}'

            caseF = f'imagesTr/NLST_{str(self.samples[index]).zfill(4)}_0000.nii.gz'
            caseM = f'imagesTr/NLST_{str(self.samples[index]).zfill(4)}_0001.nii.gz'
            caseFM = f'masksTr/NLST_{str(self.samples[index]).zfill(4)}_0000.nii.gz'
            caseMM = f'masksTr/NLST_{str(self.samples[index]).zfill(4)}_0001.nii.gz'
            caseFL = f'keypointsTr/NLST_{str(self.samples[index]).zfill(4)}_0000.csv'
            caseML = f'keypointsTr/NLST_{str(self.samples[index]).zfill(4)}_0001.csv'

        if os.path.exists(f'{self.data_dir}/{caseF}'):
            fixed_files = [f'{self.data_dir}/{caseF}', f'{self.data_dir}/{caseFM}']
            moving_files = [f'{self.data_dir}/{caseM}', f'{self.data_dir}/{caseMM}']
        elif os.path.exists(f'{self.data_dir.replace("/NLST","/NLST_testdata")}/{caseF}'):
            direc = self.data_dir.replace("/NLST","/NLST_testdata")
            fixed_files = [f'{direc}/{caseF}', f'{direc}/{caseFM}']
            moving_files = [f'{direc}/{caseM}', f'{direc}/{caseMM}']
        elif os.path.exists(f'{self.data_dir.replace("/NLST","/NLST_Validation")}/{caseF}'):
            direc = self.data_dir.replace("/NLST","/NLST_Validation")
            fixed_files = [f'{direc}/{caseF}', f'{direc}/{caseFM}']
            moving_files = [f'{direc}/{caseM}', f'{direc}/{caseMM}']

        fixed_LMs_file = f'{self.data_dir}/{caseFL}'
        moving_LMs_file = f'{self.data_dir}/{caseML}'

        # read images
        fixed = {
            'Fixed': tio.ScalarImage(fixed_files[0]),
            'FixedMask': tio.LabelMap(fixed_files[1]),
        }
        moving = {
            'Moving': tio.ScalarImage(moving_files[0]),
            'MovingMask': tio.LabelMap(moving_files[1]),
        }

        subject = tio.Subject(fixed)
        subjectM = tio.Subject(moving) 

        if self.transforms is not None:
            # print("Intensity data augmentation")
            subject = self.transforms(subject)
            subjectM = self.transforms(subjectM)

        subject.add_image(subjectM['Moving'], 'Moving')
        subject.add_image(subjectM['MovingMask'], 'MovingMask')

        if self.augm_transforms is not None:
            # print("Spatial data augmentation")
            subject = self.augm_transforms(subject)

        # Read landmarks
        if os.path.exists(fixed_LMs_file):
            fixed_LMs = np.genfromtxt(fixed_LMs_file, delimiter=',')
            moving_LMs = np.genfromtxt(moving_LMs_file, delimiter=',')
            landmarks = np.stack((fixed_LMs, moving_LMs), axis=0)
            landmarks = torch.from_numpy(landmarks)
        else:
            landmarks = []
        

        images, masks = dutils.create_tensor_from_torchio_subject(subject, image_names=[keysF[0], keysM[0]], mask_names=[keysF[1], keysM[1]])

        # return images, masks#, landmarks, case_id
        return images, masks, landmarks, case_id 

    def __len__(self):
        return len(self.samples)



# class NLSTPatchDataset(Dataset):
#     def __init__(self, data_dir= "./", samples=None, image_shape=[224,192,224],
#                  coordinate_batch_size=100, patch_size=[8,8,8],
#                  transforms=None, augm_transforms=None):
#         self.data_dir = data_dir
#         self.samples = samples
#         self.transforms = transforms
#         self.augm_transforms = augm_transforms

#         self.image_shape = image_shape
#         self.coordinate_batch_size = coordinate_batch_size
#         self.patch_size = patch_size

#         self.possible_coordinate_tensor_general = general.make_coordinate_tensor(
#                 self.image_shape, gpu=False
#         )
#         self.possible_index_tensor_general = general.make_index_tensor(
#                 self.image_shape, gpu=False
#         )

#     def __getitem__(self, index):
#         """
#         Arguments
#         ---------
#         index : int
#             index position to return the data
#         Returns
#         -------
#         tuple: (fixed, moving) where label is the ground truth label image
#         """
#         direcs = ["imagesTr", "masksTr"]
#         keysF = ["Fixed", "FixedMask"]
#         keysM = ["Moving", "MovingMask"]

#         if isinstance(self.samples[index], dict):
#             case_id = self.samples[index]['fixed'].replace('./imagesTr/', '').replace('_0000.nii.gz', '')


#             caseF = self.samples[index]['fixed'].replace('./', '')
#             caseM = self.samples[index]['moving'].replace('./', '')
#             caseFM = self.samples[index]['fixed'].replace('./images', 'masks')
#             caseMM = self.samples[index]['moving'].replace('./images', 'masks')
#             caseFL = self.samples[index]['fixed'].replace('./images', 'keypoints').replace('.nii.gz', '.csv')
#             caseML = self.samples[index]['moving'].replace('./images', 'keypoints').replace('.nii.gz', '.csv')

#         else:
#             case_id = f'NLST_{str(self.samples[index]).zfill(4)}'

#             caseF = f'imagesTr/NLST_{str(self.samples[index]).zfill(4)}_0000.nii.gz'
#             caseM = f'imagesTr/NLST_{str(self.samples[index]).zfill(4)}_0001.nii.gz'
#             caseFM = f'masksTr/NLST_{str(self.samples[index]).zfill(4)}_0000.nii.gz'
#             caseMM = f'masksTr/NLST_{str(self.samples[index]).zfill(4)}_0001.nii.gz'
#             caseFL = f'keypointsTr/NLST_{str(self.samples[index]).zfill(4)}_0000.csv'
#             caseML = f'keypointsTr/NLST_{str(self.samples[index]).zfill(4)}_0001.csv'

#         if os.path.exists(f'{self.data_dir}/{caseF}'):
#             fixed_files = [f'{self.data_dir}/{caseF}', f'{self.data_dir}/{caseFM}']
#             moving_files = [f'{self.data_dir}/{caseM}', f'{self.data_dir}/{caseMM}']
#         elif os.path.exists(f'{self.data_dir.replace("/NLST","/NLST_testdata")}/{caseF}'):
#             direc = self.data_dir.replace("/NLST","/NLST_testdata")
#             fixed_files = [f'{direc}/{caseF}', f'{direc}/{caseFM}']
#             moving_files = [f'{direc}/{caseM}', f'{direc}/{caseMM}']
#         elif os.path.exists(f'{self.data_dir.replace("/NLST","/NLST_Validation")}/{caseF}'):
#             direc = self.data_dir.replace("/NLST","/NLST_Validation")
#             fixed_files = [f'{direc}/{caseF}', f'{direc}/{caseFM}']
#             moving_files = [f'{direc}/{caseM}', f'{direc}/{caseMM}']

#         fixed_LMs_file = f'{self.data_dir}/{caseFL}'
#         moving_LMs_file = f'{self.data_dir}/{caseML}'

#         # read images
#         fixed = {
#             'Fixed': tio.ScalarImage(fixed_files[0]),
#             'FixedMask': tio.LabelMap(fixed_files[1]),
#         }
#         moving = {
#             'Moving': tio.ScalarImage(moving_files[0]),
#             'MovingMask': tio.LabelMap(moving_files[1]),
#         }

#         subject = tio.Subject(fixed)
#         subjectM = tio.Subject(moving) 

#         if self.transforms is not None:
#             # print("Intensity data augmentation")
#             subject = self.transforms(subject)
#             subjectM = self.transforms(subjectM)

#         subject.add_image(subjectM['Moving'], 'Moving')
#         subject.add_image(subjectM['MovingMask'], 'MovingMask')

#         # Read landmarks
#         if os.path.exists(fixed_LMs_file):
#             fixed_LMs = np.genfromtxt(fixed_LMs_file, delimiter=',')
#             moving_LMs = np.genfromtxt(moving_LMs_file, delimiter=',')
#             landmarks = np.stack((fixed_LMs, moving_LMs), axis=0)
#         else:
#             landmarks = []
#         landmarks = torch.from_numpy(landmarks)

#         images, masks = dutils.create_tensor_from_torchio_subject(subject, image_names=[keysF[0], keysM[0]], mask_names=[keysF[1], keysM[1]])

#         possible_coordinate_tensor = self.possible_coordinate_tensor_general[masks[0,...].flatten() > 0, :]
#         possible_index_tensor = self.possible_index_tensor_general[masks[0,...].flatten() > 0, :]

#         # random subset of coordinate tensors, number determined by self.batch_size
#         indices = torch.randperm(
#             possible_coordinate_tensor.shape[0], #device="cuda"
#         )[: self.coordinate_batch_size]

#         # get the target (the intensities of the fixed image at the coodinates)
#         fixed_intensities = images[0,...].flatten()[masks[0,...].flatten() > 0][indices]

#         coordinate_tensor = possible_coordinate_tensor[indices, :]
#         index_tensor = possible_index_tensor[indices, :]

#         #
#         #   Extract patches with coordinates as patch centers
#         #
#         patches = general.sample_all_patch_pairs_3d(torch.unsqueeze(images[:1,:],0), torch.unsqueeze(images[1:,:],0), index_tensor, self.patch_size)
        
#         # iid = 0
#         # display_utils.display_3d([patches[0,iid,:].view(self.patch_size), patches[1,iid,:].view(self.patch_size)], pfile='/home/veronika/Code/IDIR/patches.png')

#         return coordinate_tensor, patches, fixed_intensities, images[1,...]

#     def __len__(self):
#         return len(self.samples)

