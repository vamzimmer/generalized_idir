# Add noise
# np_noisy = np_target + sigma * np.random.randn(*np_target.shape).astype(np_target.dtype)
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


class NLSTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", config_file: str = "./", 
                 int_transforms=None, augm_transforms=None, val_augm_transforms=None):
                #  fold: str = "./", img_size: tuple = (32, 32, 32), img_spacing: tuple = (1, 1, 1)):
                #  dataset='images', use_mask=True, coordinate_batch_size=10000):
        super().__init__()
        self.data_dir = data_dir
        self.config_file = config_file
        self.int_transform = int_transforms
        self.val_augm_transforms = val_augm_transforms
        self.augm_transforms = augm_transforms

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        config = {}
        if config_file is not None:
            with open(config_file, 'r') as f:
                config=json.load(f)

        # Parse important parameters
        self.batch_size = config['batch_size'] if "batch_size" in config else self.args['batch_size']
        self.num_workers = config['num_workers'] if "num_workers" in config else self.args['num_workers']

        self.training_samples = config['training'] if "training" in config else self.args['training']
        self.validation_samples = config['validation'] if "validation" in config else self.args['validation']
        self.testing_samples = config['testing'] if "testing" in config else self.args['testing']

        self.patch_number = config['patch_number'] if "patch_number" in config else self.args['patch_number']
        self.patch_size = config['patch_size'] if "patch_size" in config else self.args['patch_size']
        if self.patch_size=='none':
            self.patch_size = None
        self.patch_sigma = config['patch_sigma'] if 'patch_sigma' in config else self.args['patch_sigma']

        self.image_shape = config['image_shape'] if 'image_shape' in config else self.args['image_shape']
        self.use_mask = config['use_mask'] if 'use_mask' in config else self.args['use_mask']
        

        # print(self.patch_size)
        # print(self.patch_number)
        # print(self.patch_sigma)
        
        # else:
        #     self.config['training'] = np.arange(1,81,1)
        #     self.config['validation'] = np.arange(81,91,1)
        #     self.config['testing'] = np.arange(91,101,1)

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
            self.nlst_train = NLSTDataset(f'{self.data_dir}/NLST', self.training_samples, transforms=self.int_transform, augm_transforms=self.augm_transforms,
                                          batch_size=self.patch_number, patch_size=self.patch_size, use_mask=self.use_mask,
                                          image_shape=self.image_shape, sigma=self.patch_sigma)
            self.nlst_val = NLSTDataset(f'{self.data_dir}/NLST', self.validation_samples, transforms=None, augm_transforms=self.val_augm_transforms,
                                        batch_size=self.patch_number, patch_size=self.patch_size, use_mask=self.use_mask,
                                        image_shape=self.image_shape, sigma=self.patch_sigma)

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            # get samples for subsets (train|val|test|predict)
            self.nlst_test = NLSTDataset(f'{self.data_dir}/NLST', self.testing_samples, transforms=None, augm_transforms=self.val_augm_transforms,
                                         batch_size=self.patch_number, patch_size=self.patch_size, use_mask=self.use_mask,
                                         image_shape=self.image_shape, sigma=self.patch_sigma)

        if stage in (None, "predict"):
            self.nlst_predict = None
            print()

    def set_default_arguments(self):
        """Set default arguments."""

        self.args = {}

        self.args['num_workers'] = 48
        self.args['batch_size'] = 1

        self.args['image_shape'] = (224, 192, 224)
        self.args['use_mask'] = True


        self.args['training'] = np.arange(1,81,1)
        self.args['validation'] = np.arange(81,91,1)
        self.args['testing'] = np.arange(91,101,1)

        self.args['patch_number'] = 10
        self.args['patch_size'] = (32, 32, 32)
        self.args['patch_sigma'] = 0.01


    def train_dataloader(self):
        return DataLoader(self.nlst_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.nlst_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.nlst_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.nlst_predict, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)

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
    def __init__(self, data_dir= "./", samples=None, transforms=None, augm_transforms=None,
                 batch_size=10, patch_size=(8,8,8), use_mask=True, image_shape=(224, 192, 224),
                 sigma=0.1):
        self.data_dir = data_dir
        self.samples = samples
        self.transforms = transforms
        self.augm_transforms = augm_transforms

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.use_mask = use_mask
        self.image_shape = image_shape
        self.sigma = sigma

        self.possible_index_tensor_general = general.make_index_tensor(dims=self.image_shape, gpu=False)


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
        keysF = ["Fixed", "FixedMask"]
        keysM = ["Moving", "MovingMask"]

        if isinstance(self.samples[index], dict):
            case_id = self.samples[index]['fixed'].replace('./imagesTr/', '').replace('_0000.nii.gz', '')


            caseF = self.samples[index]['fixed'].replace('./', '')
            caseM = self.samples[index]['moving'].replace('./', '')
            caseFM = self.samples[index]['fixed'].replace('./imagesTr', 'masksTr')
            caseMM = self.samples[index]['moving'].replace('./imagesTr', 'masksTr')
        else:
            case_id = f'NLST_{str(self.samples[index]).zfill(4)}'

            caseF = f'imagesTr/NLST_{str(self.samples[index]).zfill(4)}_0000.nii.gz'
            caseM = f'imagesTr/NLST_{str(self.samples[index]).zfill(4)}_0001.nii.gz'
            caseFM = f'masksTr/NLST_{str(self.samples[index]).zfill(4)}_0000.nii.gz'
            caseMM = f'masksTr/NLST_{str(self.samples[index]).zfill(4)}_0001.nii.gz'

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

        images, masks = dutils.create_tensor_from_torchio_subject(subject, image_names=[keysF[0], keysM[0]], mask_names=[keysF[1], keysM[1]])

        # print(self.patch_size)
        if self.patch_size is not None:
            # extract patches
            if self.use_mask:
                possible_index_tensor = self.possible_index_tensor_general[masks[0,...].flatten() > 0, :]
            else:
                possible_index_tensor = self.possible_index_tensor_general

            indices = torch.randperm(possible_index_tensor.shape[0])[: self.batch_size]
            fixed_centres = possible_index_tensor[indices, :]

            patches = general.sample_all_patch_pairs_3d(torch.unsqueeze(images[:1,...],0), torch.unsqueeze(images[1:,...],0), fixed_centres, patch_size=self.patch_size)

            # add noise
            # print(self.sigma)
            # patches_noisy = patches + (self.sigma**0.5) * torch.randn(patches.size())
            patches_noisy = patches + (self.sigma) * torch.randn(patches.size())

            return patches_noisy, patches, case_id
            # return patches, case_id

        else:
            # add noise
            # print(self.sigma)
            # patches_noisy = patches + (self.sigma**0.5) * torch.randn(patches.size())
            images_noisy = images + (self.sigma) * torch.randn(images.size())

            return images_noisy, images, case_id

    def __len__(self):
        return len(self.samples)


