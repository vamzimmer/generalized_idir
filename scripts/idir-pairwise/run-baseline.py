import os
import sys
import torch
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import csv

sys.path.insert(0, os.path.abspath('../../../IDIR'))
from utils import data
from utils import general
from utils import utils_itk
from utils import evaluation

from models import models
from visualization import display_utils, plot_utils
from datasets import NLSTL2RDataset, data_utils

mode = 'train' 
# mode = 'test'
mode = 'eval'
do_copy_files = False
# do_copy_files = True

#
#   Set the correct directories
#
config_dir = '.../generalized_idir/scripts/idir-pairwise'
data_dir = '../'
out_dir = '../'

# Dataset 1: L2R 2022 Task 1: NLST lung CT-CT intra-patient
config_file = f'{config_dir}/L2R2022-T1-siren.json' # SIREN, epochs 1000
config_file = f'{config_dir}/L2R2022-T1-mlp.json' # MLP. epochs 1000 

# regularization = ['bending', 'jacobian', 'hyper']
regularization = ['bending']

with open(config_file, 'r') as f:
    config=json.load(f)

print()
print(f'Experiment: {config["exp_name"]}')
print(f'Network Type: {config["network_type"]}')
# print(f'Regularization: {config["regularization"]}')
print()

# dataset = NLSTL2RDataset.NLSTDataModule(data_dir=data_dir, config_file=config_file)

data_augmentation = config["data_augmentation"] if "data_augmentation" in config else 0

if data_augmentation:
        optn = {}
        optn["whitening"] = config["whitening"]
        optn["rescaling"] = config["rescaling"]
        optn["resample"] = config["resample"] if "resample" in config else 0
        optn["resample_size"] = config["resample_size"] if "resample_size" in config else [128, 128, 128]
        optn["blur"] = config["blur"][0]
        optn["blur_prob"] = config["blur"][1]
        optn["noise"] = config["noise"][0]
        optn["noise_prob"] = config["noise"][1]
        optn["spatial_prob"] = config["spatial"][0] if "spatial" in config else 0
        optn["affine_prob"] = config["spatial"][1] if "spatial" in config else 0.5
        optn["elastic_prob"] = config["spatial"][2] if "spatial" in config else 0.5
        optn["flip_prob"] = config["flip"][0] if "flip" in config else 0
        optn["flip_axes"] = config["flip"][1] if "flip" in config else ("LR","AP", "IS")
        # train_tfm, val_tfm = data_utils.intensity_transforms(**optn)
        # train_tfm, val_tfm = data_utils.data_transforms(**optn)
        int_tfm = data_utils.intensity_transforms(**optn)
        train_tfm, val_tfm = data_utils.data_transforms(**optn)

        dataset = NLSTL2RDataset.NLSTDataModule(data_dir=data_dir, config_file=config_file,
                                                int_transforms=int_tfm, augm_transforms=train_tfm,
                                                val_augm_transforms=val_tfm)

else:

    dataset = NLSTL2RDataset.NLSTDataModule(data_dir=data_dir, config_file=config_file)


# dataset.setup('fit')
dataset.setup('test')

testloader = dataset.test_dataloader()
print('Train loader size = {}'.format(len(testloader)))
images, masks, landmarks, ids = dataset.nlst_test[0]

voxel_size = config["voxel_size"]["fixed"]
print(voxel_size)

for regularizer in regularization:

    if mode=='train' or mode=='test':

        for i, (images, masks, keypoints, idx) in enumerate(testloader):

            # if i>2:
                # continue


            case_dir = f'{out_dir}/{regularizer}/{idx[0]}'
            if not os.path.exists(case_dir):
                os.makedirs(case_dir)

            print(f'Regularizer: {regularizer}')
            print(f'Case: {idx[0]}')

            fixed = images[0,0,:,:,:]
            moving = images[0,1,:,:,:]
            fixed_mask = masks[0,0,:,:,:]
            moving_mask = masks[0,1,:,:,:]
            fixed_landmarks = keypoints[0,0,:,:].numpy()
            moving_landmarks = keypoints[0,1,:,:].numpy()
            # fixed_landmarks = landmarks[0,:,:]
            # moving_landmarks = landmarks[1,:,:]

            # print(images.size())
            # print(masks.size())
            # print(keypoints.size())

            print(torch.min(fixed), torch.max(fixed))
            print(torch.min(moving), torch.max(moving))

            # Display images
            # pfile = f'{config_dir}/{idx[0]}-input.png'
            pfile = f'{config_dir}/Input-{config["exp_name"]}-{regularizer}-{idx[0]}.png'
            display_utils.display_3d([fixed.numpy(), moving.numpy()], [fixed_mask.numpy(), moving_mask.numpy()], 
                                    show=False, pfile=pfile)

            #
            #   Model
            #
            kwargs = {}
            kwargs["epochs"] = config['epochs'] if 'epochs' in config else 2500
            kwargs["verbose"] = config["verbose"]
            if config["verbose"]:
                kwargs["progress_file"] = f'{config_dir}/Training-{config["exp_name"]}-{regularizer}-{idx[0]}.png'
                # kwargs["progress_file"] = f'{config_dir}/{idx[0]}-training-{config["exp_name"]}.png'
            kwargs["hyper_regularization"] = False
            kwargs["jacobian_regularization"] = False
            kwargs["bending_regularization"] = False
            kwargs["network_type"] = config["network_type"]
            kwargs["omega"] = config["omega"] if "omega" in config else 32
        
            kwargs["mask"] = fixed_mask.numpy()

            if regularizer=='hyper':
                kwargs["hyper_regularization"] = True
            elif regularizer=='jacobian':
                kwargs["jacobian_regularization"] = True
            elif regularizer=='bending':
                kwargs["bending_regularization"] = True

            kwargs["save_folder"] = case_dir

            # ImpReg = models.ImplicitRegistrator(moving, fixed, **kwargs)
            ImpReg = models.ImplicitRegistrator(moving, fixed, moving_landmarks=moving_landmarks, fixed_landmarks=fixed_landmarks, **kwargs)

            if mode == 'train':
                model_file = f'{case_dir}/model-inr-epochs{config["epochs"]}.pt'
                if not os.path.exists(model_file):
                    ImpReg.fit()  # training of the implicit neural representation for one image pair
                else:
                    print("Training done.")
            elif mode == 'test':
                ImpReg.infer()  # apply the implicit neural representation


                #
                #   Evaluation
                #

                # Compute registered landmarks
                registered_landmarks, _ = general.compute_landmarks(ImpReg.network, fixed_landmarks, image_size=fixed.shape)

                pfile = f'{config_dir}/LMs-{config["exp_name"]}-{regularizer}-{idx[0]}.png'
                plot_utils.display_imagepairs_landmarks(fixed,
                                                fixed_landmarks[:,::-1], moving_landmarks[:,::-1],
                                                landmarks3=registered_landmarks[:,::-1],
                                                slices=None, project=False, fig_size=[1000, 800],
                                                show=False, pfile=pfile)

                f = open(f'{case_dir}/fixed_landmarks_warped.csv', 'w')
                writer = csv.writer(f)
                writer.writerows(registered_landmarks)

                #Transform image
                image_size = fixed.shape

                print(image_size)
                # image_size = [224, 150, 224]

                # moving grid coordinates
                coords = general.make_coordinate_tensor(dims=image_size)
                # get the transformed moving grid coordinates and resample the moving image
                moving_transformed = ImpReg(coordinate_tensor=coords, output_shape=image_size)
                # nib.save(nib.Nifti1Image(moving_transformed, np.eye(4)), f'{case_dir}/moving_image_warped.nii.gz')

                # Create resampled difference image
                fixed_itk = sitk.GetImageFromArray(fixed)
                fixed_itk.SetSpacing(voxel_size)
                moving_itk = sitk.GetImageFromArray(moving)
                moving_itk.SetSpacing(voxel_size)

                fixed_itk = utils_itk.image_resample(fixed_itk, out_size=image_size[::-1])
                moving_itk = utils_itk.image_resample(moving_itk, out_size=image_size[::-1])

                difference_before = sitk.GetArrayFromImage(fixed_itk) - sitk.GetArrayFromImage(moving_itk)
                difference_after = sitk.GetArrayFromImage(fixed_itk) - moving_transformed

                # nib.save(nib.Nifti1Image(sitk.GetArrayFromImage(fixed_itk), np.eye(4)), f'{case_dir}/fixed_image.nii.gz')
                # nib.save(nib.Nifti1Image(sitk.GetArrayFromImage(moving_itk), np.eye(4)), f'{case_dir}/moving_image.nii.gz')

                pfile = f'{config_dir}/Diff-{config["exp_name"]}-{regularizer}-{idx[0]}.png'
                display_utils.display_3d([difference_before, difference_after], title=['Difference berfore', 'Difference after'],
                                        show=False, pfile=pfile)

                # save deformation and displacement
                displ_field = ImpReg.displacement.cpu().detach().numpy().reshape(image_size[0], image_size[1], image_size[2], 3)
                def_field = ImpReg.transformation.cpu().detach().numpy().reshape(image_size[0], image_size[1], image_size[2], 3)
                # nib.save(nib.Nifti1Image(displ_field, np.eye(4)), f'{case_dir}/displ_field.nii.gz')
                # nib.save(nib.Nifti1Image(def_field, np.eye(4)), f'{case_dir}/def.nii.gz')

                # visualize displacement field
                pfile = f'{config_dir}/Field-{config["exp_name"]}-{regularizer}-{idx[0]}.png'
                plot_utils.display_field_3d(displ_field, factor=200,pfile=pfile)


                # Transform masks
                mask_transformed = ImpReg.transform_no_add(ImpReg.transformation.cpu(), moving_image=moving_mask)
                mask_transformed = mask_transformed.cpu().detach().numpy().reshape(image_size[0], image_size[1], image_size[2])
                nib.save(nib.Nifti1Image(mask_transformed, np.eye(4)), f'{case_dir}/moving_mask_warped.nii.gz')

                pfile = f'{config_dir}/Output-{config["exp_name"]}-{regularizer}-{idx[0]}.png'
                display_utils.display_3d([fixed_itk, moving_itk, moving_transformed], 
                                        title=['Fixed image', 'Moving image', 'Transformed moving image'],
                                        show=False, pfile=pfile)

                # Create resampled difference image
                fixed_mask_itk = sitk.GetImageFromArray(fixed_mask)
                fixed_mask_itk.SetSpacing(voxel_size)
                moving_mask_itk = sitk.GetImageFromArray(moving_mask)
                moving_mask_itk.SetSpacing(voxel_size)

                fixed_mask_itk = utils_itk.image_resample(fixed_mask_itk, out_size=image_size[::-1])
                moving_mask_itk = utils_itk.image_resample(moving_mask_itk, out_size=image_size[::-1])

                nib.save(nib.Nifti1Image(sitk.GetArrayFromImage(fixed_mask_itk), np.eye(4)), f'{case_dir}/fixed_mask.nii.gz')
                nib.save(nib.Nifti1Image(sitk.GetArrayFromImage(moving_mask_itk), np.eye(4)), f'{case_dir}/moving_mask.nii.gz')

                pfile = f'{config_dir}/OutputMasks-{config["exp_name"]}-{regularizer}-{idx[0]}.png'
                display_utils.display_3d([fixed_itk, fixed_itk], [[fixed_mask_itk, moving_mask_itk],[fixed_mask_itk, mask_transformed]],
                                        title=['Fixed & moving mask', 'Fixed and registered'],
                                        show=False, pfile=pfile)
                
            print()


    elif mode=='eval':

        #
        #   Evaluation of performance
        #

        evaluation.evaluation(f'{data_dir}/NLST', f'{out_dir}/{regularizer}', config_file, subset='testing')

    if do_copy_files:
        # copy files
        os.system(f'cp {config_file} {config_file.replace(config_dir, out_dir)}')
        if not os.path.exists(f'{out_dir}/{regularizer}/pngs'):
            os.makedirs(f'{out_dir}/{regularizer}/pngs')
        os.system(f'mv {config_dir}/Training-{config["exp_name"]}-{regularizer}-* {out_dir}/{regularizer}/pngs')
        os.system(f'mv {config_dir}/Input-{config["exp_name"]}-{regularizer}-* {out_dir}/{regularizer}/pngs')
        os.system(f'mv {config_dir}/LMs-{config["exp_name"]}-{regularizer}-* {out_dir}/{regularizer}/pngs')
        os.system(f'mv {config_dir}/Diff-{config["exp_name"]}-{regularizer}-* {out_dir}/{regularizer}/pngs')
        os.system(f'mv {config_dir}/Field-{config["exp_name"]}-{regularizer}-* {out_dir}/{regularizer}/pngs')
        os.system(f'mv {config_dir}/Output-{config["exp_name"]}-{regularizer}-* {out_dir}/{regularizer}/pngs')
        os.system(f'mv {config_dir}/OutputMasks-{config["exp_name"]}-{regularizer}-* {out_dir}/{regularizer}/pngs')

