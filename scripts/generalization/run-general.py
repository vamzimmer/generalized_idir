import os
import sys

# import socket
# if socket.gethostname() in ['chameleon', 'prometheus']:
#     import matplotlib
#     matplotlib.use('TkAgg')
#     print('Use Matplotlib TkAgg')

import torch
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import csv

sys.path.insert(0, os.path.abspath('../../../IDIR'))
sys.path.insert(0, os.path.abspath('../IDIR'))
from utils import data
from utils import general
from utils import utils_itk
from utils import evaluation

from models import modelsGeneral as models
# from models import modelsNaive_check as models
from visualization import display_utils, plot_utils
from datasets import NLSTL2RDataset, data_utils

do_copy_files = False
# do_copy_files = True


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, required=True)
parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "eval"])
parser.add_argument("--only-tre", action="store_true")
args = parser.parse_args()

#
#   Set the correct directories
#
out_root = '../'

repo_dir = '../generalized_idir'
config_dir = f'{repo_dir}/scrips/generalization/configs/dgm4miccai23'
data_dir = '../'
enc_root = f'{repo_dir}/model_zoo'

info_file = f'{data_dir}/NLST/NLST_dataset.json'

experiments = [args.exp]
mode = args.mode

for exp in experiments:

    config_file = f'{config_dir}/L2R2022-T1-{exp}.json' 

    with open(config_file, 'r') as f:
        config=json.load(f)

    print()
    print(f'Experiment: {config["exp_name"]}')
    print(f'Network Type: {config["network_type"]}')
    print(f'Regularization: {config["regularization"]}')
    print(f'FM mapping type: {config["FM_mapping_type"]}')
    print(f'FM mapping size: {config["FM_mapping_size"]}')
    print(f'FM sigma: {config["FM_sigma"]}')
    print(f'Encoder config: {config["encoder"]}')
    print()

    out_dir = f'{out_root}/{config["exp_name"]}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if config["data_augmentation"]:
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
                                            # dataset='images', use_mask=True, coordinate_batch_size=10000)
                                                # img_size=(128,128,128), img_spacing=(2,2,2), fold='FOLD-1'
    dataset.setup('fit')
    dataset.setup('test')

    trainloader = dataset.train_dataloader()
    print('Train loader size = {}'.format(len(trainloader)))

    # images, masks, landmarks, ids = dataset.nlst_train[0]

    valloader = dataset.val_dataloader()
    print('Val loader size = {}'.format(len(valloader)))

    testloader = dataset.test_dataloader()
    print('Test loader size = {}'.format(len(testloader)))
    # images, masks, landmarks, ids = dataset.nlst_test[0]
    # print(ids)

    # if 'exp2' in exp or 'exp3' in exp:
    #     valloader = None

    voxel_size = config["voxel_size"]["fixed"]

    epochs = 2500
    if "epochs" in config:
        epochs = config["epochs"]

    # valoader = trainset.val_dataloader()
    # print('Validation loader size = {}'.format(len(valoader)))
    # images, masks, landmarks = trainset.nlst_val[0]

    if mode=='train' or mode=='test':
        
        #
        #   Model
        #
        kwargs = {}
        kwargs["epochs"] = epochs
        kwargs["batch_size"] = config["batch_size"]
        kwargs["verbose"] = config["verbose"]
        if config["verbose"]:
            kwargs["progress_file"] = f'{config_dir}/Training-{config["exp_name"]}.png'
        kwargs["hyper_regularization"] = False
        kwargs["jacobian_regularization"] = False
        kwargs["bending_regularization"] = False
        if config["regularization"]=='hyper':
            kwargs["hyper_regularization"] = True
        if config["regularization"]=='jacobian':
            kwargs["jacobian_regularization"] = True
        if config["regularization"]=='bending':
            kwargs["bending_regularization"] = True
        if "alpha_bending" in config:
            kwargs["alpha_bending"] = config["alpha_bending"]
        if "alpha_hyper" in config:
            kwargs["alpha_hyper"] = config["alpha_hyper"]
        if "alpha_jacobian" in config:
            kwargs["alpha_jacobian"] = config["alpha_jacobian"]
        # kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
        kwargs["network_type"] = config["network_type"]
        kwargs["layers"] = config["layers"]
        kwargs["lr"] = config["lr"]
        kwargs["patch_size"] = config["patch_size"] if not config["patch_size"] == "none" else None
        kwargs["FM_mapping_type"] = config["FM_mapping_type"]
        kwargs["FM_mapping_size"] = config["FM_mapping_size"]
        kwargs["FM_sigma"] = config["FM_sigma"]
        if "omega" in config:
            kwargs["omega"] = config["omega"]
        kwargs["encoder_type"] = config["encoder_type"]
        kwargs["modulation_type"] = config["modulation_type"]
        if "modulation_activation_type" in config:
            kwargs["modulation_activation_type"] = config["modulation_activation_type"]
        kwargs["encoder_config"] = config["encoder"]
        if 'checkpoint' in kwargs["encoder_config"] and os.path.exists(f'{repo_dir}/model-zoo/{kwargs["encoder_config"]["checkpoint"]}'):
        #    kwargs["encoder_config"]['checkpoint'] = f'{enc_root}/{kwargs["encoder_config"]["checkpoint"]}'
            kwargs["encoder_config"]['checkpoint'] = f'{repo_dir}/model-zoo/{kwargs["encoder_config"]["checkpoint"]}'
        kwargs["save_folder"] = out_dir

        # kwargs["mask"] = masks[0,:].numpy()
        kwargs["use_mask"] = True

        kwargs["image_shape"] = config["image_size"]
        kwargs["voxel_size"] = config["voxel_size"]["moving"]

        ImpReg = models.ImplicitRegistrator(trainDataloader=trainloader, validateDataloader=valloader, inferDataloader=testloader, 
                                            **kwargs)
        # ImpReg = models.ImplicitRegistrator(trainDataloader=trainloader, validateDataloader=None, inferDataloader=testloader, 
                                            # **kwargs)

        if mode == 'train':
            # if "checkpoint" in config:
                # print('Load checkpoint')
                # ImpReg.load_checkpoint(f'{out_dir}/{config["checkpoint"]}')
            ImpReg.fit()  # training of the implicit neural representation for one image pair
        
        if mode == 'test':
        # elif mode == 'test':
            model_file = f'{out_dir}/model-best.pt'
            if not os.path.exists(model_file):
                model_file = f'{out_dir}/checkpoint-epochs{config["epochs"]}.pt'
            print(model_file)
            ImpReg.infer(model_file)  # apply the implicit neural representation

            for i, (images, masks, keypoints, idx) in enumerate(testloader):

                # if i<9:
                    # continue
                # if not i==9:
                    # continue

                print(f'Test case {idx[0]}')
                print()

                fixed = images[0,0,:]
                moving = images[0,1,:]
                fixed_mask = masks[0,0,:]
                moving_mask = masks[0,1,:]
                fixed_landmarks = keypoints[0,0,:].numpy()
                moving_landmarks = keypoints[0,1,:].numpy()
                case_dir = f'{out_dir}/{idx[0]}'
                if not os.path.exists(case_dir):
                    os.makedirs(case_dir)
                case_id = idx[0]

                #
                #   Evaluation
                #
                # Landmarks as image indices, not as continuous world coordinates!
                with torch.no_grad():
                    registered_landmarks, _ = ImpReg.compute_landmarks(fixed_landmarks, fixed.shape, 
                                            moving_image=moving, fixed_image=fixed)
                
                plot_utils.display_imagepairs_landmarks(fixed,
                                                fixed_landmarks[:,::-1], moving_landmarks[:,::-1],
                                                landmarks3=registered_landmarks[:,::-1],
                                                slices=None, project=False, fig_size=[1000, 800],
                                                show=False, pfile=f'{config_dir}/{config["exp_name"]}-LMs-{case_id}.png')

                f = open(f'{case_dir}/fixed_landmarks_warped.csv', 'w')
                writer = csv.writer(f)
                writer.writerows(registered_landmarks)

                if not args.only_tre:

                    #Transform image
                    image_size = fixed.shape
                    # image_size = [224, 150, 224]
                    # image_size = [128, 128, 128]

                    # moving grid coordinates
                    coords = general.make_coordinate_tensor(dims=image_size)
                    indices = general.make_index_tensor(dims=image_size)
                    # get the transformed moving grid coordinates and resample the moving image
                    moving_transformed = ImpReg(moving_image=moving,coordinate_tensor=coords, index_tensor=indices, fixed_image=fixed, 
                                                output_shape=image_size, batch_size=32768)
                    nib.save(nib.Nifti1Image(moving_transformed, np.eye(4)), f'{case_dir}/moving_image_warped.nii.gz')

                    # Create resampled difference image
                    fixed_itk = sitk.GetImageFromArray(fixed)
                    fixed_itk.SetSpacing(voxel_size)
                    moving_itk = sitk.GetImageFromArray(moving)
                    moving_itk.SetSpacing(voxel_size)

                    fixed_itk = utils_itk.image_resample(fixed_itk, out_size=image_size[::-1])
                    moving_itk = utils_itk.image_resample(moving_itk, out_size=image_size[::-1])

                    difference_before = sitk.GetArrayFromImage(fixed_itk) - sitk.GetArrayFromImage(moving_itk)
                    difference_after = sitk.GetArrayFromImage(fixed_itk) - moving_transformed

                    nib.save(nib.Nifti1Image(sitk.GetArrayFromImage(fixed_itk), np.eye(4)), f'{case_dir}/fixed_image.nii.gz')
                    nib.save(nib.Nifti1Image(sitk.GetArrayFromImage(moving_itk), np.eye(4)), f'{case_dir}/moving_image.nii.gz')

                    display_utils.display_3d([difference_before, difference_after], title=['Difference before', 'Difference after'],
                                            show=False, pfile=f'{config_dir}/{config["exp_name"]}-diff-{case_id}.png')

                    # save deformation and displacement
                    displ_field = ImpReg.displacement.cpu().detach().numpy().reshape(image_size[0], image_size[1], image_size[2], 3)
                    def_field = ImpReg.transformation.cpu().detach().numpy().reshape(image_size[0], image_size[1], image_size[2], 3)
                    nib.save(nib.Nifti1Image(displ_field, np.eye(4)), f'{case_dir}/displ_field.nii.gz')
                    nib.save(nib.Nifti1Image(def_field, np.eye(4)), f'{case_dir}/def.nii.gz')

                    # visualize displacement field

                    plot_utils.display_field_3d(displ_field, pfile=f'{config_dir}/{config["exp_name"]}-field-{case_id}.png')

                    # Transform masks
                    mask_transformed = ImpReg.transform_no_add(ImpReg.transformation.cpu(), moving_image=moving_mask)
                    mask_transformed = mask_transformed.cpu().detach().numpy().reshape(image_size[0], image_size[1], image_size[2])
                    nib.save(nib.Nifti1Image(mask_transformed, np.eye(4)), f'{case_dir}/moving_mask_warped.nii.gz')

                    display_utils.display_3d([fixed_itk, moving_itk, moving_transformed], 
                                            title=['Fixed image', 'Moving image', 'Transformed moving image'],
                                            show=False, pfile=f'{config_dir}/{config["exp_name"]}-output-{case_id}.png')

                    # Create resampled difference image
                    fixed_mask_itk = sitk.GetImageFromArray(fixed_mask)
                    fixed_mask_itk.SetSpacing(voxel_size)
                    moving_mask_itk = sitk.GetImageFromArray(moving_mask)
                    moving_mask_itk.SetSpacing(voxel_size)

                    fixed_mask_itk = utils_itk.image_resample(fixed_mask_itk, out_size=image_size[::-1])
                    moving_mask_itk = utils_itk.image_resample(moving_mask_itk, out_size=image_size[::-1])

                    nib.save(nib.Nifti1Image(sitk.GetArrayFromImage(fixed_mask_itk), np.eye(4)), f'{case_dir}/fixed_mask.nii.gz')
                    nib.save(nib.Nifti1Image(sitk.GetArrayFromImage(moving_mask_itk), np.eye(4)), f'{case_dir}/moving_mask.nii.gz')

                    display_utils.display_3d([fixed_itk, fixed_itk], [[fixed_mask_itk, moving_mask_itk],[fixed_mask_itk, mask_transformed]],
                                            title=['Fixed & moving mask', 'Fixed and registered'],
                                            show=False, pfile=f'{config_dir}/{config["exp_name"]}-output-masks-{case_id}.png')

        
    elif mode=='eval':

        #
        #   Evaluation of performance
        #

        if args.only_tre:
            config['evaluation_methods'] = [
                {"name": "TRE_kp", "metric": "tre", "dest": "keypoints"
            }]

        print('evaluation')
        print(out_dir)
        evaluation.evaluation(f'{data_dir}/NLST', out_dir, config, subset='testing')

    if do_copy_files:
        # copy files
        os.system(f'cp {config_file} {config_file.replace(config_dir, out_dir)}')
        os.system(f'mkdir {out_dir}/pngs')
        os.system(f'mv {config_dir}/{config["exp_name"]}-* {out_dir}/pngs')
        os.system(f'cp {config_dir}/Training-{config["exp_name"]}.png {out_dir}')
        # case_dir = f'{out_dir}'
        # os.system(f'cp {config_file} {config_file.replace(config_dir, case_dir)}')
        # os.system(f'mkdir {case_dir}/pngs')
        # os.system(f'cp {config_dir}/{config["exp_name"]}-* {case_dir}/pngs')
