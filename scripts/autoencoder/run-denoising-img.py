import os
import sys
import matplotlib
matplotlib.use('TkAgg')
import torch
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import csv
from tqdm import tqdm
from random import randint

sys.path.insert(0, os.path.abspath('../../../IDIR'))
from utils import data, general, utils_itk, evaluation, eval_utils

# from models import modelsPatch as models
# from models import modelsNaive_check as models
from visualization import display_utils, plot_utils
from datasets import NLSTDenoisingDataset, NLSTL2RDataset, data_utils
from networks import autoencoder
from models import utils

from denoising_utils import psnr_criterion
import time
from tqdm import tqdm

mode = 'train' 
# mode = 'test'
# mode = 'eval'
user = 'KH'
# user = 'VZ'

if user == 'VZ':
    config_dir = '/home/veronika/Code/IDIR/scripts/autoencoder'
    data_dir = '/home/veronika/Data/Reg/L2R/2022/Task1/'
    out_root = '/home/veronika/Out/miccai23/autoencoder/L2R2022-T1/'
elif user == 'KH':
    config_dir = '/vol/asklepios_ssd/users/hammerni/projects/tum/IDIR/scripts/autoencoder'
    data_dir = '/vol/prometheus2/Reg/L2R/2022/Task1/'
    out_root = '/vol/asklepios_ssd/users/hammerni/results/miccai23/autoencoder/L2R2022-T1'

info_file = f'{data_dir}/NLST/NLST_dataset.json'

# experiments = ['AEexp8-img'] 
# experiments = ['AEexp9-img'] 
# experiments = ['AEexp10-img'] 

experiments = ['AEexp1-KH-img']

for exp in experiments:
    # set seed
    np.random.seed(42)
    torch.manual_seed(42)

    config_file = f'{config_dir}/L2R2022-T1-{exp}.json' 
    progress_file = f'{config_dir}/Training-{exp}.png'
    save_folder = f'{out_root}/{exp}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(config_file, 'r') as f:
        config=json.load(f)

    use_mask = config["use_mask"] if "use_mask" in config else True 


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

    int_tfm = data_utils.intensity_transforms(**optn)
    train_tfm, val_tfm = data_utils.data_transforms(**optn)
    #TODO dataset
    dataset = NLSTDenoisingDataset.NLSTDataModule(data_dir=data_dir, config_file=config_file,
                                              int_transforms=int_tfm, 
                                              augm_transforms=train_tfm, val_augm_transforms=val_tfm)
    dataset.setup('fit')

    # print('Batches in train_loader', len(train_loader))
    trainloader = dataset.train_dataloader()
    print('Train loader size = {}'.format(len(trainloader)))
    valloader = dataset.val_dataloader()
    print('Val loader size = {}'.format(len(valloader)))

    # noisy_images, images, idx = dataset.nlst_train[0]
    # print(noisy_images.size())
    # print(images.size())
    # display_utils.display_3d([images[0,...], noisy_images[0,...]], pfile=f'{config_dir}/fixed_images.png')
    # display_utils.display_3d([images[1,...], noisy_images[1,...]], pfile=f'{config_dir}/moving_images.png')

    # for i, (noisy_patches, patches, idx) in enumerate(valloader):

    # continue

    # Get the model
    image_size = config["image_size"]
    network_config = config['network']
    model = autoencoder.AutoEncoder(input_shape=image_size,
                                    **network_config)    
    model.cuda()
    print(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config['lr_scheduler'])

    initial_epoch = 0
    best_psnr = 0
    best_epoch = 0

    # define the loss function
    loss_name = config["loss"] if "loss" in config else "MSE"
    if loss_name == "L1":
        loss_function = torch.nn.L1Loss(reduction = 'sum') 
    else:
        loss_function = torch.nn.MSELoss(reduction = 'sum') 
    def criterion(recon, target):
        return loss_function(recon, target) / recon.shape[0] # normalize wrt batch dim

    # training loop
    pbar = tqdm(range(initial_epoch, config['epochs']))

    accumulated_loss = np.zeros((config['epochs'], 2))
    accumulated_loss_val = np.zeros((config['epochs'], 2))

    plotter = utils.PlotAETrainingProgress(progress_file)
   
    for epoch in pbar:
        # train loop
        train_loss = 0.0
        train_psnr = 0.0


        """
        Training loop
        """
        model.train()

        batch_id = randint(0, len(trainloader) - 1)

        tin = time.time()
        for sidx, (inputs, outputs, _) in enumerate(trainloader):

            # print(inputs.size())
            noisy = inputs.permute(1,0,2,3,4).cuda()
            target = outputs.permute(1,0,2,3,4).cuda()
            # print(noisy.size())
            # print()
            # continue

            # print(noisy.size())
            # display_utils.display_3d([noisy[0,0,...], target[0,0,...]], pfile=f'{config_dir}/trainingFI.png')
            # display_utils.display_3d([noisy[1,0,...], target[1,0,...]], pfile=f'{config_dir}/trainingMI.png')

            recon = model(noisy)
            loss = criterion(recon, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(trainloader)
            train_psnr += psnr_criterion(recon, target).item() / len(trainloader)

            if sidx == batch_id:
                target_training = target[0,0,:,:,int(image_size[-1] / 2)].cpu().detach()
                noisy_training = noisy[0,0,:,:,int(image_size[-1] / 2)].cpu().detach()
                recon_training = recon[0,0,:,:,int(image_size[-1] / 2)].cpu().detach()

        elapsed_time = time.time() - tin

        accumulated_loss[epoch, 0] = train_loss
        accumulated_loss[epoch, 1] = train_psnr


        """
        Validation loop
        """

        val_loss = 0.0
        val_psnr = 0.0

        batch_id = randint(0, len(valloader) - 1)

        model.eval()
        for sidx, (inputs, outputs, _) in enumerate(valloader):

            with torch.no_grad():
                
                noisy = inputs.permute(1,0,2,3,4).cuda()
                target = outputs.permute(1,0,2,3,4).cuda()

                recon = model(noisy)
                loss = criterion(recon, target)
                val_loss += loss.item() / len(valloader)
                val_psnr += psnr_criterion(recon, target).item() / len(valloader)

                if sidx == batch_id:
                    target_validation = target[0,0,:,:,int(image_size[-1] / 2)].cpu().detach()
                    noisy_validation = noisy[0,0,:,:,int(image_size[-1] / 2)].cpu().detach()
                    recon_validation = recon[0,0,:,:,int(image_size[-1] / 2)].cpu().detach()

        accumulated_loss_val[epoch, 0] = val_loss
        accumulated_loss_val[epoch, 1] = val_psnr

        if progress_file is not None:
            plotter(target_training, noisy_training, recon_training, 
                    target_validation, noisy_validation, recon_validation,
                    accumulated_loss[:epoch+1,:], accumulated_loss_val[:epoch+1,:], epoch)
        
        
        #TODO save model checkpoint
        if epoch % 10 == 0:
            model_file = f'{save_folder}/checkpoint-epochs{epoch}.pt'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc_loss': accumulated_loss,
                'acc_loss_val': accumulated_loss_val,
                'config': config,
                'best_psnr' : best_psnr,
                'best_epoch' : best_epoch
                }, model_file)

        #TODO save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc_loss': accumulated_loss,
                'acc_loss_val': accumulated_loss_val,
                'config': config,
                'best_psnr' : best_psnr,
                'best_epoch' : best_epoch
                }, f'{save_folder}/model-best.pt')

        pbar.set_postfix({'train_psnr' : f'{train_psnr:4g}',
                          'val_psnr' : f'{val_psnr:4g}'})

        # lr scheduling
        scheduler.step()
