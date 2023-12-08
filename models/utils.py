import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


class PlotAAETrainingProgress(object):

    def __init__(self, out_file):

        self.out_file = out_file
        self.fig = None
        self.ax = []

        self._initialize_plot()

    """
        
    """
    def _initialize_plot(self):
        plt.ion()
        # f, a = plt.subplots(1, 3, figsize=(15, 6))
        self.fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = GridSpec(1, 4, figure=self.fig)

        gs00 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:2])
        self.ax.append(self.fig.add_subplot(gs00[0, 0]))
        self.ax.append(self.fig.add_subplot(gs00[0, 1]))
        self.ax.append(self.fig.add_subplot(gs00[1, 0]))
        self.ax.append(self.fig.add_subplot(gs00[1, 1]))

        self.ax.append(self.fig.add_subplot(gs[2]))  # total loss
        self.ax.append(self.fig.add_subplot(gs[3]))  # data loss
        # self.ax.append(self.fig.add_subplot(gs[4]))  # data loss


    """
       Call
    """
    def __call__(self, image_train, recon_train,
                 image_val, recon_val,
                 train_losses, val_losses, epoch,
                 isatend=False, show=False):
        
        self.ax[0].imshow(image_train, cmap='gray')
        self.ax[0].axis('off')
        self.ax[0].set_title('(Train) Image')

        self.ax[1].imshow(recon_train, cmap='gray')
        self.ax[1].axis('off')
        self.ax[1].set_title('(Train) Recon')

        self.ax[2].imshow(image_val, cmap='gray')
        self.ax[2].axis('off')
        self.ax[2].set_title('(Val) Image')

        self.ax[3].imshow(recon_val, cmap='gray')
        self.ax[3].axis('off')
        self.ax[3].set_title('(Val) Recon')

        self.ax[4].clear()
        self.ax[4].plot(train_losses[0], label='D loss')
        self.ax[4].plot(train_losses[1], label='G loss')
        self.ax[4].plot(train_losses[2], label='ZD loss')
        self.ax[4].plot(train_losses[3], label='E loss')
        self.ax[4].plot(train_losses[4], label='GE loss')
        self.ax[4].plot(val_losses[0], label='Val loss')
        self.ax[4].set_title(f"BCE losses (epoch: {epoch})")
        self.ax[4].set_ylabel("loss")
        self.ax[4].set_xlabel("epoch")
        self.ax[4].legend()

        # plt.draw()

        # self.ax[5].clear()
        # self.ax[5].plot(train_losses[5], label='training pl')
        # self.ax[5].plot(val_losses[1], label='validation pl')
        # self.ax[5].set_title(f"Perceptual lossess (epoch: {epoch})")
        # self.ax[5].set_ylabel("loss")
        # self.ax[5].set_xlabel("epoch")
        # self.ax[5].legend()

        plt.draw()

        self.ax[5].clear()
        self.ax[5].plot(val_losses[1], label='validation mse')
        self.ax[5].set_title(f"MSE lossess (epoch: {epoch})")
        self.ax[5].set_ylabel("loss")
        self.ax[5].set_xlabel("epoch")
        self.ax[5].legend()

        plt.draw()

        if show:
            plt.pause(0.001)
            plt.show()

        self.fig.savefig(self.out_file)

        if isatend:
            plt.ioff()
            self.fig.savefig(self.out_file)
            plt.close('all')

class PlotRegTestTrainingProgress(object):

    def __init__(self, out_file):

        self.out_file = out_file
        self.fig = None
        self.ax = []

        self._initialize_plot()

    """
        
    """
    def _initialize_plot(self):
        plt.ion()
        # f, a = plt.subplots(1, 3, figsize=(15, 6))
        self.fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = GridSpec(1, 4, figure=self.fig)
        self.ax.append(self.fig.add_subplot(gs[0, 0]))  # total loss
        self.ax.append(self.fig.add_subplot(gs[0, 1]))  # data loss
        self.ax.append(self.fig.add_subplot(gs[0, 2]))  # regulariser loss
        self.ax.append(self.fig.add_subplot(gs[0, 3]))  # landmark error


    """
       Call
    """
    def __call__(self, train_losses, val_losses, test_losses, epoch,
                 isatend=False, show=False):

        self.ax[0].clear()
        self.ax[0].plot(train_losses[:,0], label='total training loss')
        self.ax[0].plot(val_losses[:,0], label='total validation loss')
        self.ax[0].plot(test_losses[:,0], label='total test loss')
        # self.ax[0].set_title("Epoch: {} | total loss: {:.4f}".format(epoch,))
        self.ax[0].set_title(f"Epoch: {epoch}")
        self.ax[0].set_ylabel("loss")
        self.ax[0].set_xlabel("epoch")
        self.ax[0].legend()

        plt.draw()

        self.ax[1].clear()
        self.ax[1].plot(train_losses[:,1], label='data training loss')
        self.ax[1].plot(val_losses[:,1], label='data validation loss')
        self.ax[1].plot(test_losses[:,1], label='data test loss')
        # self.ax[0].set_title("Epoch: {} | total loss: {:.4f}".format(epoch,))
        self.ax[1].set_title(f"Epoch: {epoch}")
        self.ax[1].set_ylabel("loss")
        self.ax[1].set_xlabel("epoch")
        self.ax[1].legend()

        plt.draw()

        self.ax[2].clear()
        self.ax[2].plot(train_losses[:,2], label='regulariser training loss')
        self.ax[2].plot(val_losses[:,2], label='regulariser validation loss')
        self.ax[2].plot(test_losses[:,2], label='regulariser test loss')
        # self.ax[0].set_title("Epoch: {} | total loss: {:.4f}".format(epoch,))
        self.ax[2].set_title(f"Epoch: {epoch}")
        self.ax[2].set_ylabel("loss")
        self.ax[2].set_xlabel("epoch")
        self.ax[2].legend()

        plt.draw()

        self.ax[3].clear()
        self.ax[3].plot(test_losses[:,3], label='test landmark error')
        self.ax[3].set_title(f"Epoch: {epoch}")
        self.ax[3].set_ylabel("error")
        self.ax[3].set_xlabel("epoch")
        self.ax[3].legend()

        plt.draw()

        if show:
            plt.pause(0.001)
            plt.show()

        self.fig.savefig(self.out_file)

        if isatend:
            plt.ioff()
            self.fig.savefig(self.out_file)
            plt.close('all')


class PlotRegTrainingProgress(object):

    def __init__(self, out_file):

        self.out_file = out_file
        self.fig = None
        self.ax = []

        self._initialize_plot()

    """
        
    """
    def _initialize_plot(self):
        plt.ion()
        # f, a = plt.subplots(1, 3, figsize=(15, 6))
        self.fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = GridSpec(1, 3, figure=self.fig)
        self.ax.append(self.fig.add_subplot(gs[0, 0]))  # total loss
        self.ax.append(self.fig.add_subplot(gs[0, 1]))  # data loss
        self.ax.append(self.fig.add_subplot(gs[0, 2]))  # regulariser loss


    """
       Call
    """
    def __call__(self, train_losses, val_losses, epoch,
                 isatend=False, show=False):

        self.ax[0].clear()
        self.ax[0].plot(train_losses[:,0], label='total training loss')
        self.ax[0].plot(val_losses[:,0], label='total validation loss')
        # self.ax[0].set_title("Epoch: {} | total loss: {:.4f}".format(epoch,))
        self.ax[0].set_title(f"Epoch: {epoch}")
        self.ax[0].set_ylabel("loss")
        self.ax[0].set_xlabel("epoch")
        self.ax[0].legend()

        plt.draw()

        self.ax[1].clear()
        self.ax[1].plot(train_losses[:,1], label='data training loss')
        self.ax[1].plot(val_losses[:,1], label='data validation loss')
        # self.ax[0].set_title("Epoch: {} | total loss: {:.4f}".format(epoch,))
        self.ax[1].set_title(f"Epoch: {epoch}")
        self.ax[1].set_ylabel("loss")
        self.ax[1].set_xlabel("epoch")
        self.ax[1].legend()

        plt.draw()

        self.ax[2].clear()
        self.ax[2].plot(train_losses[:,2], label='regulariser training loss')
        self.ax[2].plot(val_losses[:,2], label='regulariser validation loss')
        # self.ax[0].set_title("Epoch: {} | total loss: {:.4f}".format(epoch,))
        self.ax[2].set_title(f"Epoch: {epoch}")
        self.ax[2].set_ylabel("loss")
        self.ax[2].set_xlabel("epoch")
        self.ax[2].legend()

        plt.draw()

        if show:
            plt.pause(0.001)
            plt.show()

        self.fig.savefig(self.out_file)

        if isatend:
            plt.ioff()
            self.fig.savefig(self.out_file)
            plt.close('all')


class PlotAETrainingProgress(object):

    def __init__(self, out_file):

        self.out_file = out_file
        self.fig = None
        self.ax = []

        self._initialize_plot()

    """
        
    """
    def _initialize_plot(self):
        plt.ion()
        # f, a = plt.subplots(1, 3, figsize=(15, 6))
        self.fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = GridSpec(1, 4, figure=self.fig)

        gs00 = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[:2])
        self.ax.append(self.fig.add_subplot(gs00[0, 0]))
        self.ax.append(self.fig.add_subplot(gs00[0, 1]))
        self.ax.append(self.fig.add_subplot(gs00[0, 2]))
        self.ax.append(self.fig.add_subplot(gs00[1, 0]))
        self.ax.append(self.fig.add_subplot(gs00[1, 1]))
        self.ax.append(self.fig.add_subplot(gs00[1, 2]))

        self.ax.append(self.fig.add_subplot(gs[2]))  # total loss
        self.ax.append(self.fig.add_subplot(gs[3]))  # data loss
        # self.ax.append(self.fig.add_subplot(gs[0, 2]))  # regulariser loss


    """
       Call
    """
    def __call__(self, patch_train, noisy_train, recon_train,
                 patch_val, noisy_val, recon_val,
                 train_losses, val_losses, epoch,
                 isatend=False, show=False):
        
        self.ax[0].imshow(patch_train, cmap='gray')
        self.ax[0].axis('off')
        self.ax[0].set_title('(Train) Patch')

        self.ax[1].imshow(noisy_train, cmap='gray')
        self.ax[1].axis('off')
        self.ax[1].set_title('(Train) Noisy')

        self.ax[2].imshow(recon_train, cmap='gray')
        self.ax[2].axis('off')
        self.ax[2].set_title('(Train) Recon')

        self.ax[3].imshow(patch_val, cmap='gray')
        self.ax[3].axis('off')
        self.ax[3].set_title('(Val) Patch')

        self.ax[4].imshow(noisy_val, cmap='gray')
        self.ax[4].axis('off')
        self.ax[4].set_title('(Val) Noisy')

        self.ax[5].imshow(recon_val, cmap='gray')
        self.ax[5].axis('off')
        self.ax[5].set_title('(Val) Recon')

        self.ax[6].clear()
        self.ax[6].plot(train_losses[:,0], label='training loss')
        self.ax[6].plot(val_losses[:,0], label='validation loss')
        self.ax[6].set_title(f"Loss (epoch: {epoch})")
        self.ax[6].set_ylabel("loss")
        self.ax[6].set_xlabel("epoch")
        self.ax[6].legend()

        plt.draw()

        self.ax[7].clear()
        self.ax[7].plot(train_losses[:,1], label='training psnr')
        self.ax[7].plot(val_losses[:,1], label='validation psnr')
        self.ax[7].set_title(f"PSNR (epoch: {epoch})")
        self.ax[7].set_ylabel("loss")
        self.ax[7].set_xlabel("epoch")
        self.ax[7].legend()


        plt.draw()

        if show:
            plt.pause(0.001)
            plt.show()

        self.fig.savefig(self.out_file)

        if isatend:
            plt.ioff()
            self.fig.savefig(self.out_file)
            plt.close('all')

class PlotTrainingProgress(object):

    def __init__(self, out_file):

        self.out_file = out_file
        self.fig = None
        self.ax = []

        self._initialize_plot()

    """
        
    """
    def _initialize_plot(self):
        plt.ion()
        # f, a = plt.subplots(1, 3, figsize=(15, 6))
        self.fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = GridSpec(1, 1, figure=self.fig)
        self.ax.append(self.fig.add_subplot(gs[0, 0]))  # loss


    """
       Call
    """
    def __call__(self, train_losses, total_losses, epoch,
                 isatend=False, show=False):

        self.ax[0].clear()
        self.ax[0].plot(train_losses, label='data training loss')
        self.ax[0].plot(total_losses, label='total training loss')
        # self.ax[0].set_title("Epoch: {} | total loss: {:.4f}".format(epoch,))
        self.ax[0].set_title(f"Epoch: {epoch}")
        self.ax[0].set_ylabel("loss")
        self.ax[0].set_xlabel("epoch")
        self.ax[0].legend()

        plt.draw()

        if show:
            plt.pause(0.001)
            plt.show()

        self.fig.savefig(self.out_file)

        if isatend:
            plt.ioff()
            self.fig.savefig(self.out_file)
            plt.close('all')