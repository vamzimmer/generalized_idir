import numpy as np
import scipy.ndimage
from scipy.ndimage import map_coordinates
import math

from utils import metrics

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Taken from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Useful for monitoring current value and running average over iterations.
    """
    def __init__(self):
        self.reset()
        self.values = np.array([])
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = np.array([])
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values = np.append(self.values,val)
        self.std = np.std(self.values)


def compute_tre(fix_lms, mov_lms, spacing_fix, spacing_mov, disp=None, fix_lms_warped=None):
    
    if disp is not None:
        fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
        fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
        fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
        fix_lms_disp = np.array((fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)).transpose()

        fix_lms_warped = fix_lms + fix_lms_disp
        
    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)


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



def compute_dice(fixed,moving,moving_warped,labels):
    dice = []
    for i in labels:
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            dice.append(np.NAN)
        else:
            dice.append(metrics.compute_dice_coefficient((fixed==i), (moving_warped==i)))
    mean_dice = np.nanmean(dice)
    return mean_dice, dice


def compute_hd95(fixed,moving,moving_warped,labels):
    hd95 = []
    for i in labels:
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            hd95.append(np.NAN)
        else:
            hd95.append(metrics.compute_robust_hausdorff(metrics.compute_surface_distances((fixed==i), (moving_warped==i), np.ones(3)), 95.))
    mean_hd95 =  np.nanmean(hd95)
    return mean_hd95,hd95


def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet


def cohend(d1, d2):
    """
    Cohen’s d measures the difference between the mean from two Gaussian-distributed variables. 
    It is a standard score that summarizes the difference in terms of the number of standard deviations. 
    Because the score is standardized, there is a table for the interpretation of the result, summarized as:

        Small Effect Size: d=0.20
        Medium Effect Size: d=0.50
        Large Effect Size: d=0.80

    The Cohen’s d calculation is not provided in Python; we can calculate it manually.

    If Cohen's d is bigger than 1, the difference between the two means is larger than one standard deviation, 
    anything larger than 2 means that the difference is larger than two standard deviations.

    https://machinelearningmastery.com/effect-size-measures-in-python/
    """

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return abs((u1 - u2) / s)