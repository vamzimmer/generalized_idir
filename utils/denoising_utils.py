import torch

def psnr_criterion(u, g):
    batch_size = u.shape[0]
    max_int = 1.0
    return torch.mean(20 * torch.log10(max_int / torch.sqrt(torch.mean((u.view(batch_size, -1) - g.view(batch_size, -1)) ** 2, -1))))
