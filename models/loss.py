import torch
import torch.nn.functional as F


def PixelLoss(pred, target, mean, useAffine=True) :
    # Affine Result
    if useAffine :
        pred = pred*mean
        target= target*mean
    
    # Compute Loss
    loss = F.l1_loss(pred, target, reduction="none")
        
    return loss
