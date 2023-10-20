import torch

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim


def loss_fn(pred, target, loss_type="L2", batch_average=True):
    if loss_type == "L2":
        loss = F.mse_loss(pred, target)

    elif loss_type == "L1":
        loss = F.l1_loss(pred, target, reduction="none").flatten(1).mean(1)

    elif loss_type == "SSIM":
        loss = 1 - ssim(pred, target, data_range=1, size_average=False)

    elif loss_type == "L1-SSIM":
        B, T, C, H, W = pred.shape
        bf = pred.reshape(B * T, C, H, W)
        gt = target.reshape(B * T, C, H, W)

        loss = 0.7 * F.l1_loss(bf, gt, reduction="none").flatten(1).mean(1) + 0.3 * (
            1 - ssim(bf, gt, data_range=1, size_average=False)
        )

    elif loss_type == "Fusion1":
        loss = 0.3 * F.mse_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion2":
        loss = 0.3 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(1)
        +0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion3":
        loss = 0.5 * F.mse_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion4":
        loss = 0.5 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion5":
        loss = 0.7 * F.mse_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion6":
        loss = 0.7 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion7":
        loss = 0.7 * F.mse_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.3 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(1)

    elif loss_type == "Fusion8":
        loss = 0.5 * F.mse_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.5 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(1)

    elif loss_type == "Fusion9":
        loss = 0.9 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion10":
        loss = 0.7 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion11":
        loss = 0.9 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == "Fusion12":
        loss = 0.8 * F.l1_loss(pred, target, reduction="none").flatten(1).mean(
            1
        ) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))

    return loss.mean() if batch_average else loss
