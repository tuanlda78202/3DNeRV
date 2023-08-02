import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def psnr_one(pred, gt):
    EPS = 1e-8
    mse = torch.mean((pred - gt) ** 2)
    db = -10 * torch.log10(mse + EPS)  # bcs normalized
    return db.item()


def psnr_batch(batch_pred, batch_gt, batch_size, frame_interval):
    psnr_list = []

    for batch_idx in range(batch_size):
        for fi_idx in range(frame_interval):
            psnr_list.append(
                psnr_one(batch_pred[batch_idx][fi_idx], batch_gt[batch_idx][fi_idx])
            )

    result = sum(psnr_list) / len(psnr_list)
    return round(result, 2)
