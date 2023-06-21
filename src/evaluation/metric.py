import torch


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    # db = 20 * torch.log10(255.0 / torch.sqrt(mse))
    db = -10 * torch.log10(mse)  # bcs normalized
    return round(db.item(), 2)


def psnr_batch(batch_pred, batch_gt, bs, fi):
    psnr_list = []

    for batch_idx in range(bs):
        for fi_idx in range(fi):
            psnr_list.append(
                psnr(batch_pred[batch_idx][fi_idx], batch_gt[batch_idx][fi_idx])
            )

    return sum(psnr_list) / len(psnr_list)
