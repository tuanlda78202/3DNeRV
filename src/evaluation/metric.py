import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


def psnr2(img1, img2):
    mse = (img1 - img2) ** 2
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr


def psnr_fn_single(output, gt):
    l2_loss = F.mse_loss(output.detach(), gt.detach(), reduction="none")
    psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
    return psnr.cpu()


def psnr_fn_batch(output_list, gt):
    psnr_list = [psnr_fn_single(output.detach(), gt.detach()) for output in output_list]
    return torch.stack(psnr_list, 0).cpu()


def psnr_fn(output_list, target_list):
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction="mean")
        psnr = -10 * torch.log10(l2_loss + 1e-9)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1)  # (batchsize, num_stage)
    return psnr


def msssim_fn_single(output, gt):
    msssim = ms_ssim(
        output.float().detach(), gt.detach(), data_range=1, size_average=False
    )
    return msssim.cpu()


def msssim_fn_batch(output_list, gt):
    msssim_list = [
        msssim_fn_single(output.detach(), gt.detach()) for output in output_list
    ]
    # for output in output_list:
    #     msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    #     msssim_list.append(msssim)
    return torch.stack(msssim_list, 0).cpu()


def msssim_fn(output_list, target_list):
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(
                output.float().detach(),
                target.detach(),
                data_range=1,
                size_average=True,
            )
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0)  # (num_stage)
    msssim = msssim.view(1, -1).expand(
        output_list[-1].size(0), -1
    )  # (batchsize, num_stage)
    return msssim
