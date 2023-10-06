import sys
import os

sys.path.append(os.getcwd())
os.environ["WANDB_DIR"] = "./saved"

import argparse
import torch
import wandb
import numpy as np
from tqdm import tqdm
import src.dataset as module_data
import src.model as module_arch
from config.parse_config import ConfigParser
import src.dataset.build as module_data
import src.model.nerv3d as module_arch
import src.evaluation.metric as module_metric

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(config):
    # Config
    name_wandb = "infer-" + str(config["trainer"]["name"])
    batch_size = config["dataloader"]["args"]["batch_size"]
    frame_interval = config["dataloader"]["args"]["frame_interval"]

    # Dataset & DataLoader
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    model = config.init_obj("arch", module_arch).to(device)
    model.load_state_dict(torch.load(config.resume)["state_dict"])

    # Metrics
    psnr_metric = config.init_ftn("psnr", module_metric)
    msssim_metric = config.init_ftn("msssim", module_metric)

    model.eval()

    wandb.init(project="nerv3d", entity="tuanlda78202", name=name_wandb, config=config)

    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Inference UVG",
        total=len(dataloader),
        unit="it",
    )

    with torch.no_grad():
        psnr_video, msssim_video = [], []
        for batch_idx, data in enumerate(tqdm_batch):
            data = data.permute(0, 4, 1, 2, 3).cuda()
            pred = model(data)

            # PSNR & MS-SSIM
            data = data.permute(0, 2, 1, 3, 4)

            psnr_batch = psnr_metric(
                pred, data, batch_size=batch_size, frame_interval=frame_interval
            )

            msssim_batch = msssim_metric(pred, data, batch_size=batch_size)

            data = torch.mul(data, 255).type(torch.uint8)
            pred = torch.mul(pred, 255).type(torch.uint8)

            pred = pred.cpu().detach().numpy()
            data = data.cpu().detach().numpy()

            tqdm_batch.set_postfix(psnr=psnr_batch, msssim=msssim_batch)

            wandb.log(
                {
                    "psnr_batch": psnr_batch,
                    "msssim_batch": msssim_batch,
                    "pred": wandb.Video(pred, fps=4, format="mp4"),
                    "data": wandb.Video(data, fps=4, format="mp4"),
                },
            )

            psnr_video.append(psnr_batch)
            msssim_video.append(msssim_batch)

            # del pred, data

        wandb.log(
            {
                "psnr_video": sum(psnr_video) / len(psnr_video),
                "msssim_video": sum(msssim_video) / len(msssim_video),
            }
        )

        wandb.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Inference with NeRV3D")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    config = ConfigParser.from_args(args)

    main(config)
