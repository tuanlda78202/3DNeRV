import sys
import os

sys.path.append(os.getcwd())
os.environ["WANDB_DIR"] = "./saved"

import torch
import wandb
import argparse
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
    # GLOBAL VARIABLES
    PROJECT = config["trainer"]["project"]
    NAME = "test-" + str(config["trainer"]["name"])
    MODE = config["trainer"]["mode"]
    BS = config["dataloader"]["args"]["batch_size"]
    FI = config["dataloader"]["args"]["frame_interval"]

    # Dataset & DataLoader
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

    # Model
    device = torch.device("cuda")
    model = config.init_obj(
        "arch", module_arch, frame_interval=FI, arch_mode="test"
    ).to(device)
    model.load_state_dict(torch.load(config.resume)["state_dict"])

    # Metrics
    psnr_metric = config.init_ftn(
        "psnr", module_metric, batch_size=BS, frame_interval=FI
    )
    msssim_metric = config.init_ftn("msssim", module_metric, batch_size=BS)

    wandb.init(
        project=PROJECT,
        entity="tuanlda78202",
        name=NAME,
        mode=MODE,
        config=config,
    )

    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="🥳 Inference {}: ".format(NAME),
        total=len(dataloader),
        unit="it",
    )

    model.eval()

    with torch.no_grad():
        test_psnr_video, test_msssim_video = [], []

        for batch_idx, data in enumerate(tqdm_batch):
            data = data.permute(0, 4, 1, 2, 3).cuda()
            pred = model(data)

            # PSNR & MS-SSIM
            data = data.permute(0, 2, 1, 3, 4)

            psnr_batch = psnr_metric(pred, data)
            msssim_batch = msssim_metric(pred, data)

            data = torch.mul(data, 255).type(torch.uint8)
            pred = torch.mul(pred, 255).type(torch.uint8)
            pred = pred.cpu().detach().numpy()
            data = data.cpu().detach().numpy()

            tqdm_batch.set_postfix(psnr=psnr_batch, msssim=msssim_batch)

            wandb.log(
                {
                    "PSNR Test": psnr_batch,
                    "MS-SSIM Test": msssim_batch,
                    "Prediction": wandb.Video(pred, fps=FI, format="mp4"),
                    "Ground Truth": wandb.Video(data, fps=FI, format="mp4"),
                },
            )

            test_psnr_video.append(psnr_batch)
            test_msssim_video.append(msssim_batch)

        wandb.log(
            {
                "Avg. PSNR Test": sum(test_psnr_video) / len(test_psnr_video),
                "Avg. MS-SSIM Test": sum(test_msssim_video) / len(test_msssim_video),
            }
        )

        print(
            "🎉 Avg. PSNR Test: {:.4f} dB & Avg. MS-SSIM Test: {:.4f}".format(
                sum(test_psnr_video) / len(test_psnr_video),
                sum(test_msssim_video) / len(test_msssim_video),
            )
        )

        wandb.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Testing NeRV3D")

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
