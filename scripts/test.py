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
import src.model.hnerv3d as module_arch
import src.evaluation.metric as module_metric

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(config):
    logger = config.get_logger("test")

    # Config
    name_wandb = "infer-" + str(config["name"])
    batch_size = config["dataloader"]["args"]["batch_size"]
    frame_interval = config["dataloader"]["args"]["frame_interval"]

    # Dataset & DataLoader
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

    del dataset

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    model = config.init_obj("arch", module_arch).to(device)
    # model = torch.compile(model)  # Just for torch.compile() training

    # CKPT
    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    model.load_state_dict(checkpoint["state_dict"])

    # Criterion & Metrics
    metrics = config.init_ftn("metrics", module_metric)

    model.eval()

    wandb.init(project="nerv3d", entity="tuanlda78202", name=name_wandb, config=config)

    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Inference UVG",
        total=len(dataloader),
        unit="it",
    )

    with torch.no_grad():
        psnr_video = []
        for batch_idx, data in enumerate(tqdm_batch):
            data = data.permute(0, 4, 1, 2, 3).cuda()
            pred = model(data)

            # PSNR
            data = data.permute(0, 2, 1, 3, 4)
            psnr_batch = metrics(
                pred, data, batch_size=batch_size, frame_interval=frame_interval
            )

            data = torch.mul(data, 255).type(torch.uint8)
            pred = torch.mul(pred, 255).type(torch.uint8)

            pred = pred.reshape(batch_size, frame_interval, 3, 720, 1280)
            data = data.reshape(batch_size, frame_interval, 3, 720, 1280)

            pred = pred.cpu().detach().numpy()
            data = data.cpu().detach().numpy()

            tqdm_batch.set_postfix(psnr=psnr_batch)

            wandb.log(
                {
                    "psnr_batch": psnr_batch,
                    "pred": wandb.Video(pred, fps=4, format="mp4"),
                    "data": wandb.Video(data, fps=4, format="mp4"),
                },
            )

            psnr_video.append(psnr_batch)

            del pred, data

        wandb.log({"psnr_video": sum(psnr_video) / len(psnr_video)})

        wandb.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Inference with NeRV3D")
    args.add_argument(
        "-c",
        "--config",
        default=None,  # "config/uvg-720p/beauty-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml",
        type=str,
        help="config file path (default: None)",
    )

    args.add_argument(
        "-r",
        "--resume",
        default=None,  # "../ckpt/720p/beauty-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpte300.pth",
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)

    main(config)
