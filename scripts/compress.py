import sys
import os

sys.path.append(os.getcwd())
os.environ["WANDB_DIR"] = "./saved"

import argparse
import torch
import wandb
import numpy as np
from tqdm import tqdm
import src.model as module_arch
from config.parse_config import ConfigParser
import src.dataset.build as module_data
import src.model.hnerv3d as module_arch
import src.evaluation.metric as module_metric
from src.model.hnerv3d import *
from src.compression.compression import *
from collections import defaultdict

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(config):
    logger = config.get_logger("test")

    NAME = "compress-" + str(config["trainer"]["name"])
    BS = config["dataloader"]["args"]["batch_size"]
    FI = config["dataloader"]["args"]["frame_interval"]
    IMG_SIZE = config["arch"]["args"]["img_size"]
    compress = config["compression"]

    # Dataset & DataLoader
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    full_model = config.init_obj("arch", module_arch).to(device)
    full_model.load_state_dict(torch.load(config.resume)["state_dict"])
    full_model.eval()

    # Metrics
    metrics = config.init_ftn("metrics", module_metric)

    # EED
    encoder_model = state(full_model, compress["raw_decoder_path"])

    embedding, embed_bit = embedding_compress(
        dataloader, encoder_model, compress["embedding_path"], compress["embed_qp"]
    )

    decoder_model, decoder_bit = dcabac_compress(
        compress["raw_decoder_path"],
        compress["stream_path"],
        compress["model_qp"],
        compress["compressed_decoder_path"],
    )

    # Training
    wandb.init(project="nerv3d", entity="tuanlda78202", name=NAME, config=config)

    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Compress UVG",
        total=len(dataloader),
        unit="it",
    )

    with torch.no_grad():
        psnr_video = []

        for batch_idx, data in enumerate(tqdm_batch):
            data = data.permute(0, 4, 1, 2, 3).cuda()

            embed = torch.from_numpy(embedding[str(batch_idx)]).cuda()
            pred = decoder_model(embed)

            data = data.permute(0, 2, 1, 3, 4)
            pred = pred.permute(0, 2, 1, 3, 4)

            psnr_batch = metrics(pred, data, batch_size=BS, frame_interval=FI)

            data = torch.mul(data, 255).type(torch.uint8)
            pred = torch.mul(pred, 255).type(torch.uint8)

            tqdm_batch.set_postfix(psnr=psnr_batch)

            wandb.log(
                {
                    "psnr_batch": psnr_batch,
                    "pred": wandb.Video(
                        pred.cpu().detach().numpy(), fps=4, format="mp4"
                    ),
                    "data": wandb.Video(
                        data.cpu().detach().numpy(), fps=4, format="mp4"
                    ),
                },
            )

            psnr_video.append(psnr_batch)

            del pred, data

        wandb.log(
            {
                "PSNR": sum(psnr_video) / len(psnr_video),
                "BPP": (embed_bit + decoder_bit)
                * 8
                / (len(dataset) * FI * IMG_SIZE[0] * IMG_SIZE[1]),
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
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)

    main(config)
