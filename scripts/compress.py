import sys
import os

sys.path.append(os.getcwd())
os.environ["WANDB_DIR"] = "./saved"

import collections
import argparse
import torch
import wandb
import numpy as np
from tqdm import tqdm
import src.model as module_arch
from config.parse_config import ConfigParser
import src.dataset.build as module_data
import src.model.nerv3d as module_arch
import src.evaluation.metric as module_metric
from src.model.nerv3d import *
from src.compression.utils import *
from utils.util import make_dir

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(config):
    logger = config.get_logger("test")

    # GLOBAL VARIABLES
    BS = config["dataloader"]["args"]["batch_size"]
    FI = config["dataloader"]["args"]["frame_interval"]
    IMG_SIZE = config["arch"]["args"]["img_size"]
    DECODER_DIM = config["arch"]["args"]["decode_dim"]
    MODE = config["trainer"]["mode"]
    DIR = config["compression"]["compress_dir"]
    compress = config["compression"]
    NAME = (
        "compress_qp" + str(compress["model_qp"]) + "_" + str(config["trainer"]["name"])
    )
    make_dir(DIR)

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
    psnr_metric = config.init_ftn("psnr", module_metric)
    msssim_metric = config.init_ftn("msssim", module_metric)

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
        decoder_dim=DECODER_DIM,
    )

    # Training
    wandb.init(
        project="nerv3d", entity="tuanlda78202", name=NAME, mode=MODE, config=config
    )

    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Compress UVG",
        total=len(dataloader),
        unit="it",
    )

    with torch.no_grad():
        psnr_video, msssim_video = [], []

        for batch_idx, data in enumerate(tqdm_batch):
            data = data.permute(0, 4, 1, 2, 3).cuda()

            embed = torch.from_numpy(embedding[str(batch_idx)]).cuda()
            pred = decoder_model(embed)

            data = data.permute(0, 2, 1, 3, 4)
            pred = pred.permute(0, 2, 1, 3, 4)

            psnr_batch = psnr_metric(pred, data, batch_size=BS, frame_interval=FI)
            msssim_batch = msssim_metric(pred, data, batch_size=BS)

            data = torch.mul(data, 255).type(torch.uint8)
            pred = torch.mul(pred, 255).type(torch.uint8)

            tqdm_batch.set_postfix(psnr=psnr_batch)

            wandb.log({"psnr_batch": psnr_batch, "msssim_batch": msssim_batch})

            psnr_video.append(psnr_batch)
            msssim_video.append(msssim_batch)

            del pred, data

        wandb.log(
            {
                "psnr": sum(psnr_video) / len(psnr_video),
                "msssim": sum(msssim_video) / len(msssim_video),
                "bpp": (embed_bit + decoder_bit)
                * 8
                / (len(dataset) * FI * IMG_SIZE[0] * IMG_SIZE[1]),
            }
        )

        wandb.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="NeRV3D Compression")

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

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    options = [
        CustomArgs(
            ["-m", "--mqp"],
            type=int,
            target="compression;model_qp",
        ),
        CustomArgs(
            ["-e", "--eqp"],
            type=int,
            target="compression;embed_qp",
        ),
    ]

    config = ConfigParser.from_args(args, options)

    main(config)
