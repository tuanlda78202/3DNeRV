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
from utils import load_yaml, state
import nnc

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(config):
    # Config
    logger = config.get_logger("test")

    name_wandb = "compress-" + str(config["trainer"]["name"])
    batch_size = config["dataloader"]["args"]["batch_size"]
    frame_interval = config["dataloader"]["args"]["frame_interval"]
    compress = config["compression"]

    # Dataset & DataLoader
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    full_model = config.init_obj("arch", module_arch).to(device)

    # Criterion & Metrics
    metrics = config.init_ftn("metrics", module_metric)

    # CKPT
    logger.info("Loading checkpoint: {} ...".format(config.resume))
    full_model.load_state_dict(torch.load(config.resume)["state_dict"])
    full_model.eval()

    # Encoder
    encoder_state = state(full_model, raw_decoder_path=compress["raw_decoder_path"])
    encoder_model = HNeRVMaeEncoder(HNeRVMae())
    encoder_model.load_state_dict(encoder_state)
    encoder_model.eval()

    # Compression
    nnc.compress_model(
        compress["raw_decoder_path"],
        bitstream_path=compress["stream_path"],
        qp=compress["qp"],
    )
    nnc.decompress_model(
        compress["stream_path"], model_path=compress["compressed_decoder_path"]
    )

    # Decoder Reconstruct
    decoder_model = HNeRVMaeDecoder(HNeRVMae())
    decoder_model.load_state_dict(torch.load(compress["compressed_decoder_path"]))
    decoder_model.eval()

    # Training
    wandb.init(project="nerv3d", entity="tuanlda78202", name=name_wandb, config=config)

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

            embedding = encoder_model(data)

            pred = decoder_model(embedding)

            data = data.permute(0, 2, 1, 3, 4)
            pred = pred.permute(0, 2, 1, 3, 4)

            psnr_batch = metrics(
                pred, data, batch_size=batch_size, frame_interval=frame_interval
            )

            data = torch.mul(data, 255).type(torch.uint8)
            pred = torch.mul(pred, 255).type(torch.uint8)

            # pred = pred.reshape(batch_size, frame_interval, 3, 720, 1280)
            # data = data.reshape(batch_size, frame_interval, 3, 720, 1280)

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
        default="beauty.yaml",  # check.yaml
        type=str,
        help="config file path (default: None)",
    )

    args.add_argument(
        "-r",
        "--resume",
        default="../ckpt/720p/beauty-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_ckpte300.pth",
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
